# Phase 15: Temporal validity — состояние мира на момент отчёта

## Проблема

Отчёт "borked" от 2020 года может быть невалидным в 2024: Proton обновился, игра пропатчилась, DXVK/VKD3D улучшились. Модель обучается на устаревших labels как если бы они были вечно верными.

`report_age_days` (SHAP 1.10 — самая важная фича) — proxy для этого, но не causal: модель учит "старые отчёты = borked" вместо "старый Proton = borked". При инференсе на новом отчёте `report_age_days ≈ 0`, но модель не знает какой Proton.

**Ключевой insight:** мы не передаём в модель состояние software stack на момент отчёта. Proton version есть в данных, но закодирован как категория, а не как ordinal "поколение совместимости".

---

## Phase 15.1 — Proton version as ordinal (1 день, +0.005-0.015 F1)

### Числовое кодирование версии Proton

**Что:** Парсить `proton_version` и `custom_proton_version` в числовой score.

**Примеры парсинга:**
```
"Proton 8.0-2"           → 8.02
"Proton 9.0-3"           → 9.03
"Proton Experimental"    → 99.0  (всегда новейший)
"GE-Proton9-27"          → 9.27
"GE-Proton8-25"          → 8.25
"Proton 5.13-6"          → 5.136
"Proton 3.7-8"           → 3.78
None / ""                → NaN
```

**Фичи:**
- `proton_version_numeric` — числовой ordinal
- `proton_major_version` — integer major (5, 6, 7, 8, 9)
- `is_ge_proton` — binary: GE-Proton vs Valve Proton
- `is_proton_experimental` — binary: bleeding edge
- `proton_generation` — bucket: legacy(≤5), stable(6-7), modern(8-9), experimental(99)

**Гипотеза:** Proton 5.x → 9.x = 4 года прогресса. "borked" на Proton 5 ≠ "borked" на Proton 9. Числовая версия позволяет модели учить: "эта игра borked на старом Proton, works на новом" вместо "эта игра borked".

**Инференс:** Да — пользователь указывает версию Proton.
**Эффект:** Medium (+0.005-0.015 F1). Заменяет часть сигнала `report_age_days` causal'ным.
**Стоимость:** ~30 строк (regex parsing).

---

## Phase 15.2 — Time-decay sample weighting (1 день, +0.003-0.010 F1)

### Свежие отчёты весят больше

**Что:** sample_weight = time_decay(report_age):
```python
# Exponential decay: half-life = 2 years
weight = 0.3 + 0.7 * exp(-report_age_days * ln(2) / 730)
```
- Отчёт сегодня: weight = 1.0
- Год назад: weight = 0.79
- 2 года: weight = 0.65
- 4 года: weight = 0.48

**Комбинация с IRT:**
```python
final_weight = time_decay * irt_confidence
```
Старый отчёт от ненадёжного annotator → минимальный вес.

**Гипотеза:** Устаревшие "borked" отчёты — главный источник false negatives. Proton 9 починил тысячи игр, но старые отчёты тянут score вниз.

**Осторожность:** Phase 12.3 показал что sample weighting отрицательно при неполном coverage. Но time-decay применяется ко ВСЕМ отчётам (не зависит от contributor data), поэтому не имеет проблемы coverage.

**Инференс:** Нет (train-time only).
**Эффект:** Low-medium (+0.003-0.010 F1).
**Стоимость:** ~10 строк.

---

## Phase 15.3 — Game-temporal features (1-2 дня, +0.005-0.012 F1)

### Тренды совместимости per game

**Что:** Для каждого отчёта вычислить temporal context его игры:

**Фичи (per-report, computed from train data only):**
1. `game_verdict_trend` — linear regression slope verdict score по времени для этой игры
   - Положительный = игра улучшается со временем (Proton фиксы)
   - Отрицательный = игра ухудшается (breaking updates)
   - 0 = стабильно

2. `game_latest_verdict_score` — средний verdict score последних 5 отчётов на эту игру (в train set)
   - Более свежий сигнал чем общий avg_verdict_score

3. `game_proton_version_range` — разница между max и min proton_version_numeric для этой игры
   - Большой range = игра тестировалась на разных версиях Proton

4. `report_proton_vs_game_median` — proton_version отчёта vs медиана proton_version для игры
   - > 0: автор на более новом Proton чем типично для этой игры
   - < 0: автор на старом Proton

5. `game_has_recent_reports` — есть ли отчёты за последние 6 месяцев (binary)

**Leakage protection:** Все фичи вычисляются только из train set. При инференсе — из всех доступных отчётов (они уже в прошлом).

**Инференс:** Частично — trend и latest_verdict доступны из исторических отчётов.
**Эффект:** Medium (+0.005-0.012 F1).
**Стоимость:** Medium. ~60 строк (per-game temporal aggregation).

---

## Phase 15.3b — Per-game optimal Proton (1 день, +0.003-0.008 F1)

### Proton regressions и non-monotonность

**Проблема:** "Новее Proton = лучше" не всегда верно. Proton regressions реальны:
- Wine обновления ломают Win32 хаки старых игр
- DXVK/VKD3D изменения меняют shader behaviour
- Valve рекомендует конкретную версию, а не latest

**Фичи:**

1. `game_best_proton_version` — proton_version_numeric с максимальным % works среди отчётов
2. `report_proton_vs_best` = report.proton - game_best_proton
   - = 0: автор на оптимальной версии
   - > 0: автор на более новом (возможный regression)
   - < 0: автор на старом (недостаёт фиксов)
3. `report_proton_matches_recommended` (binary) — proton_version совпадает с PICS `recommended_runtime`
4. `game_has_proton_regression` (binary) — есть pattern works→borked при увеличении proton_version
5. `game_proton_stability` — variance verdict score across proton versions (высокая = нестабильная совместимость)

**Важно:** НЕ ставить monotonic constraint на `proton_version_numeric`. LightGBM сам найдёт non-monotonic splits: "Proton 7 works, Proton 8 borked, Proton 9 works".

**Leakage protection:** `game_best_proton` вычисляется только из train set.
**Инференс:** Да — best_proton и recommended_runtime доступны per-game.
**Эффект:** Low-medium (+0.003-0.008 F1).
**Стоимость:** ~30 строк.

---

## Phase 15.4 — Proton-epoch interaction features (1 день, +0.003-0.008 F1)

### Взаимодействие версии Proton с game/hardware

**Что:** Interaction features между proton_version и другими:

1. `proton_gen × dx_version` — Proton 9 + DX12 vs Proton 5 + DX12 (VKD3D прогресс)
2. `proton_gen × has_anticheat` — античит на новом Proton может работать (EAC поддержка с Proton 7+)
3. `proton_gen × is_steam_deck` — Deck = всегда свежий Proton
4. `proton_gen × game_age` — новая игра + новый Proton vs старая игра + старый Proton

**Гипотеза:** LightGBM может найти эти interactions сам, но explicit features ускоряют обучение и уменьшают требования к depth.

**Инференс:** Да.
**Эффект:** Low (+0.003-0.008 F1).
**Стоимость:** ~20 строк.

---

## Phase 15.5 — Temporal label correction (2 дня, +0.005-0.015 F1)

### Коррекция устаревших labels

**Что:** Для каждой игры, если есть и старые и новые отчёты:
- Старый "borked" + новые "works" → relabel старый к "works" (Proton починил)
- Старый "works" + новые "borked" → оставить оба (regression)

**Алгоритм:**
1. Для каждой game × gpu_family, отсортировать отчёты по времени
2. Если последние 3+ отчёта unanimous "works" (tinkering или oob), а старые "borked":
   - Relabel старые borked → works (если proton_version старых < proton_version новых)
3. Обратное (works → borked) НЕ делать — regressions реальны

**Комбинация с IRT:**
- IRT difficulty на "современных" отчётах будет точнее
- Старые relabeled отчёты получают lower IRT confidence

**Инференс:** Нет (train-time denoising).
**Эффект:** Medium (+0.005-0.015 F1). Прямо атакует temporal label staleness.
**Стоимость:** ~50 строк.

---

## Порядок реализации

```
Phase 15.1 (1 день):    Proton version numeric            → +0.005-0.015 F1
Phase 15.6 (1-2 дня):   Proton × Game SVD embeddings       → +0.005-0.015 F1
Phase 15.3 (1-2 дня):   Game-temporal features             → +0.005-0.012 F1
Phase 15.3b (1 день):   Per-game optimal Proton            → +0.003-0.008 F1
Phase 15.5 (2 дня):     Temporal label correction          → +0.005-0.015 F1
Phase 15.7 (2-3 дня):   FM multi-way interactions           → +0.008-0.020 F1
Phase 15.2 (1 день):    Time-decay weighting               → +0.003-0.010 F1
Phase 15.4 (1 день):    Interaction features               → +0.003-0.008 F1
                                                     Итого: +0.015-0.050 F1
```

**Приоритет:** 15.1 → 15.3 → 15.3b → 15.5 → 15.2 → 15.4

15.1 — быстрый win, causal замена report_age_days.
15.3 + 15.3b — game-level temporal context + regression detection.
15.5 — label correction для устаревших borked.

---

## Phase 15.6 — Proton × Game SVD embeddings (1-2 дня, +0.005-0.015 F1)

### Learned interactions через matrix factorization

**Что:** SVD decomposition матрицы совместимости Proton × Game.

**Матрица:** `M[proton_version_i, game_j]` = avg verdict score
- Rows: proton versions (major buckets: 3-4, 5, 6, 7, 8, 9, experimental, GE variants)
- Cols: app_ids (31K игр)
- Значение: средний verdict score всех отчётов с данным Proton на данной игре (0=borked, 1=tinkering, 2=oob)
- NaN для unseen пар

**SVD → embeddings:**
- U·Σ → Proton embeddings (8-dim): "профиль совместимости этой версии Proton"
- V^T → Game embeddings в Proton-space (8-dim): "как эта игра зависит от версии Proton"

**Что кодируют:**
- Proton 5 и Proton 6 близки в embedding space → похожий набор поддерживаемых игр
- Game A и Game B близки → одинаково реагируют на обновления Proton
- Dot product Proton_emb × Game_emb ≈ predicted compatibility

**Фичи (per-report):**
- `proton_emb_0..7` — embedding версии Proton из отчёта (inference: пользователь указывает Proton)
- `game_proton_emb_0..7` — embedding игры в Proton-space (inference: per-game, precomputed)
- `proton_game_dot` — dot product (single scalar predicted compatibility)

**Преимущество перед manual interactions (Phase 15.4):**
- Learned, не hand-crafted
- Ловит неочевидные паттерны: "Proton 8.0-2 хорошо работает с играми типа X, плохо с Y"
- Аналогично GPU×Game SVD (уже даёт +0.01 F1), но в другом пространстве

**Sparse handling:**
- Proton versions → bucket by major (5, 6, 7, 8, 9, exp) + GE vs Valve split ≈ 12-15 rows
- Каждая row имеет тысячи games → достаточно dense для SVD
- Missing values → fill with global mean

**Инференс:** Да — пользователь указывает proton_version → embedding, game_proton_emb precomputed.
**Эффект:** Medium (+0.005-0.015 F1). Новый тип embedding, orthogonal к GPU×Game SVD.
**Стоимость:** ~40 строк (аналогично существующим SVD embeddings в `features/embeddings.py`).

---

## Phase 15.7 — Factorization Machines для multi-way interactions (2-3 дня, +0.008-0.020 F1)

### Learned embeddings: game × proton × gpu × driver → compatibility

**Проблема:** SVD работает для 2D (GPU×Game, Proton×Game). Реальная совместимость — multi-way interaction: одна и та же игра может работать на GPU_A + Proton_9 но быть borked на GPU_B + Proton_8. Текущие SVD embeddings не ловят эти cross-interactions.

**Factorization Machines (FM):**

```
ŷ = w₀ + Σ wᵢxᵢ + Σᵢ Σⱼ>ᵢ <vᵢ, vⱼ> xᵢxⱼ
```

Каждый categorical value (game_id, proton_version, gpu_family, mesa_version, kernel_major) получает k-dim embedding vector v. Все pairwise interactions моделируются через dot products `<vᵢ, vⱼ>`. Complexity O(nk) вместо O(n²).

**Input features для FM (one-hot encoded):**
- `app_id` (31K values) → 8-dim embedding
- `proton_version_bucket` (~15 values) → 4-dim embedding
- `gpu_family` (~100 values) → 4-dim embedding
- `driver_version_bucket` (mesa/nvidia major, ~20 values) → 4-dim embedding
- `kernel_major` (~10 values) → 2-dim embedding
- Optional: `engine`, `dx_version`, `anticheat`

**Target:** verdict score (0=borked, 1=tinkering, 2=oob) или binary (borked vs works)

**Что FM кодирует автоматически:**
- game_A + proton_9 = works, game_A + proton_5 = borked (Proton-specific fix)
- game_B + nvidia = works, game_B + amd = borked (driver-specific issue)
- game_C + proton_8 + nvidia = works, game_C + proton_8 + amd = borked (3-way interaction через pairwise)
- Proton regressions: game_D + proton_7 = works, game_D + proton_8 = borked

**Использование в pipeline (stacking):**

```python
# Step 1: Train FM on (game, proton, gpu, driver, ...) → verdict
fm_model = FactorizationMachine(k=8, ...)
fm_model.fit(X_categorical, y_verdict)

# Step 2: FM predictions → single feature in LightGBM
X_train["fm_compatibility_score"] = fm_model.predict(X_train_categorical)
X_test["fm_compatibility_score"] = fm_model.predict(X_test_categorical)

# Step 3: Also extract learned embeddings as features
X_train["fm_game_emb_0..7"] = fm_model.get_embedding("app_id", X_train["app_id"])
X_train["fm_proton_emb_0..3"] = fm_model.get_embedding("proton_bucket", X_train["proton_bucket"])
```

**FM output features для LightGBM:**
- `fm_compatibility_score` — predicted compatibility (single scalar, encodes all interactions)
- `fm_game_emb_0..7` — learned game embedding (inference-time, per-game)
- `fm_proton_emb_0..3` — learned proton embedding (inference-time, per-version)
- `fm_gpu_emb_0..3` — learned gpu embedding (inference-time, per-family)

**Leakage protection:**
- FM обучается на train set only
- При инференсе: embeddings precomputed для known entities, OOV → zero vector

**Библиотеки:**
- `xlearn` — C++ FM implementation, fast
- `pytorch` custom — full control, ~50 строк
- `lightfm` — designed for recommendations, good API

**Преимущество перед отдельными SVD:**
- Одна модель вместо 3+ отдельных SVD (GPU×Game, CPU×Game, Proton×Game)
- Автоматические cross-interactions (GPU×Proton, Game×Driver, etc.)
- Scalable — O(nk) для n categorical fields

**Инференс:** Да — все embeddings per-entity, precomputed.
**Эффект:** Medium-high (+0.008-0.020 F1). Заменяет/дополняет все SVD embeddings + добавляет cross-interactions.
**Стоимость:** Medium. ~80 строк (FM training + feature extraction).

---

## Результаты (2026-03-14)

**Все эксперименты отрицательные или нейтральные.**

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| baseline (IRT) | 0.7702 | — | 0.838 | 0.886 | 0.587 |
| 15.1 proton numeric | 0.7668 | −0.003 | 0.839 | 0.886 | 0.575 |
| 15.2 time decay | 0.7702 | 0.000 | 0.838 | 0.886 | 0.587 |
| 15.3 game temporal | 0.7599 | −0.010 | 0.838 | 0.883 | 0.560 |
| 15.3b optimal proton | 0.7632 | −0.007 | 0.839 | 0.886 | 0.565 |
| 15.5 temporal correction | 0.7706 | +0.000 | 0.838 | 0.886 | 0.588 |
| 15.6 proton×game SVD | 0.7531 | −0.017 | 0.826 | 0.883 | 0.551 |
| 15.7 FM | 0.7574 | −0.013 | 0.833 | 0.884 | 0.555 |
| combined | 0.7351 | −0.035 | 0.807 | 0.877 | 0.522 |

### Выводы

1. **`report_age_days` уже оптимально кодирует temporal signal.** LightGBM через `report_age_days` + game aggregates неявно учитывает temporal dynamics. Explicit proton version features redundant или noisy.

2. **Game-temporal features вносят leakage или шум.** `game_verdict_trend` и `game_latest_verdict_score` слишком коррелируют с target (circular dependency: predict verdict from nearby verdicts).

3. **Proton×Game SVD слишком sparse.** 12 proton buckets × 31K games — большинство ячеек пустые. SVD подгоняется к noise.

4. **FM underfitting.** 20 epochs SGD мало, но даже при convergence FM score будет redundant с IRT difficulty.

5. **Temporal label correction нейтральна.** Pattern "old borked + new works + proton upgrade" слишком редок для значимого эффекта.

**Статус: Phase 15 закрыт. Нет ML value. `report_age_days` — достаточный temporal signal.**

---

## Зависимости

- `proton_version`, `custom_proton_version` — уже в таблице `reports`
- `report_age_days` — уже feature, Phase 15.1 дополняет его
- IRT (Phase 12.8) — для комбинации с time-decay
- Game aggregates (Phase 9.2) — 15.3 расширяет их temporal dimension

---

## Метрики

| Метрика | Phase 14 (expected) | Цель Phase 15 |
|---|---|---|
| F1 macro | ~0.78 | **> 0.79** |
| works_oob F1 | ~0.60 | **> 0.62** |
| borked F1 | ~0.84 | **> 0.85** |

## Ключевой вывод

Phases 12-14 улучшали модель через новые features и label denoising. Phase 15 атакует фундаментальную проблему: **временная валидность labels**. Отчёт — snapshot совместимости на конкретный момент (Proton X, game version Y, driver Z). Модель должна знать этот контекст, чтобы отличить "сломано навсегда" от "было сломано, починили".

`proton_version_numeric` — causal замена `report_age_days`. Вместо "старый отчёт = менее надёжный" → "старый Proton = хуже совместимость". Это правильная генерализация: при инференсе пользователь указывает свою версию Proton.
