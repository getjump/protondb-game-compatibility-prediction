# Phase 13: Contributor-aware relabeling

## Контекст

Phase 12 показал: IRT даёт +0.019 F1 (0.7245 → 0.7438) даже при неполных данных. `irt_game_difficulty` — объективная мера "сложности" пары (game, GPU), очищенная от annotator bias.

Текущий pipeline имеет два этапа очистки labels:
1. **Rule-based relabeling (Phase 8):** tinkering → works_oob если `actions_json` не содержит effective actions (51% tinkering relabeled)
2. **Cleanlab noise removal (Phase 9.3):** удаление 3% "шумных" samples по confident learning

Оба этапа слепы к annotator identity. С contributor data можно сделать умнее.

---

## Phase 13.1 — IRT-informed relabeling (2 дня, +0.005-0.015 F1)

### Замена Cleanlab на IRT-based label correction

**Что:** Вместо Cleanlab (удаляет samples) — IRT-based коррекция (исправляет labels, сохраняя данные).

**Алгоритм:**

1. Fit IRT на tinkering/oob reports с contributor data
2. Для каждого report получить:
   - θ_j — строгость автора
   - d_i — объективная сложность (game, gpu_family)
   - P_irt = σ(θ_j − d_i) — IRT-предсказание P(tinkering)
3. Сравнить raw label с IRT prediction:
   - Если raw = tinkering, но P_irt < 0.3 (IRT говорит: даже строгий автор не поставил бы tinkering) → **relabel to works_oob**
   - Если raw = works_oob, но P_irt > 0.7 (IRT говорит: даже лояльный автор поставил бы tinkering) → **relabel to tinkering**
4. Порог зависит от confidence: relabel только если |θ| > 1 (extreme annotator) ИЛИ item имеет 3+ annotators (надёжная оценка d)

**Почему лучше Cleanlab:**
- Cleanlab удаляет noisy samples → потеря данных (3% = ~8K samples)
- IRT relabeling исправляет labels → сохраняет данные + улучшает качество
- Cleanlab не знает annotator identity → может удалить correct labels от строгих авторов
- IRT знает кто автор → адресная коррекция

**Ограничение:** Работает только для ~35-55% reports (с contributor data). Для остальных — fallback на Cleanlab.

**Инференс:** Нет (train-time only).
**Эффект:** Medium-high (+0.005-0.015 F1). Заменяет deletion на correction.
**Стоимость:** ~40 строк.

---

## Phase 13.2 — Contributor-aware rule-based relabeling (1 день, +0.003-0.008 F1)

### Улучшение Phase 8 relabeling с учётом annotator bias

**Что:** Текущий Phase 8 relabeling: tinkering → works_oob если нет effective actions в `actions_json`. Это грубо — игнорирует annotator context.

**Новая схема (3 уровня уверенности):**

**Level 1 — Hard relabel (высокая уверенность):**
- Условие: нет actions + contributor θ > 1.5 (очень строгий автор)
- Действие: tinkering → works_oob
- Логика: строгий автор поставил tinkering без actions = его "tinkering" — другим людям works_oob

**Level 2 — Soft relabel (средняя уверенность):**
- Условие: нет actions + contributor θ ∈ [0.5, 1.5] (умеренно строгий)
- Действие: label smoothing → y = 0.3 (вместо hard 0 или 1)
- Логика: может быть tinkering, может быть oob — мягкий label

**Level 3 — Keep original (нет данных для коррекции):**
- Условие: есть effective actions, ИЛИ contributor θ < 0.5, ИЛИ нет contributor data
- Действие: оставить raw label
- Логика: объективные действия подтверждают tinkering, или автор не строгий

**Эффект vs Phase 8:**
- Phase 8: бинарно relabels 51% tinkering → oob (слишком агрессивно)
- Phase 13.2: градуированный relabeling с учётом кто написал отчёт

**Инференс:** Нет.
**Эффект:** Low-medium (+0.003-0.008 F1).
**Стоимость:** ~30 строк (модификация существующего `apply_relabeling`).

---

## Phase 13.3 — Hybrid noise pipeline (2 дня, +0.008-0.020 F1)

### Объединение всех источников для label cleaning

**Что:** Единый pipeline замена текущих Phase 8 + Cleanlab:

```
Input: raw labels + contributor data + actions_json + IRT params
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
Has contributor   Has actions    Neither
    │               │               │
    ▼               ▼               ▼
IRT relabel     Rule-based      Cleanlab
(Phase 13.1)    (Phase 13.2)    (fallback)
    │               │               │
    └───────────────┼───────────────┘
                    ▼
            Merged clean labels
                    ▼
            Label smoothing α=0.15
            (Stage 2 training)
```

**Приоритет источников:**
1. **IRT + actions** (оба доступны): IRT relabel с actions как confirmation
2. **Только IRT** (есть contributor, нет actions): IRT relabel с conservative threshold
3. **Только actions** (нет contributor): rule-based relabel (Phase 8, без изменений)
4. **Ни то ни другое**: Cleanlab для удаления самых шумных

**Confidence score для каждого sample:**
```python
def label_confidence(report):
    if has_irt and has_actions:
        # IRT и actions agree → high confidence
        if irt_agrees_with_actions:
            return 1.0
        # Disagree → trust actions (objective signal)
        return 0.7
    elif has_irt:
        # IRT only → confidence depends on |theta| and item coverage
        return 0.5 + 0.3 * min(1, item_annotator_count / 5)
    elif has_actions:
        # Actions only → Phase 8 logic
        return 0.8 if has_effective_actions else 0.6
    else:
        # No signal → Cleanlab or default
        return 0.5
```

Confidence → sample weight в LightGBM: `weight = 0.3 + 0.7 * confidence`.

**Инференс:** Нет.
**Эффект:** High (+0.008-0.020 F1). Объединяет все denoising механизмы.
**Стоимость:** Medium. ~80 строк (новый модуль `ml/label_cleaning.py`).

---

## Phase 13.4 — Iterative IRT refinement (2-3 дня, +0.003-0.008 F1)

### Self-training loop с IRT

**Алгоритм:**
1. Fit IRT → relabel → train model
2. Используя model predictions, найти дополнительные mismatches
3. Re-fit IRT с обновлёнными labels
4. Повторить 2-3 раза

**Каждый раунд:**
- Раунд 1: IRT relabel (conservative, threshold 0.3/0.7)
- Раунд 2: model-guided relabel (threshold 0.85) + IRT re-fit
- Раунд 3: final model-guided (threshold 0.90) — cap 2% per round

**Контроль drift:** мониторить % relabeled per round. Если > 5% — порог слишком агрессивный.

**Инференс:** Нет.
**Эффект:** Low-medium (+0.003-0.008 F1). Diminishing returns после 2 раундов.
**Стоимость:** Medium. ~50 строк.

---

## Phase 13.5 — Annotator embeddings (2-3 дня, +0.003-0.008 F1)

### SVD на contributor×game matrix

**Что:** Построить co-occurrence matrix: contributors × games (значение = verdict score). SVD → contributor embeddings, которые ловят паттерны beyond one-dimensional strictness (θ).

**Матрица:** `C[contributor_i, game_j] = verdict_score` (0=works_oob, 1=tinkering)
- SVD(C) → U (contributor vectors), Σ, V^T (game vectors)
- Contributor embedding = row of U·Σ (truncated to k dims)

**Что кодируют:**
- Группы contributors с похожим voting pattern (например: "строгие к AAA но лояльные к инди")
- Latent biases, которые IRT θ (один скаляр) не ловит

**Использование:**
- Фича в LightGBM: `contributor_emb_0..k` (~8 dims) — train-time only
- Game embeddings из V^T — inference-time (per-game, аналогично текущим game SVD)

**Ограничение:** Матрица sparse. При 57K contributors и 31K games — нужен достаточный overlap. Текущие данные: 18K contributors с 2+ играми — может хватить для 8 dims.

**Инференс:** Game-side embeddings — да. Contributor-side — нет.
**Эффект:** Low-medium (+0.003-0.008 F1).
**Стоимость:** ~50 строк (аналогично существующим GPU/CPU SVD embeddings).

**Почему пробовать:** IRT θ — один скаляр "строгость". Реальные annotator biases многомерны: кто-то строг к FPS но лояльен к RPG, кто-то наоборот. SVD embeddings могут это поймать.

---

## Результаты (2026-03-14)

**Условия:** contributor coverage 43% train / 66% test. IRT: 9020 contributors, 13951 items. Все эксперименты включают IRT features (Phase 12.8).

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| **baseline** (Phase 8+Cleanlab+IRT) | 0.7566 | — | 0.837 | 0.856 | 0.578 |
| 13.1a IRT only (без Phase 8, без Cleanlab) | 0.7691 | +0.013 | 0.837 | 0.886 | 0.584 |
| 13.1b Phase 8 + IRT | 0.7571 | +0.001 | 0.837 | 0.859 | 0.575 |
| 13.1c IRT aggressive (0.4/0.6) | 0.7686 | +0.012 | 0.837 | 0.886 | 0.583 |
| **13.2 contributor-aware** | **0.7711** | **+0.015** | 0.837 | 0.886 | 0.591 |
| 13.3a hybrid | 0.7557 | −0.001 | 0.837 | 0.856 | 0.574 |
| 13.3b hybrid+cleanlab | 0.7579 | +0.001 | 0.837 | 0.857 | 0.580 |

### Cumulative progress (from original baseline)

| Stage | F1 macro | ΔF1 | works_oob F1 |
|---|---|---|---|
| Original baseline (Phase 11) | 0.7245 | — | 0.503 |
| + IRT features (Phase 12.8) | 0.7545 | +0.030 | 0.569 |
| + Contributor-aware relabel (Phase 13.2) | **0.7711** | **+0.047** | **0.591** |

### Выводы

1. **13.2 (contributor-aware relabel) — лучший: +0.015 F1** поверх IRT baseline. Graduated relabeling по θ (strict→hard relabel, moderate→soft, lenient→keep) лучше бинарного Phase 8.

2. **Phase 8 вредит при наличии IRT.** 13.1b (Phase 8 + IRT) почти нулевой эффект. Phase 8 слишком агрессивно relabels 51% tinkering → перетирает тонкую IRT коррекцию.

3. **IRT relabel один заменяет Phase 8 + Cleanlab.** 13.1a (IRT only) даёт +0.013 без Phase 8 и Cleanlab. tinkering F1: 0.856 → 0.886 (+0.030).

4. **Sample weights/confidence вредят** (13.3a/b). Как и в Phase 12.3 — LightGBM с label smoothing уже устойчив, дополнительное weighting ухудшает.

5. **Aggressive vs conservative IRT thresholds (0.4/0.6 vs 0.3/0.7)** — разницы нет. IRT relabel работает на оба варианта.

### TODO

- [ ] Интегрировать 13.2 в основной pipeline (заменить Phase 8 + Cleanlab)
- [ ] Перезапустить после полного сбора данных (~55% coverage)
- [ ] Phase 13.4: iterative refinement (IRT → relabel → retrain → re-fit IRT)

---

## Порядок реализации

```
Phase 13.2 (done):     Contributor-aware relabel          → +0.015 F1 ✅
Phase 13.1 (done):     IRT relabel                        → +0.013 F1 ✅
Phase 13.3 (done):     Hybrid pipeline                    → −0.001 F1 ❌
Phase 13.4 (done):      Iterative refinement               → +0.001 F1 (marginal)
Phase 13.5 (done):      Annotator embeddings (SVD)         → +0.002 F1 (borked +0.008!)
```

**Вывод: 13.2 — winner.** Заменяет Phase 8 + Cleanlab. Hybrid pipeline и confidence weights не помогают.

---

## Зависимости

- Phase 12.8 (IRT) — нужны θ и d параметры
- `report_contributors` таблица — contributor_id для связи с IRT
- `extracted_data.actions_json` — для rule-based relabeling
- Cleanlab — **больше не нужен** (13.2 заменяет)

---

## Метрики

| Метрика | Phase 12 | Цель Phase 13 | Факт Phase 13 |
|---|---|---|---|
| F1 macro | 0.7545 | > 0.76 | **0.7711 ✅** |
| works_oob F1 | 0.569 | > 0.58 | **0.591 ✅** |
| borked F1 | 0.840 | ≥ 0.84 | **0.837** |

## Ключевой вывод

IRT атакует label noise с двух сторон одновременно:
- **Feature side:** `irt_game_difficulty` как input (Phase 12.8, +0.030 F1)
- **Label side:** contributor-aware relabeling (Phase 13.2, +0.015 F1)

Эффекты аддитивны: **+0.047 F1 total** (0.7245 → 0.7711). Это больше чем все 14 экспериментов Phase 9 и все модельные эксперименты Phase 11 вместе взятые.
