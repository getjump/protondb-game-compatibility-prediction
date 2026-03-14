# Phase 12: Contributor-aware модель

## Контекст

Phase 11 упёрся в потолок: F1 macro = 0.7234, works_oob F1 = 0.504. Альтернативные модели (CatBoost, XGBoost, HistGBM) — все хуже baseline. Bottleneck — label noise ~15-20% на границе tinkering↔works_oob.

**Новые данные:** ProtonDB Reports API (undocumented) даёт contributor data:
- `contributor_id` — уникальный ID автора отчёта
- `report_tally` — сколько всего отчётов написал этот автор
- `playtime` — общее время в Steam (минуты)
- `playtime_linux` — время в Steam на Linux (минуты)

**Текущее покрытие:** ~55% отчётов (API отдаёт макс. 40 последних отчётов на игру). Сбор продолжается.

**Ключевое открытие:** `contributor_id` связывает отчёты одного автора по разным играм — это именно то, чего не хватало для Dawid-Skene (Phase 9.3 провалился из-за анонимности).

---

## Phase 12.1 — Contributor features (1 день, +0.003-0.008 F1)

### Прямые фичи из contributor data

**Что:** Добавить в feature matrix 4 фичи из `report_contributors`:
- `contributor_tally` — сколько отчётов написал автор (опыт → точность)
- `contributor_playtime` — общее время в Steam (вовлечённость)
- `contributor_playtime_linux` — время на Linux (Linux experience)
- `contributor_linux_ratio` — `playtime_linux / playtime` (dedicated Linux user vs occasional)

**Гипотеза:**
- Опытные авторы (tally > 50) более calibrated: их tinkering действительно tinkering
- Casual авторы (tally = 1) чаще субъективны на границе tinkering↔oob
- Linux-dedicated users (high linux_ratio) лучше понимают, что "из коробки"

**Инференс:** Нет — contributor data доступна только в training time (мы не знаем, кто напишет отчёт заранее). Но это нормально: эти фичи нужны для denoising (sample weighting, noise estimation), не для prediction.

**Эффект:** Low-medium (+0.003-0.008 F1). Прямой сигнал слабый, но ценен для downstream (12.2, 12.3).

**Стоимость:** ~20 строк. LEFT JOIN на `report_contributors`, fillna(median).

**Ограничение:** ~55% покрытие. Для остальных — медиана или 0.

---

## Phase 12.2 — Dawid-Skene denoising (2-3 дня, +0.005-0.015 F1)

### Annotator reliability estimation

**Что:** Dawid-Skene EM-алгоритм для оценки reliability каждого contributor'а.

Phase 9.3 провалился потому что `contributor_id` был недоступен. Теперь он есть.

**Алгоритм:**
1. Построить confusion matrix π_j для каждого contributor j: P(label=k | true=l)
2. E-step: оценить P(true=l | labels, π)
3. M-step: обновить π_j
4. Итерировать до сходимости
5. Использовать estimated true labels вместо noisy raw labels

**Данные для DS:**
- ~3200 contributors с 2+ отчётами по разным играм (текущий сбор)
- После полного сбора: ~10-15K contributors с overlap
- DS работает лучше, когда один "item" (игра) оценен несколькими annotators → нужна группировка по (app_id, hardware_group)

**Проблема:** В отличие от классического DS, наши annotators не оценивают одну и ту же "задачу" — разные авторы играют на разном железе. Решение:

**Модификация DS для нашего случая:**
- "Item" = (app_id, gpu_family) — пара (игра, GPU)
- Если 3+ авторов оценили одну игру на похожем GPU, DS может оценить consensus
- Для пар с 1 автором — fallback на prior per-contributor reliability

**Два варианта использования:**
- (a) **Hard relabeling:** заменить raw label на DS-estimated label → retrain
- (b) **Soft weighting:** использовать DS-estimated P(true_label) как sample weight в LightGBM

**Инференс:** Нет — train-time only (denoising).
**Эффект:** Medium-high (+0.005-0.015 F1). Прямо атакует root cause.
**Стоимость:** Medium. ~80 строк DS + ~30 строк integration.

---

## Phase 12.3 — Confidence weighting (1 день, +0.003-0.008 F1)

### Sample weights по contributor reliability

**Что:** Вместо uniform sample weights — weight = f(contributor_quality):
- `report_tally ≥ 20` → weight 1.0 (experienced, calibrated)
- `report_tally 5-19` → weight 0.8
- `report_tally 1-4` → weight 0.5 (possibly noisy)
- no contributor data → weight 0.7 (unknown)

Или continuous: `weight = log1p(report_tally) / log1p(max_tally)` (capped 0.3..1.0).

**Гипотеза:** Авторы с tally=1 более вероятно ошибаются на границе tinkering↔oob. Их отчёты менее надёжны → меньший вес при обучении. Эффект аналогичен label smoothing, но per-sample.

**Комбинация с Phase 12.2:** DS даёт per-annotator reliability → более точные weights чем эвристика по tally. Но 12.3 можно применить и без DS (fallback).

**Инференс:** Нет (train-time only).
**Эффект:** Low-medium (+0.003-0.008 F1).
**Стоимость:** ~15 строк. LightGBM `sample_weight` параметр.

---

## Phase 12.4 — Annotator embeddings (2-3 дня, +0.005-0.010 F1)

### SVD embeddings для contributor'ов

**Что:** Построить co-occurrence matrix: contributors × games (значение = verdict score). SVD → contributor embeddings (аналогично текущим GPU/CPU/Game embeddings).

**Матрица:** `C[contributor_i, game_j] = verdict_score` (1=borked, 2=tinkering, 3=oob)
- SVD(C) → U (contributor vectors), Σ, V^T (game vectors)
- Contributor embedding = row of U·Σ (truncated to k dims)

**Что кодируют:**
- Группы contributors с похожим voting pattern (строгие vs лояльные)
- Latent "annotator style" — систематический bias (всегда tinkering vs всегда oob)

**Использование:**
- (a) Фича в LightGBM: `contributor_emb_0..k` (~8-16 dims)
- (b) Debiasing: вычесть contributor effect из prediction
- (c) Кластеризация annotators → per-cluster model (ensemble)

**Инференс:** Нет напрямую — но contributor embeddings можно использовать для debiasing game embeddings.
**Эффект:** Medium (+0.005-0.010 F1). Зависит от overlap density.
**Стоимость:** Medium. ~50 строк (аналогично существующим SVD embeddings).

**Ограничение:** Нужен достаточный overlap (contributor видел 2+ игры). Текущие данные: 2993 contributors с 2+ играми, но после полного сбора будет больше.

---

## Phase 12.5 — Annotator-aware label correction (2 дня, +0.005-0.012 F1)

### Адресная коррекция по annotator bias

**Что:** Для каждого contributor вычислить bias: `tinkering_ratio = count(tinkering) / count(tinkering + oob)`. Авторы с extreme bias (ratio > 0.9 или < 0.1 при 5+ отчётах) — систематически miscalibrated.

**Алгоритм:**
1. Для каждого contributor: вычислить personal tinkering_ratio
2. Вычислить global tinkering_ratio (по всей базе)
3. Если personal ratio сильно отличается от global (>2σ) — contributor biased
4. Для biased contributors на Stage 2: использовать soft relabeling с shrinkage к global prior

**Пример:** Contributor с 20 отчётами, 19 из них "tinkering" (ratio=0.95). Global ratio=0.55. Его tinkering отчёты — кандидаты на relabeling: P(true=tinkering) = shrunk estimate < 0.95.

**Инференс:** Нет (train-time denoising).
**Эффект:** Medium (+0.005-0.012 F1). Прямая атака на annotator bias — основной тип noise.
**Стоимость:** Low-medium. ~40 строк.

---

## Phase 12.6 — Self-training с contributor priors (2-3 дня, +0.005-0.010 F1)

### Iterative self-training усиленный contributor data

**Что:** R9 из Phase 11 (self-training), но с contributor-informed thresholds:
- High-tally contributors (≥20 reports): high threshold для relabeling (0.95) — доверяем их labels
- Low-tally contributors (1-3 reports): lower threshold (0.85) — агрессивнее корректируем
- No contributor data: middle threshold (0.90)

**Алгоритм:**
1. Train initial model
2. Find disagreements: model P > threshold vs raw label
3. Threshold varies per contributor reliability
4. Relabel + retrain
5. 3 раунда: thresholds 0.95→0.90→0.85 (для low-tally), cap 5% per round

**Гипотеза:** Uniform self-training (R9) рискует корректировать правильные labels опытных авторов. Contributor-aware thresholds защищают reliable labels и агрессивно корректируют unreliable.

**Инференс:** Нет.
**Эффект:** Medium (+0.005-0.010 F1). Сочетает Phase 12.2 (denoising) и R9 (self-training).
**Стоимость:** Medium. ~60 строк.

---

## Phase 12.7 — Бинарная задача + contributor-based tinkering/oob split (2 дня, +0.010-0.030 F1)

### R1 из Phase 11 усиленный contributor data

**Что:** Объединить R1 (binary: works vs borked) с contributor-based split:
1. Бинарная модель: borked vs works (устраняет boundary noise)
2. Для `works` — split на tinkering/oob используя:
   - (a) extracted_data.actions_json (rule-based, Phase 8) — если есть
   - (b) contributor bias (Phase 12.5) — для reports без actions
   - (c) Per-game consensus с weighting по contributor reliability

**Гипотеза:** R1 одна даёт +0.03-0.05 F1 за счёт устранения boundary noise. Contributor data улучшает tinkering/oob split: вместо rule-based heuristic — data-driven split по annotator consensus, weighted по reliability.

**Инференс:** Частично. Бинарная модель — да. Split rules — да (actions + game consensus).
**Эффект:** High (+0.010-0.030 F1). Лучшее из Phase 11 + contributor data.
**Стоимость:** Medium. ~50 строк.

---

## Phase 12.8 — Item Response Theory (3-4 дня, +0.010-0.025 F1)

### IRT для decomposition annotator effect и game difficulty

**Инсайт:** "Шум" на границе tinkering↔works_oob — не случайный noise, а систематическая heterogeneity in standards. Разные пользователи имеют разный порог "что считать tinkering". Это одномерная шкала строгости — идеальный случай для IRT.

**Модель (1PL / Rasch):**

Каждый contributor j имеет параметр строгости θ_j.
Каждая пара (game, hardware_group) i имеет параметр сложности d_i.

```
P(verdict = tinkering | item_i, contributor_j) = σ(θ_j − d_i)
```

- θ_j > 0: строгий автор (склонен ставить tinkering)
- θ_j < 0: лояльный автор (склонен ставить works_oob)
- d_i > 0: сложная пара (объективно ближе к tinkering)
- d_i < 0: простая пара (объективно ближе к works_oob)

**Почему IRT лучше Dawid-Skene:**
- DS моделирует annotator как confusion matrix (K×K параметров) — избыточно для 1 subjective boundary
- IRT: 1 параметр на annotator (θ) + 1 на item (d) — парсимоничнее, лучше generalization
- IRT даёт continuous difficulty score d_i — прямая фича для LightGBM
- DS collapse'ит annotator effect в discrete label → потеря информации

**Реализация:**

1. **Подготовка данных:**
   - Фильтр: только reports с verdict ∈ {tinkering, works_oob} И имеющие contributor_id
   - Item = (app_id, gpu_family) — группировка по игре и GPU семейству
   - Response: 1 = tinkering, 0 = works_oob
   - Минимум: item с 2+ annotators, annotator с 2+ items

2. **Оптимизация (EM или gradient descent):**
   ```python
   # PyTorch или scipy.optimize
   # Параметры: θ[n_contributors], d[n_items], + optional discrimination a[n_items]
   # Loss: binary cross-entropy
   # Regularization: N(0, 1) prior на θ и d

   logit = theta[contributor_idx] - d[item_idx]
   loss = BCE(sigmoid(logit), response) + λ * (θ² + d²)
   ```

   Альтернатива: `py-irt` library (Bayesian IRT с MCMC).

3. **Извлечение фичей:**
   - `game_irt_difficulty` = d_i — "объективная" сложность пары (game, gpu_family)
   - `contributor_strictness` = θ_j — для sample weighting (|θ| > 2 → extreme annotator → lower weight)

4. **Интеграция в pipeline:**
   - `game_irt_difficulty` → inference-time фича (per-game, по аналогии с game aggregates)
   - `contributor_strictness` → train-time sample weight: `w = 1 / (1 + |θ|)` — extreme annotators get downweighted
   - Soft relabeling: для annotator с θ=2 (очень строгий), его tinkering → P(true_tinkering) = σ(θ−d) adjusted

**Расширение до 2PL:**

```
P(tinkering) = σ(a_i * (θ_j − d_i))
```

Параметр `a_i` (discrimination) — насколько чётко item разделяет строгих и лояльных. Высокий a_i: "все согласны" (объективно tinkering или oob). Низкий a_i: "пограничный случай" (даже experts расходятся).

`a_i` → фича `game_irt_discrimination` = proxy для label uncertainty. Низкий a → conformal prediction set {tinkering, oob}.

**API output (Phase 12.8 + 12.7):**

```json
{
  "prediction": "works",
  "detail": {
    "difficulty": 0.3,
    "discrimination": 1.8,
    "strict_users_say": "tinkering",
    "casual_users_say": "works_oob",
    "note": "Требуется выбор GE-Proton. Опытные пользователи считают это тюнингом."
  }
}
```

**Инференс:** Частично:
- `game_irt_difficulty` — да (precomputed per game×gpu_family, аналогично game aggregates)
- `game_irt_discrimination` — да (precomputed)
- `contributor_strictness` — нет (train-time only, для weighting/relabeling)

**Эффект:** High (+0.010-0.025 F1).
- d_i — более чистый сигнал чем raw verdict aggregates (очищен от annotator effect)
- Sample weighting по |θ| — точнее чем эвристика по report_tally
- a_i → uncertainty estimation → лучше calibration

**Стоимость:** Medium. ~100 строк (IRT fitting + feature extraction).

**Требования к данным:**
- Items (game, gpu_family) с 3+ annotators — чем больше overlap, тем точнее d_i
- Contributors с 3+ items — чем больше, тем точнее θ_j
- При текущем 55% coverage (после полного сбора): ~5K contributors с 2+ games, ~2K items с 3+ annotators — достаточно для 1PL

---

## Результаты (2026-03-14)

### Прогон 1: IRT без сходимости

**Условия:** coverage 30.7% train / 54.5% test. IRT не сошёлся (inf в theta из-за log(0) в инициализации).

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| baseline | 0.7245 | — | 0.834 | 0.837 | 0.503 |
| 12.8a IRT features | 0.7412 | +0.017 | 0.838 | 0.848 | 0.538 |
| 12.8c IRT full | 0.7434 | +0.019 | 0.838 | 0.848 | 0.544 |
| 12.1+12.3+12.8 all | 0.7438 | +0.019 | 0.839 | 0.844 | 0.549 |

### Прогон 2: IRT с фиксом (аналитический градиент + правильная инициализация)

**Условия:** coverage 43.2% train / 65.8% test. IRT сошёлся за 0.3s, 111 итераций. 9020 contributors, 13951 items.

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| **baseline** | 0.7245 | — | 0.834 | 0.837 | 0.503 |
| 12.1 contrib features | 0.7232 | −0.001 | 0.837 | 0.834 | 0.498 |
| 12.3a weight log | 0.7202 | −0.004 | 0.834 | 0.834 | 0.493 |
| 12.3b weight bucket | 0.7196 | −0.005 | 0.834 | 0.835 | 0.490 |
| 12.1+12.3 combined | 0.7253 | +0.001 | 0.837 | 0.835 | 0.504 |
| **12.8a IRT features** | **0.7519** | **+0.027** | 0.837 | 0.856 | 0.563 |
| 12.8b IRT weights | 0.7210 | −0.004 | 0.834 | 0.833 | 0.496 |
| **12.8c IRT full** | **0.7536** | **+0.029** | 0.837 | 0.857 | 0.567 |
| **12.1+12.3+12.8 all** | **0.7545** | **+0.030** | 0.840 | 0.855 | 0.569 |

### Выводы

1. **IRT — прорыв: +0.030 F1** (0.7245 → 0.7545). Фикс инициализации дал дополнительные +0.011 (с 0.7438 до 0.7545). Аналитический градиент: сходимость за 0.3s вместо 111s.

2. **works_oob F1: 0.503 → 0.569** (+0.066) — самый большой прогресс за всю историю проекта. IRT difficulty score — объективная мера "сложности" пары (game, GPU), очищенная от annotator bias.

3. **tinkering F1: 0.837 → 0.857** (+0.020) — неожиданный бонус: IRT улучшает обе стороны Stage 2 boundary.

4. **12.1 (contributor features)** — нейтральный/отрицательный. report_tally и playtime не прямые сигналы.

5. **12.3 (sample weighting)** — отрицательный (−0.004..−0.005). Downweighting при неполном coverage вредит.

6. **IRT weights (12.8b)** — отрицательный (−0.004). LightGBM с label smoothing уже устойчив к extreme annotators.

7. **Комбинация 12.1+12.3+12.8** — лучший результат (0.7545), но прирост над 12.8c (0.7536) минимален (+0.001). IRT features — доминирующий фактор.

### Прогон 3: 2PL IRT (discrimination parameter)

**Условия:** те же. 2PL сошёлся за 1.3s, 261 итерация. NLL: 2086 (vs 4813 для 1PL). Discrimination `a`: mean=4.15, std=3.36, range=[0.05, 20.09].

| Experiment | F1 macro | ΔF1 vs 1PL | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| 1PL baseline | 0.7705 | — | 0.836 | 0.885 | 0.590 |
| **2PL features+relabel** | **0.7724** | **+0.002** | 0.838 | 0.887 | 0.593 |
| **2PL disc_aware 0.3** | **0.7734** | **+0.003** | 0.838 | 0.887 | 0.595 |
| 2PL disc_aware 0.5 | 0.7712 | +0.001 | 0.838 | 0.887 | 0.588 |
| 2PL disc_aware 0.8 | 0.7717 | +0.001 | 0.838 | 0.886 | 0.591 |
| 2PL disc_aware 1.0 | 0.7712 | +0.001 | 0.838 | 0.886 | 0.589 |

**Вывод:** 2PL даёт стабильные +0.002-0.003 F1 поверх 1PL. `irt_game_discrimination` — полезная фича. Disc-aware relabeling с min_disc=0.3 лучший вариант, но фильтрует всего 2 items — эффект в основном от features, не от relabeling.

### TODO

- [x] Пофиксить IRT инициализацию (clip initial theta, аналитический градиент)
- [x] Попробовать 2PL IRT (с discrimination parameter a_i) — +0.003 F1
- [x] Интегрировать IRT в основной pipeline (`ml/irt.py`, `train.py`)
- [ ] Перезапустить после полного сбора данных (~55% coverage)
- [x] Phase 13: IRT-informed relabeling (см. PLAN_ML_13.md) — +0.015 F1

---

## Порядок реализации

```
Phase 12.1 (1 день):   Contributor features                       → +0.003-0.008 F1
Phase 12.3 (1 день):   Confidence weighting (можно без 12.2)      → +0.003-0.008 F1
Phase 12.8 (3-4 дня):  IRT — главный эксперимент                  → +0.010-0.025 F1
Phase 12.7 (закрыт):   Binary + contributor split          — IRT решает boundary лучше
Phase 12.2 (закрыт):   Dawid-Skene                        — IRT строго лучше (1 param vs K×K)
Phase 12.5 (закрыт):   Annotator bias correction          — Phase 13.2 делает это через θ
Phase 12.6 → 13.4:     Self-training с IRT                 → перенесён в PLAN_ML_13
Phase 12.4 → 13.5:     Annotator embeddings (SVD)          → перенесён в PLAN_ML_13
```

**Приоритет:** 12.1 → 12.3 → 12.8 → 12.7 → 12.5 → 12.6 → 12.2 → 12.4

12.1 и 12.3 — быстрые wins, можно сделать сразу.
12.8 — IRT, главный эксперимент: математически чистое разделение annotator effect и game difficulty.
12.7 — binary task + contributor-informed split (комбинируется с 12.8).
12.2 — Dawid-Skene как альтернатива 12.8 (грубее, но проще; делать если IRT не даст результат).
12.4 — experimental, зависит от плотности overlap.

---

## Зависимости от данных

| Phase | Мин. покрытие | Текущее | Достаточно? |
|---|---|---|---|
| 12.1 (features) | 10%+ | ~54% (сбор идёт) | ✅ Да |
| 12.3 (weights) | 10%+ | ~54% | ✅ Да |
| 12.8 (IRT) | 25%+, 3K+ contributors с 3+ items | ~54%, 8.3K contributors | ✅ Да |
| 12.7 (binary + split) | 15%+ | ~54% | ✅ Да |
| 12.2 (Dawid-Skene) | 20%+, 5K+ contributors | ~54%, 55K unique | ✅ Да |
| 12.5 (bias correction) | 15%+, contributors с 5+ reports | ~54% | ✅ Да |
| 12.6 (self-training) | 10%+ | ~54% | ✅ Да |
| 12.4 (embeddings) | 30%+, 5K+ contributors с 2+ играми | ~54%, 18K c 2+ играми | ✅ Да |

**Сбор данных:** ~54% (16.8K/31K игр), ETA ~4 часа до полного покрытия. Максимум: ~55% (API лимит 40 reports/game). Все фазы разблокированы.

---

## Метрики

| Метрика | Phase 11 | Цель Phase 12 |
|---|---|---|
| F1 macro | 0.7234 | **> 0.75** |
| works_oob F1 | 0.504 | **> 0.55** |
| borked F1 | 0.825 | **≥ 0.83** |
| ECE | 0.012 | **< 0.010** |

## Ключевой вывод

Contributor data — единственный реалистичный путь к прорыву через label noise ceiling. Все предыдущие архитектурные и фичевые эксперименты (Phase 9: 14 попыток, Phase 11.2: 7 моделей) подтвердили: bottleneck не в модели, а в качестве labels. `contributor_id` позволяет:
1. **IRT (Phase 12.8)** — decompose verdict на annotator strictness (θ) и game difficulty (d). Difficulty — объективная фича, strictness — инструмент weighting. Математически чистое решение для одномерной subjective boundary.
2. Оценить reliability каждого annotator (Dawid-Skene — fallback если IRT не сработает)
3. Взвесить samples по надёжности (confidence weighting)
4. Адресно корректировать biased annotators (bias correction)
5. Усилить self-training per-contributor thresholds

Ключевой insight Phase 12: "шум" tinkering↔oob — не noise, а heterogeneity in annotator standards. IRT моделирует эту природу напрямую, извлекая объективный сигнал (d_i) из субъективных оценок.
