# Phase 11: Прорыв через постановку задачи и качество данных

## Контекст

После Phase 9 (14 экспериментов в 9.4–9.5 — все отрицательные) и запланированного Phase 10 (label quality):
- F1 macro: 0.7253, works_oob F1: 0.508
- 119 фичей, cascade LightGBM (Stage 1: borked/works, Stage 2: tinkering/oob)
- **Потолок**: фичевые и архитектурные улучшения исчерпаны. Bottleneck — label noise (~15–20%) на границе tinkering↔works_oob

Два направления:
- (A) Переформулировка задачи — убрать субъективную границу tinkering↔oob
- (B) Альтернативные модели и техники — другой inductive bias

Ожидаемый суммарный эффект: **+0.02–0.05 F1** (поверх Phase 10).

---

## Phase 11.1 — Переформулировка задачи (2–3 дня, потенциально +0.03–0.05 F1)

### R1. Бинарная задача: works vs borked

**Что:** Объединить tinkering + works_oob → `works`. Одна модель, binary classification.

**Гипотеза:** 80% ошибок cascade — на границе tinkering↔works_oob. Эта граница субъективна: один пользователь считает выбор GE-Proton "tinkering", другой — "works out of the box". Бинарная задача устраняет этот шум. Borked F1 уже 0.825 — бинарная модель будет стабильнее.

**Разделение tinkering/oob:** После бинарной предикции `works` — rule-based split на основе extracted_data (есть ли effective actions в `actions_json`). Это чище ML-предикции, т.к. actions — объективный сигнал.

**Инференс:** ✅ Да.
**Эффект:** **High (+0.03–0.05 F1 macro)**. Устраняет основной источник ошибок.
**Стоимость:** Низкая. Упрощение pipeline. ~30 строк.

### R2. Regression на ProtonDB tier/score

**Что:** Вместо classification — regression на `protondb_score` (continuous) или ordinal на `protondb_tier` (borked→platinum, 5 уровней). Target = per-game community consensus, а не per-report субъективный verdict.

**Гипотеза:** ProtonDB tier — агрегат тысяч отчётов, noise усреднён. Per-report verdict — single noisy observation. Предсказание tier = предсказание того, что community в целом думает об игре. Tier доступен для 20K+ игр.

**Проблема:** Tier — per-game, а не per-(game, hardware). Одна и та же игра может быть platinum на NVIDIA и silver на AMD. Решение: предсказывать residual `actual_verdict − predicted_tier` или использовать tier как baseline + hardware adjustment.

**Инференс:** ✅ Да — tier публичен.
**Эффект:** Medium-high (+0.01–0.03 F1). Зависит от покрытия tier данных.
**Стоимость:** Medium. Смена target + evaluation. ~50 строк.

### R3. Предсказание P(works) с conformal prediction

**Что:** Вместо hard class — калиброванная вероятность P(works). Для неопределённых случаев — conformal prediction set {tinkering, works_oob} с гарантированным покрытием (1−α).

**Гипотеза:** Субъективная граница tinkering↔oob — не ошибка модели, а genuine ambiguity. Честный ответ для borderline игры: "вероятно работает, может потребоваться настройка" с confidence interval, а не forced hard decision.

**Инференс:** ✅ Да — `mapie` или manual conformal.
**Эффект:** Medium. Не улучшает F1 напрямую, но улучшает UX и калибровку.
**Стоимость:** Низкая. `pip install mapie`, ~20 строк.

---

## Phase 11.2 — Альтернативные модели (2–3 дня, +0.003–0.010 F1)

### R4. CatBoost для Stage 2

**Что:** Drop-in замена LightGBM → CatBoost для Stage 2 (tinkering vs works_oob).

**Гипотеза:** CatBoost имеет:
- **Ordered boosting** — каждое дерево обучается на случайной перестановке данных, что даёт implicit regularization. При label noise (~15–20%) это критично: LightGBM фитит noisy samples, CatBoost — меньше.
- **Нативные categorical features** — gpu_family (100 категорий), engine, developer обрабатываются через ordered target statistics без label encoding. Устраняет information loss от frequency encoding.
- **Symmetric trees** — другой inductive bias, может найти паттерны которые LightGBM пропускает.

**Инференс:** ✅ Да — те же фичи.
**Эффект:** Medium (+0.003–0.010 F1). Ordered boosting — главное преимущество для noisy data.
**Стоимость:** Низкая. `pip install catboost`, ~20 строк. Drop-in API.

### R5. XGBoost + HistGradientBoosting

**Что:** Протестировать XGBoost (gpu_hist) и sklearn HistGradientBoostingClassifier как альтернативы Stage 2.

**Гипотеза:** Разные алгоритмы сплитов → разные decision boundaries. HistGradientBoosting поддерживает `interaction_constraints` и `monotonic_cst` — можно задать domain knowledge (больше RAM → не хуже, новее kernel → не хуже).

**Инференс:** ✅ Да.
**Эффект:** Low-medium (+0.002–0.005 F1).
**Стоимость:** Низкая. ~15 строк каждый.

### Результаты Phase 11.2

**Дата:** 2026-03-14

**Все альтернативные модели показали отрицательный эффект. LightGBM остаётся лучшим.**

| Experiment | F1 eval | ΔF1 | borked | tinkering | works_oob | accuracy | time |
|---|---|---|---|---|---|---|---|
| **LightGBM (baseline)** | **0.7234** | — | 0.825 | 0.841 | 0.504 | 0.7747 | 60s |
| CatBoost (default) | 0.7185 | −0.005 | 0.825 | 0.839 | 0.491 | 0.7718 | 52s |
| CatBoost (noise_robust) | 0.7192 | −0.004 | 0.825 | 0.838 | 0.494 | 0.7710 | 50s |
| CatBoost (deeper) | 0.7185 | −0.005 | 0.825 | 0.838 | 0.492 | 0.7704 | 58s |
| CatBoost (langevin) | 0.7179 | −0.006 | 0.825 | 0.833 | 0.496 | 0.7649 | 30s |
| XGBoost (default) | 0.7218 | −0.002 | 0.825 | 0.842 | 0.498 | 0.7750 | 69s |
| XGBoost (deep) | 0.7205 | −0.003 | 0.825 | 0.840 | 0.496 | 0.7728 | 36s |
| HistGBM (default) | 0.7137 | −0.010 | 0.825 | 0.840 | 0.476 | 0.7711 | 37s |

#### Выводы

1. **CatBoost (все варианты)** — хуже на −0.004..−0.006. Ordered boosting не помогает: ProtonDB label noise не annotator-specific (анонимные отчёты), а task-inherent (субъективность границы tinkering↔oob). Ordered target statistics для categoricals не лучше LightGBM frequency encoding на наших данных.

2. **XGBoost default** — ближе всего к baseline (−0.002), но всё равно хуже. Hist-based splitting XGBoost и LightGBM концептуально близки — ожидаемый результат.

3. **XGBoost deep** (depth=8, max_leaves=63) — ранний early stopping (1087 итераций vs 2996), что говорит об overfitting с глубокими деревьями.

4. **HistGBM** — худший результат (−0.010). Не поддерживает float targets (label smoothing невозможно), что объясняет большую часть разницы. Label smoothing (Phase 9.1) даёт +0.008 F1 — как раз примерно та разница.

5. **Stage 1 (borked F1=0.825)** одинаков у всех — ожидаемо, т.к. Stage 1 shared.

**Главный вывод:** LightGBM cross_entropy + label smoothing — оптимальная комбинация. Преимущество LightGBM не в алгоритме boosting, а в поддержке `cross_entropy` с float targets (label smoothing). CatBoost и XGBoost поддерживают soft labels иначе, и для нашего типа noise (task-inherent, не annotator-specific) это не помогает.

**Статус:** Phase 11.2 закрыт. Альтернативные модели не улучшают baseline.

---

## Phase 11.3 — Feature selection и regularization (1–2 дня, +0.003–0.008 F1)

### R6. Boruta / SHAP-based feature pruning

**Что:** 119 фичей — вероятно 20–30 шумных. Boruta (shadow features) или recursive SHAP elimination: удалить фичи с mean |SHAP| < threshold.

**Гипотеза:** При noisy labels шумные фичи вредят сильнее — модель фитит случайные корреляции. Phase 9.5 показал: добавление фичей ухудшает (ALL combined = −0.002). Значит и обратное может помочь: удаление слабых.

**Инференс:** ✅ Да — меньше фичей = быстрее.
**Эффект:** Medium (+0.003–0.008 F1). Особенно для Stage 2.
**Стоимость:** Низкая. `pip install boruta`, ~15 строк. Или ручной SHAP pruning.

### R7. Monotonic constraints для domain knowledge

**Что:** Задать monotonic constraints в LightGBM:
- `ram_gb`: monotone_increasing (больше RAM → не хуже)
- `deck_verified_status`: monotone_increasing (verified → лучше)
- `total_reports`: monotone_increasing (больше отчётов → надёжнее оценка)

**Гипотеза:** Domain constraints = implicit regularization. Предотвращает overfit на noisy корреляции (напр. "4GB RAM → works_oob" из-за выборки старых простых игр на слабом железе).

**Инференс:** ✅ Да — только параметр обучения.
**Эффект:** Low (+0.001–0.003 F1). Regularization effect.
**Стоимость:** Тривиально. 1 строка: `monotone_constraints=[...]`.

---

## Phase 11.4 — Валидация и data leakage (1 день, корректность метрик)

### R8. GroupKFold по app_id

**Что:** Проверить data leakage через game aggregates (P1+P2). Текущий time-based split может допускать: агрегаты (frac_any_customization и т.д.) считаются по всем данным, включая test reports.

**Проверка:** Пересчитать агрегаты только на train part (leave-test-out). Сравнить F1. Если разница > 0.01 — есть leakage.

**Гипотеза:** Game aggregates (26 фичей) — сильнейшая группа (+0.024 F1). Если они включают test data → метрики оптимистичны. GroupKFold по app_id гарантирует: все отчёты одной игры либо в train, либо в test.

**Инференс:** ✅ Критично — при инференсе агрегаты доступны только из train.
**Эффект:** Корректность метрик. Может "ухудшить" F1, но даст honest baseline.
**Стоимость:** Низкая. ~20 строк.

---

## Phase 11.5 — Self-training и active learning (3–5 дней, +0.003–0.010 F1)

### R9. Iterative self-training (P22 из Phase 9, не реализован)

**Что:** Train → find high-confidence disagreements (P>0.90 vs label) → relabel → retrain. Пороги: 0.95 → 0.90 → 0.85 за 3 раунда, cap 5% per round.

**Гипотеза:** Model-guided расширение rule-based relabeling (Phase 8). В отличие от Dawid-Skene, работает с single-annotator данными. В отличие от Cleanlab (удаляет), self-training исправляет — сохраняет объём данных.

**Инференс:** ✅ Да — только обучение.
**Эффект:** Medium (+0.003–0.007 F1). Расширяет Cleanlab: не удаляет, а корректирует.
**Стоимость:** Low-medium. ~30 строк. Cap per round предотвращает drift.

### R10. Active learning: manual review top-uncertain samples

**Что:** Найти 500–1000 samples с максимальной неопределённостью модели (P ≈ 0.5 для Stage 2) + разногласием с Cleanlab. Вручную проверить и перелейблить.

**Гипотеза:** 500 вручную размеченных boundary samples > 50000 noisy. Targeted human-in-the-loop для самых спорных случаев. Одноразовая инвестиция с кумулятивным эффектом.

**Инференс:** ✅ Да.
**Эффект:** Medium-high (+0.005–0.010 F1). Прямое уменьшение noise.
**Стоимость:** Высокая (ручной труд). ~2–3 часа разметки + ~10 строк кода для selection.

---

## Phase 11.6 — Экспериментальные подходы (5+ дней)

### R11. Two-tower model для cold-start

**Что:** Обучить маленькую модель: hardware tower (GPU, CPU, RAM, driver) + game tower (engine, categories, DRM, anticheat) → dot product → compatibility score. Prediction используется как 1 фича в LightGBM.

**Гипотеза:** Лучше SVD для новых игр: SVD требует отчётов, two-tower работает с metadata. Для существующих игр — дополняет SVD другим inductive bias (learned non-linear interactions vs linear decomposition).

**Инференс:** ✅ Да — оба tower'а используют inference-time данные.
**Эффект:** Medium (+0.003–0.008 F1). Главная ценность — cold-start.
**Стоимость:** Medium-high. PyTorch, ~100 строк.

### R12. Contrastive learning для game embeddings

**Что:** SimCLR-style: позитивные пары — отчёты одной игры с одинаковым verdict, негативные — разный verdict. Результат: embeddings которые явно кодируют compatibility.

**Гипотеза:** Текущий SVD кодирует co-occurrence (какие GPU/игры встречаются вместе), а не compatibility. Contrastive embeddings будут alignment-aware: "эта игра похожа на ту, потому что обе работают на этом железе" vs "обе популярны".

**Инференс:** ✅ Да — precomputed.
**Эффект:** Medium (+0.003–0.008 F1). Зависит от качества контрастных пар.
**Стоимость:** High. PyTorch + training loop. ~150 строк.

### R13. Multi-task learning

**Что:** Один NN предсказывает одновременно:
- verdict (borked/works) — основная задача
- has_customizations (yes/no) — proxy для tinkering
- fault_count (regression) — proxy для severity
- protondb_tier (ordinal) — community consensus

Shared representation через общие hidden layers.

**Гипотеза:** Auxiliary tasks действуют как regularizer: shared layers учат более robust representation. Предсказание has_customizations — более объективный сигнал чем verdict (бинарный факт vs субъективная оценка).

**Инференс:** ✅ Да.
**Эффект:** Medium (+0.003–0.008 F1). Зависит от quality of auxiliary targets.
**Стоимость:** High. PyTorch multi-head. ~200 строк.

---

## Порядок реализации

```
Phase 11.1 (2–3 дня):  R1 (binary + rules) или R3 (conformal)   → +0.03–0.05 F1
Phase 11.2 (2–3 дня):  R4 (CatBoost Stage 2)                     → +0.003–0.010 F1
Phase 11.3 (1–2 дня):  R6 (Boruta pruning) + R7 (monotonic)      → +0.003–0.008 F1
Phase 11.4 (1 день):   R8 (GroupKFold leakage check)              → корректность
Phase 11.5 (3–5 дней):  R9 (self-training) + R10 (active learn)   → +0.003–0.010 F1
Phase 11.6 (5+ дней):  R11–R13 (deep learning)                    → +0.003–0.008 F1
                                                           Итого:   +0.02–0.05 F1
```

## Метрики

| Метрика | Phase 9 финал | Цель Phase 10 | Цель Phase 11 |
|---|---|---|---|
| F1 macro | 0.7253 | > 0.74 | **> 0.77** |
| works_oob F1 | 0.508 | > 0.55 | **> 0.60** |
| borked F1 | 0.825 | ≥ 0.825 | **≥ 0.83** |
| ECE | 0.008 | < 0.010 | **< 0.008** |

## Ключевой вывод

Модель упёрлась в потолок feature engineering и архитектурных изменений (Phase 9.4–9.5: 14 экспериментов, все ≤ 0). Два реалистичных пути:

1. **Чистить лейблы** (Phase 10 → 11.5): extracted_data relabeling, confidence weighting, self-training, active learning
2. **Менять постановку** (Phase 11.1): убрать субъективную границу tinkering↔oob через бинарную задачу + rule-based split, или conformal prediction для honest uncertainty

R1 (бинарная задача + rules) — самый перспективный эксперимент: устраняет корневую причину 80% ошибок.
