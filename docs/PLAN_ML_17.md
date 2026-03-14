# Phase 17: Финальная оптимизация и production readiness

## Контекст

Cumulative progress: **0.7245 → 0.7776 (+0.053 F1)**.

Что сработало и почему:
- IRT (+0.030): decomposition annotator bias от game difficulty — прямая атака на root cause
- Contributor-aware relabel (+0.017): адресная коррекция labels от strict annotators
- Class weight (+0.006): компенсация temporal class shift (oob растёт со временем)

Что НЕ сработало:
- Новые features (PICS, temporal, annotator SVD) — IRT + existing features уже ловят основной signal
- Sample weighting (contributor tally, time decay) — LightGBM с label smoothing уже robust
- Alternative models (CatBoost, XGBoost) — LightGBM cross_entropy оптимален
- Temporal features — `report_age_days` уже достаточный proxy

**Паттерн:** feature engineering исчерпан. Прирост идёт от denoising (IRT, relabeling) и balancing. Оставшийся gap — genuine ambiguity на Stage 2 boundary.

---

## Phase 17.1 — Hyperparameter tuning Stage 2 (1 день, +0.003-0.008 F1)

### Systematic Optuna sweep

**Что:** Stage 2 hyperparameters были подобраны manually в Phase 9. С IRT features и новым relabeling оптимальные значения могли сместиться.

**Параметры для sweep:**
```python
{
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.01, 0.1, 1.0],
    "reg_lambda": [0.01, 0.1, 1.0],
    "label_smoothing": [0.10, 0.15, 0.20],
    "oob_class_weight": [1.0, 1.3, 1.5, 1.8, 2.0],
}
```

**Метод:** Optuna TPE sampler, 100 trials, 3-fold CV на train set. Optimize F1 macro.

**Гипотеза:** IRT features изменили ландшафт — возможно нужны другие regularization params или label smoothing alpha.

**Эффект:** Low-medium (+0.003-0.008 F1). Diminishing returns, но systematic search может найти карманы.
**Стоимость:** ~30 строк + compute time (~2 часа на 100 trials).

---

## Phase 17.2 — Stage 2 ensemble (1-2 дня, +0.003-0.010 F1)

### Bagging / stacking для Stage 2

**Что:** Вместо одного LightGBM Stage 2 — ensemble из 3-5 моделей с разными:
- Random seeds (bagging)
- Feature subsets (random subspace)
- Label smoothing values (α=0.10, 0.15, 0.20)

**Averaging:** simple mean of P(works_oob) across models.

**Гипотеза:** Stage 2 boundary noisy → single model overfit к конкретным noisy patterns. Ensemble усредняет noise. Bagging даёт +0.002-0.005 F1 на noisy задачах (classic result).

**Альтернатива — stacking:**
- Level 0: LightGBM с разными feature subsets (text-only, hardware-only, IRT-only, all)
- Level 1: Logistic regression на OOF predictions Level 0
- Ловит complementary signals из разных feature groups

**Инференс:** Да — все модели одного типа, inference ×3-5 по времени (приемлемо).
**Эффект:** Low-medium (+0.003-0.010 F1).
**Стоимость:** ~50 строк.

---

## Phase 17.3 — Conformal prediction (1 день, UX + честные метрики)

### Prediction sets с гарантированным покрытием

**Что:** Вместо forced hard prediction — prediction set с coverage guarantee.

```python
from mapie.classification import MapieClassifier
mapie = MapieClassifier(cascade, method="lac", cv="prefit")
mapie.fit(X_cal, y_cal)
y_pred_sets, y_pred_proba = mapie.predict(X_test, alpha=0.1)
# y_pred_sets[i] = {tinkering, works_oob} для borderline cases
```

**Ожидание:**
- ~30% predictions: {tinkering, works_oob} (honest: "мы не уверены")
- ~60% predictions: singleton (confident)
- ~10% predictions: {tinkering} или {works_oob} с high confidence

**API output:**
```json
{
  "prediction": "works",
  "confidence": 0.62,
  "prediction_set": ["tinkering", "works_oob"],
  "note": "Borderline — может потребоваться настройка"
}
```

**Метрики:** Set size distribution, coverage, conditional coverage per class.

**Эффект:** Не улучшает F1, но даёт честную uncertainty. Для UX важнее чем hard F1.
**Стоимость:** ~20 строк. `pip install mapie`.

---

## Phase 17.4 — GroupKFold validation (0.5 дня, метрики)

### Проверка честности метрик

**Что:** Текущий time-based split: 64% test games overlap с train. Game aggregates и IRT difficulty вычислены на тех же играх → возможный optimistic bias.

**Эксперимент:**
- GroupKFold по app_id: все reports одной игры в одном fold
- Сравнить F1 с time-based split
- Если gap > 0.02 — метрики optimistic

**Также:** Test на completely cold-start scenario: train на 80% games, test на 20% unseen games.

**Эффект:** Честность метрик. Может "ухудшить" F1, но даст honest baseline.
**Стоимость:** ~20 строк.

---

## Phase 17.5 — Feature pruning Stage 2 (1 день, +0.002-0.005 F1)

### SHAP-based pruning для Stage 2

**Что:** Stage 2 имеет 121 feature, но top-15 дают 80%+ gain. Noisy features вредят при label noise.

**Алгоритм:**
1. Compute SHAP values на validation set
2. Rank features by mean |SHAP|
3. Iteratively remove bottom feature, retrain, check F1
4. Stop when F1 starts dropping

**Ожидание:** 60-80 features достаточно. Removing 40+ noisy features → small F1 improvement + faster inference.

**Эффект:** Low (+0.002-0.005 F1). Regularization effect.
**Стоимость:** ~30 строк.

---

## Phase 17.6 — Production pipeline integration (2 дня)

### Интеграция всех findings в production-ready pipeline

**Что:**
1. IRT fitting → cached artifacts (theta.json, difficulty.json)
2. Feature extraction pipeline включает IRT + error features
3. Stage 2 class weight зафиксирован
4. Conformal prediction для API
5. Export: model_cascade.pkl + irt_params.json + embeddings.npz

**Новый inference flow:**
```
Input: (app_id, gpu, proton_version, ...)
  → Feature extraction (hardware, text, game metadata)
  → IRT features (precomputed difficulty per game×gpu)
  → Error features (precomputed agreement per game)
  → Cascade predict (Stage 1 → Stage 2)
  → Conformal prediction set
  → SHAP explanation (top-3 factors)
  → API response
```

**IRT при инференсе:**
- `irt_game_difficulty` — precomputed per (game, gpu_family), stored in DB
- `irt_contributor_strictness` — не доступен (unknown future contributor), fill 0
- `contributor_consistency` — не доступен, fill median
- `game_verdict_agreement` — precomputed per game

**Стоимость:** ~100 строк refactoring.

---

## Порядок реализации

```
Phase 17.4 (0.5 дня):  GroupKFold validation               → честные метрики
Phase 17.1 (1 день):   Optuna hyperparam tuning             → +0.003-0.008 F1
Phase 17.2 (1-2 дня):  Stage 2 ensemble                     → +0.003-0.010 F1
Phase 17.5 (1 день):   Feature pruning                      → +0.002-0.005 F1
Phase 17.3 (1 день):   Conformal prediction                 → UX improvement
Phase 17.6 (2 дня):    Production integration                → ship it
                                                       Итого: +0.005-0.015 F1
```

**Приоритет:** 17.4 → 17.1 → 17.3 → 17.6 → 17.2 → 17.5

17.4 — сначала убедиться что метрики честные.
17.1 — quick tuning win.
17.3 — conformal prediction для honest UX.
17.6 — production readiness.

---

## Метрики

| Метрика | Phase 16 | Цель Phase 17 (final) |
|---|---|---|
| F1 macro | 0.7776 | **> 0.79** |
| works_oob F1 | 0.603 | **> 0.62** |
| borked F1 | 0.846 | **≥ 0.85** |
| Calibration (ECE) | ~0.012 | **< 0.010** |
| Conformal set size | — | **< 1.5 avg** |

## Результаты (2026-03-14)

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob | oob_recall |
|---|---|---|---|---|---|---|
| baseline | 0.7771 | — | 0.846 | 0.885 | 0.601 | 0.550 |
| **17.1 best hparams** | **0.7811** | **+0.004** | 0.846 | 0.881 | **0.617** | **0.608** |
| **17.2 ensemble 5 seeds** | **0.7813** | **+0.004** | 0.846 | 0.884 | 0.614 | 0.557 |
| 17.4 GroupKFold | 0.7683 | −0.009 | — | — | — | — |
| 17.5 top-30 features | 0.7774 | +0.000 | 0.846 | 0.883 | 0.603 | 0.564 |

**Best hparams:** `reg_alpha=1.0, reg_lambda=1.0, oob_weight=1.8` (stronger regularization).

**GroupKFold gap: 0.009** — метрики slightly optimistic but acceptable.

**Cumulative: 0.7245 → 0.7811 (+0.057 F1)**

## Ключевой вывод

Phase 17 — финализация. Основные прорывы сделаны (IRT + relabel + class weight = +0.053 F1). Оставшийся потенциал: +0.005-0.015 через tuning, ensemble, pruning. Главный фокус — production readiness и honest uncertainty (conformal prediction).

## Итоги всего проекта (Phases 9-17)

| Phase | Approach | ΔF1 | Key insight |
|---|---|---|---|
| 9 (14 exp) | Feature engineering | 0.000 | Feature ceiling reached |
| 11.2 | Alternative models | −0.002 | LightGBM optimal |
| **12.8** | **IRT features** | **+0.030** | **Decompose annotator bias** |
| **13.2** | **Contributor-aware relabel** | **+0.017** | **Replace Cleanlab + Phase 8** |
| 14 | PICS features | 0.000 | Redundant with existing |
| 15 | Temporal features | −0.003..−0.017 | report_age_days sufficient |
| **16** | **Class weight + error features** | **+0.006** | **Compensate class shift** |

Total: **+0.053 F1** (0.7245 → 0.7776). Всё из denoising и balancing, не из features.
