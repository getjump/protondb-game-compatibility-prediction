# Phase 18: SOTA approaches — soft labels, threshold optimization, robust loss

## Контекст

Cumulative: **0.7245 → 0.7776 (+0.053 F1)**. Bottleneck: works_oob recall 0.557, Stage 2 boundary = 74% ошибок.

**Комплексный анализ проблемы:**

Наша задача — не классический noisy labels, а **subjective annotation disagreement**. Граница tinkering↔works_oob — inherently subjective: разные пользователи имеют разный порог. IRT моделирует эту subjectivity (+0.030 F1), но remaining errors — genuine ambiguity, не noise.

**Что исчерпано:**
- Feature engineering: PICS, temporal, annotator SVD — все redundant или negative
- Alternative models: CatBoost, XGBoost, HistGBM — все хуже LightGBM
- Self-training iterations: marginal (+0.001)
- FM / Proton×Game SVD: sparse, underfit

**SOTA ресерч выявил 3 высокоприоритетных направления:**

1. **Post-hoc threshold optimization** — zero-cost, no retraining
2. **IRT-derived adaptive soft labels** — замена fixed α=0.15 на per-sample smoothing
3. **Robust focal loss** — focus на hard boundary cases

---

## Phase 18.1 — Post-hoc threshold optimization (0.5 дня, +0.010-0.030 F1)

### Оптимизация decision boundary без перетренировки

**Paper:** "Multiclass threshold-based classification" (2025, arXiv:2511.21794)

**Идея:** Вместо `argmax(P(c))` → `argmax(P(c) / t_c)` где t_c оптимизируется на validation set для максимизации F1 macro.

**Почему это должно сработать:**
- works_oob recall = 0.557 при F1 = 0.603
- Модель часто предсказывает tinkering когда P(oob) = 0.4-0.5 → bordering threshold
- Снижение порога для oob class с 0.5 до ~0.35-0.40 увеличит recall за счёт precision
- Оптимальный trade-off findable через grid search на calibration set

**Алгоритм:**
```python
# На calibration set:
y_proba = cascade.predict_proba(X_cal)

best_f1 = 0
for t_borked in np.arange(0.3, 0.7, 0.05):
    for t_oob in np.arange(0.2, 0.6, 0.05):
        t = np.array([t_borked, 1.0, t_oob])  # tinkering = reference
        y_pred = np.argmax(y_proba / t, axis=1)
        f1 = f1_macro(y_cal, y_pred)
        if f1 > best_f1: ...
```

**Инференс:** Да — thresholds фиксируются, добавляются в CascadeClassifier.
**Эффект:** Medium-high (+0.010-0.030 F1). Literature reports +0.02-0.04 на imbalanced 3-class.
**Стоимость:** ~20 строк. Zero retraining cost.

---

## Phase 18.2 — IRT-derived adaptive soft labels (1-2 дня, +0.005-0.015 F1)

### Per-sample label smoothing вместо global α=0.15

**Paper:** "Learning with Confidence: Training Better Classifiers from Soft Labels" (2025, Springer MLKD)

**Идея:** Вместо фиксированного label smoothing α=0.15 для всех → adaptive smoothing по IRT:

```python
# Текущий подход:
y_smooth = y * 0.85 + (1-y) * 0.15  # одинаково для всех

# Новый подход:
p_irt = sigmoid(theta_j - d_i)  # IRT prediction P(tinkering)
alpha_i = confidence_from_irt(theta_j, d_i, n_annotators_for_item)

# Для tinkering report (y=0 в Stage 2):
#   strict annotator (theta=3) + easy item (d=-2) → P_irt=0.99 → alpha=0.40 (aggressive smoothing)
#   lenient annotator (theta=-1) + hard item (d=2) → P_irt=0.05 → alpha=0.02 (trust label)
y_smooth_i = y_i * (1 - alpha_i) + (1 - y_i) * alpha_i
```

**Формула alpha:**
```python
# Расстояние между IRT prediction и raw label → мера disagreement
alpha_i = clip(|P_irt - y_i| * 0.5, 0.05, 0.40)
```

**Почему лучше fixed smoothing:**
- Fixed α=0.15 одинаково сглаживает reliable и unreliable labels
- Adaptive: reliable annotator на easy item → minimal smoothing (trust label)
- Unreliable annotator на ambiguous item → aggressive smoothing (don't trust)
- IRT уже знает кто reliable и что ambiguous

**Инференс:** Нет (train-time only).
**Эффект:** Medium (+0.005-0.015 F1). Более principled чем fixed smoothing.
**Стоимость:** ~30 строк. Модификация `train_stage2`.

---

## Phase 18.3 — Robust Focal Loss для Stage 2 (1 день, +0.005-0.010 F1)

### Focus на hard boundary cases

**Paper:** "Robust-GBDT" (Luo et al., 2023, KAIS 2025, arXiv:2310.05067)

**Focal loss:**
```
L = -(1 - p_t)^γ * log(p_t)
```
- γ=0: standard CE
- γ=2: downweight easy examples, focus on hard boundary

**Почему для нас:** 52% ошибок = oob→tinkering. Это hard boundary cases. Standard CE тратит gradient budget на easy examples (borked с confidence 0.95). Focal loss перераспределяет budget на Stage 2 boundary.

**Реализация через LightGBM custom objective:**
```python
def focal_binary_objective(y_true, y_pred, gamma=2.0):
    p = 1 / (1 + np.exp(-y_pred))
    grad = p - y_true  # standard CE gradient
    focal_weight = (1 - p) ** gamma * y_true + p ** gamma * (1 - y_true)
    grad = grad * focal_weight
    hess = p * (1 - p) * focal_weight  # approximate
    return grad, hess
```

**Комбинация с adaptive soft labels (18.2):** focal loss + soft labels = focus на hard examples с adaptive label confidence.

**Инференс:** Нет (train-time only, model predicts same way).
**Эффект:** Low-medium (+0.005-0.010 F1).
**Стоимость:** ~30 строк custom objective.

---

## Phase 18.4 — Cleanlab + IRT combined relabeling (1 день, +0.003-0.010 F1)

### Два ортогональных сигнала для label correction

**Что:** Cleanlab использует model predictions (OOF) для обнаружения mislabels. IRT использует annotator behavior. Они ловят разные типы ошибок:

- **IRT ловит:** strict annotator says tinkering, IRT says oob → annotator bias
- **Cleanlab ловит:** all features predict works_oob, but label says tinkering → feature-label mismatch

**Алгоритм:**
1. Run Cleanlab OOF → get label quality score per sample
2. Run IRT → get P_irt per sample
3. Combine: `reliability = cleanlab_score * irt_confidence`
4. Low reliability → aggressive soft label; high reliability → trust label

**Или:** Use Cleanlab quality scores as features в LightGBM (meta-learning). "Model thinks this label is probably wrong" → useful signal.

**Эффект:** Low-medium (+0.003-0.010 F1). Complementary to IRT.
**Стоимость:** ~30 строк.

---

## Phase 18.5 — Disagreement features (0.5 дня, +0.003-0.008 F1)

### Explicit annotation disagreement as features

**Paper:** Research shows disagreement behind an aggregated label indicates semantics (ambiguity/difficulty), not just noise.

**Новые фичи (per game×gpu_family):**
- `game_label_entropy` — entropy of verdict distribution (high = ambiguous game)
- `game_disagreement_rate` — % pairs of annotators who disagree on this game
- `game_irt_discrimination` — из 2PL IRT (уже пробовали, +0.003)

**Отличие от текущих:**
- `game_verdict_agreement` (Phase 16.6) — уже есть, но per-game only
- Новое: per-(game, gpu_family) disagreement — более granular, IRT items level

**Эффект:** Low (+0.003-0.008 F1).
**Стоимость:** ~20 строк.

---

## Phase 18.6 — Deferred Re-Weighting (DRW) (0.5 дня, +0.002-0.005 F1)

### Train first without class weights, then apply

**Paper:** LDAM (Label-Distribution-Aware Margin Loss, 2019)

**Идея:** Class weights в начале обучения мешают модели учить хорошие splits. DRW:
1. Первые N% iterations: train без class weights (learn features)
2. Оставшиеся iterations: apply class weights (bias to minority)

**Реализация:** LightGBM `init_model` — train 1000 rounds without weights, then continue 1000 rounds with oob weight 1.5.

**Эффект:** Low (+0.002-0.005 F1).
**Стоимость:** ~15 строк.

---

## Порядок реализации

```
Phase 18.1 (0.5 дня):  Threshold optimization              → +0.010-0.030 F1
Phase 18.2 (1-2 дня):  Adaptive soft labels                 → +0.005-0.015 F1
Phase 18.3 (1 день):   Focal loss Stage 2                   → +0.005-0.010 F1
Phase 18.4 (1 день):   Cleanlab + IRT combined              → +0.003-0.010 F1
Phase 18.5 (0.5 дня):  Disagreement features                → +0.003-0.008 F1
Phase 18.6 (0.5 дня):  Deferred Re-Weighting                → +0.002-0.005 F1
                                                       Итого: +0.015-0.040 F1
```

**Приоритет:** 18.1 → 18.2 → 18.3 → 18.4 → 18.5 → 18.6

18.1 — zero-cost win (no retraining). Самый высокий ROI.
18.2 — principled замена fixed label smoothing.
18.3 — complementary к 18.2 (focal + soft labels).

---

## Зависимости

- Phase 12.8 (IRT θ, d) — для adaptive soft labels и disagreement features
- Calibration set — для threshold optimization
- Cleanlab (pip install cleanlab) — для 18.4

---

## Метрики

| Метрика | Phase 16 | Цель Phase 18 |
|---|---|---|
| F1 macro | 0.7776 | **> 0.80** |
| works_oob F1 | 0.603 | **> 0.65** |
| works_oob recall | 0.557 | **> 0.65** |
| borked F1 | 0.846 | **≥ 0.84** |

## Ключевой вывод

Phase 12-16 исчерпали feature engineering и label denoising. Phase 18 атакует с другой стороны:

1. **Decision boundary** (18.1): оптимизировать where to draw the line, не how to learn features
2. **Loss function** (18.2, 18.3): как модель учится, не на чём
3. **Label quality** (18.4): ортогональный сигнал к IRT

Threshold optimization (18.1) — самый перспективный: zero-cost, literature reports +0.02-0.04 F1 на imbalanced multi-class. Наш works_oob recall 0.557 при confidence threshold 0.5 → значительный room для improvement.

## Ссылки

- [Multiclass threshold-based classification (2025)](https://arxiv.org/abs/2511.21794)
- [Learning with Confidence: Soft Labels (2025)](https://link.springer.com/article/10.1007/s10994-025-06860-8)
- [Robust-GBDT: Focal Loss for GBDT (2023/2025)](https://arxiv.org/abs/2310.05067)
- [Training GBDT with Label Noise (2024)](https://arxiv.org/abs/2409.08647)
- [LDAM: Deferred Re-Weighting (2019)](https://arxiv.org/abs/1906.07413)
- [Multi-annotator Deep Learning (MaDL, 2023)](https://arxiv.org/abs/2304.02539)
- [Cleanlab: Confident Learning (2021)](https://github.com/cleanlab/cleanlab)
- [Focal Loss for LightGBM](https://github.com/jrzaurin/LightGBM-with-Focal-Loss)
