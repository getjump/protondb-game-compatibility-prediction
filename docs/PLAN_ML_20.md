# Phase 20: Task reformulation и финальные эксперименты

## Контекст

F1=0.7801 при 15-20% subjective label noise. Теоретический ceiling ~0.82-0.85. Все feature engineering, model architecture, label denoising, и data quality approaches исчерпаны.

**Cumulative progress:** 0.7245 → 0.7801 (+0.056 F1) за Phases 12-19.

**Оставшиеся направления:** не "как предсказывать лучше", а "что предсказывать" и "как model over-relies на отдельные features".

---

## Phase 20.1 — Binary deployment: borked vs works (1 день)

### Переосмысление задачи для production

**Что:** Вместо 3-class cascade → binary model (borked vs works). Tinkering/oob split — rule-based из structured fields.

**Мотивация:**
- Binary F1=0.906 vs 3-class F1=0.780 — огромная разница
- Для пользователя: "игра работает/не работает" важнее чем "нужен ли GE-Proton"
- Tinkering/oob split объективно из данных: has customization flags → tinkering, else → oob

**Deployment flow:**
```
Input: (game, hardware, proton)
  → Binary model: P(works) = 0.87
  → If works: rule-based split
    - has cust_flags / launch_options → "works with tweaking"
    - else → "works out of the box"
  → SHAP: "top factors: game_emb, irt_difficulty, gpu_family"
```

**Не experiment — это production-ready pipeline.** Binary model уже обучается как Stage 1.

---

## Phase 20.2 — Per-(game, hardware) aggregated prediction (1-2 дня, evaluation)

### Предсказание для пары (игра, конфигурация)

**Что:** Вместо per-report evaluation → per-(app_id, gpu_family) evaluation. Aggregation predictions множества reports для одной game+hardware пары.

**Алгоритм:**
```python
# Для каждой (app_id, gpu_family) в test set:
reports = all reports for this pair
predictions = model.predict(reports)
aggregated = mode(predictions)  # или weighted by confidence
ground_truth = mode(actual verdicts)
```

**Метрики:**
- Per-pair accuracy/F1 вместо per-report
- Coverage: % пар с confident prediction
- Сравнить с ProtonDB tier (community consensus)

**Гипотеза:** Per-report errors часто cancel out при aggregation. Если 8 из 10 reports для game+gpu предсказаны правильно — aggregated prediction правильный.

**Стоимость:** ~40 строк evaluation.

---

## Phase 20.3 — Variant feature investigation (0.5 дня, диагностика)

### Stage 2 без variant

**Факт:** `variant` имеет gain 310K — в 5x больше следующего feature (contributor_consistency 37K). Модель может over-rely.

**Experiment:**
- (a) Drop variant из Stage 2 → проверить насколько другие features компенсируют
- (b) Variant interaction: variant × irt_difficulty, variant × gpu_family

**Гипотеза:** Если drop variant → F1 падает сильно, модель слишком зависима от одного feature. Если падает мало — variant redundant с другими.

---

## Phase 20.4 — Curriculum learning (1 день, +0.003-0.008 F1)

### Staged training: easy → hard

**Что:** Обучать Stage 2 в два этапа:
1. Первые 1000 rounds: только на explicit oob labels (clean, ~17% train)
2. Продолжить с `init_model` на всех данных ещё 1000 rounds

**Гипотеза:** Model learns clean decision boundary first, then adapts to noisy data without forgetting.

---

## Phase 20.5 — Error cascade correction (1 день, +0.002-0.005 F1)

### Метамодель на ошибках

**Что:** Train secondary model:
- Input: primary cascade predictions + confidence + IRT features + report metadata
- Target: "primary model ошибётся? (binary)"
- Use: flag uncertain predictions, adjust confidence

**Стоимость:** ~30 строк. Logistic regression on OOF predictions.

---

## Phase 20.6 — Ensemble of time windows (1 день, +0.002-0.005 F1)

### Модели на разных временных окнах

**Что:** Train 3 models на разных time windows:
- Model A: last 2 years
- Model B: last 4 years
- Model C: all data

Average predictions. Каждая модель ловит разные patterns.

---

## Порядок

```
Phase 20.3 (0.5 дня):  Variant ablation                    → диагностика
Phase 20.2 (1-2 дня):  Per-game evaluation                 → evaluation metric
Phase 20.1 (1 день):   Binary deployment                   → production pipeline
Phase 20.4 (1 день):   Curriculum learning                 → +0.003-0.008 F1
Phase 20.5 (1 день):   Error cascade                       → +0.002-0.005 F1
Phase 20.6 (1 день):   Time window ensemble                → +0.002-0.005 F1
```
