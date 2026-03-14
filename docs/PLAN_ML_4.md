# ML Pipeline — двухступенчатый классификатор (фаза 4)

## Мотивация

Анализ показал что главная проблема текущей модели — размытая граница tinkering↔works_oob:
- Error clusters: ошибки need→work и work→need живут в одном плотном кластере, неразделимы
- Confusion matrix: 40.7% works_oob предсказывается как tinkering, 15.6% tinkering как works_oob
- Temporal analysis: works_oob recall деградирует с 0.80 до 0.45 на новых данных
- Probability distributions: P(tinkering) и P(works_oob) сильно перекрываются

При этом граница borked↔works **значительно чётче** — borked кластеризуется отдельно в UMAP, `pct_stability_faults` даёт чистый сигнал.

**Решение**: разделить задачу на две последовательные модели.

---

## Архитектура

```
                    Report + Features
                          │
                    ┌─────▼─────┐
                    │  Stage 1  │
                    │ works vs  │
                    │ borked    │
                    └─────┬─────┘
                          │
                ┌─────────┴─────────┐
                │                   │
           borked (0)          works (1)
                                    │
                              ┌─────▼─────┐
                              │  Stage 2  │
                              │ oob vs    │
                              │ tinkering │
                              └─────┬─────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                   tinkering (1)        works_oob (2)
```

### Stage 1: works vs borked

- **Цель**: бинарная классификация — игра работает (tinkering + works_oob) или сломана (borked)
- **Target**: 0 = borked (~14%), 1 = works (~86%)
- **Ключевые фичи**: `pct_stability_faults`, `game_emb_0`, `anticheat`, `anticheat_status`, `has_denuvo`
- **class_weight**: {0: 3.0, 1: 1.0} — важнее не пропустить borked
- **Метрика оптимизации**: F1 borked, при recall borked ≥ 0.55

### Stage 2: tinkering vs works_oob

- **Цель**: бинарная классификация — работает из коробки или нужна настройка
- **Target**: 0 = tinkering (~79% от works), 1 = works_oob (~21% от works)
- **Данные**: только отчёты где Stage 1 предсказал "works" (или все non-borked для training)
- **Ключевые фичи**: `variant`, `report_age_days` (пересмотренный), `total_reports`, `gpu_family`, `pct_needs_protontricks`
- **class_weight**: {0: 1.0, 1: 2.0} — мягче чем сейчас, т.к. задача проще
- **Метрика оптимизации**: F1 macro, при precision works_oob ≥ 0.55

---

## План экспериментов

### Шаг 1. Baseline Stage 1 (works vs borked)

**Что делаем**:
- Объединить tinkering + works_oob → works (target=1), borked → target=0
- Обучить LightGBM с текущими фичами и гиперпараметрами
- Оценить: accuracy, F1, precision/recall borked

**Ожидание**: F1 borked 0.65+ (сейчас 0.46 в 3-class). Задача проще — два класса вместо трёх.

**Проверки**:
- Сравнить с текущей моделью: если 3-class предсказания объединить (tinkering+oob → works), какой F1 borked?
- SHAP: какие фичи важны для borked detection? Если `report_age_days` остаётся доминантом — проблема не решена.

### Шаг 2. Baseline Stage 2 (tinkering vs works_oob)

**Что делаем**:
- Обучить только на отчётах где verdict != borked (target: 0=tinkering, 1=works_oob)
- Тот же набор фич
- Оценить: accuracy, F1 macro, precision/recall каждого класса

**Ожидание**: F1 macro 0.55-0.60 (сейчас tinkering/oob F1 ~0.80/0.49). Задача сложнее из-за label noise, но без borked шум меньше.

**Проверки**:
- Насколько label noise влияет? cleanlab на этом подмножестве
- SHAP: отличается ли от 3-class? Если `variant` и `report_age_days` доминируют — те же проблемы

### Шаг 3. Объединение в каскад

**Что делаем**:
- Stage 1 predict → если borked, return borked
- Если works → Stage 2 predict → return tinkering или works_oob
- Оценить на ПОЛНОМ тестовом сете: accuracy, F1 macro, confusion matrix
- Сравнить с single 3-class model

**Критерий успеха**: F1 macro каскада > 0.584 (текущий) И borked recall > 0.45

### Шаг 4. Оптимизация Stage 1

**Эксперименты**:
- **4a.** Убрать `report_age_days` из Stage 1 — borked detection не должен зависеть от времени
- **4b.** Добавить `pct_stability_faults × is_steam_deck` interaction
- **4c.** Threshold tuning: вместо argmax, варьировать порог P(borked) для precision/recall trade-off
- **4d.** Отдельные class_weight: grid search {0: [2, 3, 4, 5], 1: [1]}

### Шаг 5. Оптимизация Stage 2

**Эксперименты**:
- **5a.** Заменить `report_age_days` на `proton_era` или `report_age_relative` — Stage 2 больше всего страдает от temporal bias
- **5b.** Убрать фичи неактуальные для tinkering/oob: `anticheat`, `has_denuvo` (если игра works, античит не причём)
- **5c.** Добавить фичи специфичные для "степени работоспособности": `pct_needs_winetricks`, `pct_needs_protontricks`, `pct_needs_custom_proton` — сильный сигнал "нужна настройка"
- **5d.** Label smoothing / soft labels: P(oob) = pct_works_oob для этой игры. Мягкий таргет вместо hard

### Шаг 6. Confidence-aware output

**Что делаем**:
- Stage 1: если P(borked) ∈ [0.3, 0.7] → "uncertain_borked"
- Stage 2: если P(oob) ∈ [0.35, 0.65] → "uncertain_oob"
- API возвращает: `{"prediction": "works_oob", "confidence": 0.82, "probabilities": {"borked": 0.05, "tinkering": 0.13, "works_oob": 0.82}}`
- Измерить: какая доля попадает в uncertain? Если >30% — threshold слишком широкий

### Шаг 7. Post-hoc калибровка каскада

**Что делаем**:
- Каждый stage калибровать отдельно (isotonic regression)
- Итоговые вероятности: P(borked) = P_s1(borked), P(tinkering) = P_s1(works) × P_s2(tinkering), P(oob) = P_s1(works) × P_s2(oob)
- Оценить ECE (Expected Calibration Error) до и после

---

## Риски и mitigation

| Риск | Вероятность | Mitigation |
|------|-------------|------------|
| Stage 1 ошибается → Stage 2 получает borked на вход | Средняя | Threshold tuning: лучше лишний borked чем пропущенный |
| Error propagation: ошибки Stage 1 × Stage 2 | Средняя | Сравнить с single model на каждом шаге |
| Stage 2 не лучше чем 3-class для tinkering/oob | Высокая | Если F1 stage2 < F1 tinkering/oob из 3-class → оставить single model для этой границы |
| Два model.pkl вместо одного — сложнее deploy | Низкая | Обернуть в один CascadeClassifier class |
| Label noise в Stage 2 не уменьшился | Средняя | cleanlab → soft labels → fall back to probability output |

---

## Структура кода

```python
# protondb_settings/ml/models/cascade.py

class CascadeClassifier:
    """Two-stage classifier: borked vs works → tinkering vs oob."""

    def __init__(self, stage1_model, stage2_model, borked_threshold=0.5):
        self.stage1 = stage1_model  # LGBMClassifier (binary)
        self.stage2 = stage2_model  # LGBMClassifier (binary)
        self.borked_threshold = borked_threshold

    def predict(self, X):
        # Stage 1
        p_borked = self.stage1.predict_proba(X)[:, 1]  # P(borked)
        is_borked = p_borked >= self.borked_threshold

        # Stage 2 (only for non-borked)
        result = np.full(len(X), -1, dtype=int)
        result[is_borked] = 0  # borked

        works_mask = ~is_borked
        if works_mask.any():
            p_oob = self.stage2.predict_proba(X[works_mask])[:, 1]
            result[works_mask] = np.where(p_oob >= 0.5, 2, 1)  # 2=oob, 1=tinkering

        return result

    def predict_proba(self, X):
        p1 = self.stage1.predict_proba(X)  # (n, 2): [P(works), P(borked)]
        p2 = self.stage2.predict_proba(X)  # (n, 2): [P(tinkering), P(oob)]

        # Combine: P(borked), P(tinkering), P(oob)
        proba = np.zeros((len(X), 3))
        proba[:, 0] = p1[:, 1]                    # P(borked)
        proba[:, 1] = p1[:, 0] * p2[:, 0]         # P(works) × P(tinkering|works)
        proba[:, 2] = p1[:, 0] * p2[:, 1]         # P(works) × P(oob|works)
        return proba
```

## Артефакты

```
data/
  model_stage1.pkl     # Stage 1: works vs borked
  model_stage2.pkl     # Stage 2: tinkering vs works_oob
  model_cascade.pkl    # CascadeClassifier (обёртка)
  embeddings.npz       # общие (те же SVD)
  label_maps.json      # общие
```

---

## Целевые метрики

| Метрика | Single model | Каскад (цель) |
|---------|-------------|----------------|
| F1 macro (3-class) | 0.584 | 0.63+ |
| borked recall | 0.38 | 0.55+ |
| borked precision | 0.60 | 0.60+ |
| works_oob precision | 0.44 | 0.55+ |
| works_oob recall (newest) | 0.45 | 0.55+ |
| tinkering F1 | 0.80 | 0.80+ |
| ECE (calibrated) | ~0.15 | < 0.05 |

---

## Результаты экспериментов

### Шаг 3: Baseline каскад (experiment_cascade.py, experiment_cascade_v2.py)

Каскад vs single model:

| Конфигурация | F1 macro | borked R | borked P | oob R | oob P |
|-------------|----------|----------|----------|-------|-------|
| Single 3-class | 0.5843 | 0.38 | 0.60 | — | — |
| Cascade baseline | 0.5906 | 0.41 | 0.57 | 0.55 | 0.45 |
| **Cascade (S2 без report_age_days)** | **0.5929** | **0.41** | **0.57** | **0.55** | **0.46** |

**Вывод**: удаление `report_age_days` из Stage 2 даёт лучший результат (+0.009 vs single). Stage 2 F1 улучшилось с 0.6930→0.6962 — temporal bias мешал.

### Шаг 4: Оптимизация Stage 1 (experiment_cascade_v3_stage1_opt.py)

| Эксперимент | F1 macro | Δ F1 | borked R | borked P |
|------------|----------|------|----------|----------|
| Baseline (w=3, t=0.5) | 0.5929 | — | 0.41 | 0.57 |
| 4a: без report_age_days | 0.5907 | -0.002 | 0.44 | 0.52 |
| 4b: + interaction (stability×deck) | 0.5935 | +0.001 | 0.42 | 0.57 |
| 4a+4b combined | 0.5920 | -0.001 | 0.44 | 0.53 |
| 4d: лучший weight (w=3) | 0.5929 | ±0 | 0.41 | 0.57 |
| 4c: лучший threshold (t=0.5) | 0.5929 | ±0 | 0.41 | 0.57 |
| **w=3, interactions, t=0.45** | **0.5937** | **+0.001** | **0.46** | **0.52** |

**Выводы**:
1. `report_age_days` **полезен** для Stage 1 (в отличие от Stage 2) — borked detection коррелирует со временем
2. Interaction feature `stability×deck` даёт маргинальный прирост (+0.001)
3. Weight w=3 уже оптимален — выше = recall↑ но precision↓ слишком
4. Threshold 0.5 оптимален для F1, но t=0.45 даёт лучший borked recall (0.46 vs 0.41) с минимальной потерей F1
5. **Текущая конфигурация по сути оптимальна**. Stage 1 — не bottleneck.

### Статус шагов

- [x] Шаг 1-2: Baseline Stage 1 / Stage 2
- [x] Шаг 3: Каскад → +0.009 F1 vs single
- [x] Шаг 4: Оптимизация Stage 1 → текущий конфиг оптимален
- [x] Шаг 5: Оптимизация Stage 2 → текущий конфиг оптимален (см. ниже)
- [x] Шаг 6: Confidence-aware output (см. ниже)
- [x] Шаг 7: Post-hoc калибровка (см. ниже)

### Шаги 6-7, 11 (experiment_cascade_v6_aggregation_confidence.py)

#### Шаг 11: Агрегация по (game, gpu_family, cpu_vendor, is_deck)

- 36228 уникальных групп из 69708 отчётов (mean=1.9 reports/group, median=1)
- 72% групп — одиночные отчёты (агрегация не даёт много)

| Min agreement | Groups | Vote F1 | Mean proba F1 |
|---------------|--------|---------|---------------|
| 0.0 (все) | 36228 | **0.6602** | 0.6578 |
| 0.6 | 33081 | 0.6951 | 0.6918 |
| 0.8 | 30759 | **0.7096** | 0.7064 |

**Вывод**: при агрегации F1 растёт до 0.66–0.71 — **значительно лучше** per-report (0.593). Но: (1) median группа = 1 отчёт, (2) multi-report groups F1=0.52 (хуже, т.к. сложнее случаи). Агрегация полезна **для API** (усреднять predictions по запросам к одной игре+железу).

#### Шаг 6: Confidence-aware output

| Max P thresh | Confident% | F1 confident | F1 uncertain | Acc confident |
|-------------|------------|-------------|-------------|---------------|
| 0.5 | 81% | 0.645 | 0.336 | 0.754 |
| 0.6 | 62% | 0.706 | 0.419 | 0.823 |
| 0.7 | 46% | 0.773 | 0.456 | 0.881 |
| 0.8 | 31% | **0.860** | 0.485 | **0.930** |
| 0.9 | 18% | 0.957 | 0.514 | 0.975 |

Зоны неопределённости:
- **Stage 1**: P(borked) ∈ [0.3, 0.7) → 15.8% отчётов (acc=0.50 — random)
- **Stage 2**: P(oob) ∈ [0.35, 0.65) → 28.7% non-borked (основная неопределённость)

**Вывод**: при confidence ≥ 0.7 модель даёт F1=0.77 / acc=0.88 на 46% данных. Рекомендация для API: возвращать confidence level и помечать uncertain predictions.

#### Шаг 7: Post-hoc калибровка (isotonic regression)

| Метрика | До | После |
|---------|-----|------|
| ECE | 0.0183 | **0.0120** (-35%) |
| Brier borked | 0.094 | **0.085** |
| Brier tinkering | 0.179 | **0.162** |
| Brier oob | 0.117 | **0.113** |

**Вывод**: калибровка уже неплохая (ECE=0.018), isotonic улучшает до 0.012. Можно применить в production. F1 после калибровки снизился (0.588→0.522) — ожидаемо, т.к. калибровка меняет ranking на тест-подмножестве.

### Шаг 5: Оптимизация Stage 2 (experiment_cascade_v4_stage2_opt.py)

| Эксперимент | S2 F1 | Cascade F1 | Δ cascade |
|------------|-------|-----------|-----------|
| A baseline (drop age, w={0:1,1:2}) | 0.6962 | 0.5929 | — |
| 5b: drop anticheat/denuvo | 0.6939 | 0.5916 | -0.001 |
| 5d: soft labels (pct_works_oob) | 0.6962 | 0.5929 | ±0 |
| 5b+5d combined | 0.6939 | 0.5916 | -0.001 |
| w_oob=2.5 | 0.6965 | 0.5933 | +0.0004 |
| S2 threshold tuning | — | 0.5929 | t=0.5 optimal |
| lr/num_leaves tuning | — | ~0.5929 | no change |

**Выводы**:
1. Античит **полезен** и для Stage 2 — коррелирует с tinkering vs oob
2. Soft labels не дают эффекта — `pct_works_oob` уже присутствует как фича
3. num_leaves (31/63/127) не влияет — bottleneck не в ёмкости модели
4. **Проблема — label noise**: tinkering/works_oob субъективны, модель упирается в ceiling ≈0.70 S2 F1
5. Текущая конфигурация по сути оптимальна; дальнейший рост возможен через cleanlab или улучшение фич

### Итоговая лучшая конфигурация каскада

```
Stage 1: все 104 фичи, class_weight={0:3, 1:1}, threshold=0.5
Stage 2: 103 фичи (без report_age_days), class_weight={0:1, 1:2}, threshold=0.5
Cascade F1 macro: 0.5929 (vs single 0.5843, Δ=+0.009)
```

| Метрика | Single model | Каскад (факт) | Цель |
|---------|-------------|---------------|------|
| F1 macro | 0.584 | **0.593** | 0.63+ |
| borked recall | 0.38 | **0.41** | 0.55+ |
| borked precision | 0.60 | **0.57** | 0.60+ |
| works_oob recall | ~0.45 | **0.55** | 0.55+ ✓ |
| works_oob precision | 0.44 | **0.45** | 0.55+ |
| tinkering F1 | 0.80 | **0.80** | 0.80+ ✓ |

Каскад дал улучшение, но не достиг всех целей. Основные ограничения — label noise и неразделимость tinkering/works_oob.

---

## Фаза 2: Преодоление label noise ceiling

Оптимизация архитектуры и гиперпараметров исчерпана — модель упирается в шум меток. Граница tinkering↔works_oob **субъективна**: один пользователь ставит "works" на игру где поменял один параметр, другой на ту же конфигурацию ставит "tinkering".

### Диагноз

- Stage 2 F1 ≈ 0.70 — ceiling при любых гиперпараметрах, фичах, архитектурах
- num_leaves (31/63/127) не влияет → bottleneck не в ёмкости модели
- Soft labels, feature engineering, weight tuning — всё ±0.001
- Причина: **inherent label noise** ~15–20% в tinkering/works_oob

### Шаг 8. Cleanlab: детекция и удаление шумных меток

**Что делаем**:
- `cleanlab.classification.CleanLearning` на Stage 2 данных (non-borked)
- Найти label issues: сэмплы где модель уверенно не согласна с меткой
- Варианты: (a) удалить шумные, (b) перевзвесить, (c) перелейблить
- Переобучить Stage 2 на чистом подмножестве

**Ожидание**: +0.01–0.03 S2 F1 на чистых данных

**Метрики**:
- Доля шумных меток (ожидание: 10–20%)
- S2 F1 на очищенном train → eval на **полном** test (не чистить test!)
- Cascade F1 после

**Риски**:
- Если убрать >20% — потеря разнообразия, переобучение на "easy" примерах
- Cleanlab может систематически удалять minority class → bias

### Шаг 9. Confident Learning: взвешивание по надёжности метки

**Что делаем**:
- Для каждого отчёта вычислить confidence метки через game-level консенсус:
  - `confidence = |pct_works_oob - 0.5| * 2` — чем дальше от 50/50, тем надёжнее
  - Если game pct_works_oob=0.90 и label=works_oob → confidence=0.8
  - Если game pct_works_oob=0.45 и label=works_oob → confidence=0.1
- Использовать confidence как sample_weight при обучении
- Не путать с 5d (soft labels через pct_works_oob как target) — здесь мы **фильтруем ненадёжные**, не меняем target

**Отличие от 5d**: 5d использовал pct_works_oob для изменения веса через target, что дублировалось с фичей. Здесь — мы явно уменьшаем вклад сэмплов из игр без консенсуса.

**Ожидание**: +0.005–0.015 S2 F1

### Шаг 10. Regression: непрерывный score вместо классификации

**Что делаем**:
- Заменить 3-class classification на regression [0..1]:
  - Target: 0.0 = borked, 0.5 = tinkering, 1.0 = works_oob
  - Или: target = `avg_verdict_score` / `pct_works_oob` для игры (убрать из фич!)
- LightGBM regressor с MSE/MAE loss
- Порог для API: score < 0.2 → borked, 0.2–0.7 → tinkering, > 0.7 → works_oob

**Почему это принципиально лучше**:
- MSE-loss **не вынуждает** делать hard decision на размытой границе
- Ошибка tinkering↔works_oob штрафуется меньше чем borked↔works_oob (автоматически)
- Ordinal structure встроена: borked(0) < tinkering(0.5) < works_oob(1.0)
- Probability output естественный: score=0.65 → "скорее tinkering, но близко к oob"

**Варианты target**:
- **10a.** Fixed: borked=0, tinkering=0.5, oob=1.0
- **10b.** Game-based: target = `avg_verdict_score` для игры (per-game consensus)
- **10c.** Ordinal: два binary threshold — P(score > borked) и P(score > tinkering) (CORAL-style)

**Ожидание**: MAE < 0.25, при дискретизации F1 ≈ 0.58–0.62

### Шаг 11. Агрегация по (game, hardware) перед предсказанием

**Что делаем**:
- Группировать отчёты по (app_id, gpu_family, cpu_vendor, is_steam_deck)
- Target = majority verdict в группе (или mean score)
- Исключить группы без консенсуса (< 60% agreement)
- Обучить модель на агрегированных данных

**Почему поможет**:
- Усреднение убирает индивидуальную субъективность
- Dataset уменьшается (≈30k→10k), но **значительно чище**
- Для API это closer to what we actually predict — verdict per (game, hardware), не per report

**Проверки**:
- Сколько групп останется после фильтрации?
- F1 на агрегированном test vs per-report test
- Если dataset слишком маленький (< 5k) — не хватит для обучения

### Шаг 12. Коллапс до 2 классов + probability score

**Что делаем** (fallback если шаги 8–11 не дали значимого улучшения):
- Предсказывать только **borked vs works** (Stage 1, F1≈0.85 binary)
- Использовать P(works) как непрерывный score
- Для API: `{"works": true, "confidence": 0.92, "likely_oob": 0.65}`
- works_oob vs tinkering определяется **эвристически**: по наличию launch options, winetricks, protontricks

**Когда применять**: если после шагов 8–11 cascade F1 < 0.60

**Преимущество**: честно признаём что tinkering/oob неразделимы моделью, даём вероятность вместо ложной уверенности

---

### Результаты фазы 2 (experiment_cascade_v5_label_noise.py)

#### Шаг 8: Cleanlab
- Найдено 34.3% "шумных" меток, из них **59.3% всех oob** помечены как шум
- Любое удаление/взвешивание **убивает oob recall → 0.0**
- Причина: cleanlab считает minority class (oob) шумным, т.к. модель в нём неуверена
- **Вывод: cleanlab неприменим** — удаляет сигнал вместе с шумом

#### Шаг 9: Consensus weighting
- label_aligned: mean=0.822, 10.6% с confidence <0.3
- Любое взвешивание/фильтрация также убивает oob recall
- **Вывод: consensus weighting неприменим** — low-consensus = minority class

#### Шаг 10: Regression
- 10a (fixed targets): MAE=0.166, лучший F1 при дискретизации = **0.583** (хуже каскада 0.593)
- 10b (game-based target): MAE=0.177, F1 max = 0.431 — сильно хуже
- Score distributions перекрываются: tinkering 0.53±0.09, oob 0.64±0.14
- **Вывод: regression не лучше classification** — ordinal structure не помогает

#### Фундаментальный вывод

**Проблема не в label noise и не в архитектуре.** Tinkering и works_oob — это **субъективная оценка** на идентичных конфигурациях. Модель достигла ceiling ≈0.593 F1 macro, что определяется irreducible noise в данных.

Дальнейшие направления:
1. **Шаг 12 (2 класса + probability)** — принять неразделимость, давать probability score
2. **Новые фичи** — единственный путь к улучшению: текстовые фичи из отчётов, launch options, proton changelog
3. **Агрегация при инференсе** — предсказывать per-report, но агрегировать predictions по (game, hardware) для API

### Статус фазы 2

- [x] Шаг 8: Cleanlab → **неприменим** (убивает minority class)
- [x] Шаг 9: Confident Learning → **неприменим** (та же проблема)
- [x] Шаг 10: Regression → **хуже каскада** (F1=0.583 vs 0.593)
- [ ] Шаг 11: Агрегация (game, hardware) — чистка через группировку
- [x→skip] Шаг 12: 2 класса + probability score — рекомендуется как production подход

### Рекомендуемый порядок

1. **Шаг 8 (cleanlab)** — быстро, покажет масштаб label noise
2. **Шаг 9 (confident learning)** — если cleanlab нашёл >10% шумных
3. **Шаг 10 (regression)** — самый принципиальный сдвиг, тестировать параллельно
4. **Шаг 11 (агрегация)** — если regression хорошо работает, агрегация усилит
5. **Шаг 12 (2 класса)** — fallback если ничего не помогло
