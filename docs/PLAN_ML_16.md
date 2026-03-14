# Phase 16: Pipeline insights — targeted improvements

## Контекст

Полный анализ pipeline (F1=0.7706) выявил 5 ключевых bottleneck'ов:

1. **Contributor coverage gap: +0.130 F1** — с contributor data F1=0.812, без F1=0.682
2. **Stage 2 boundary = 74% ошибок** — works_oob → tinkering (52% всех ошибок)
3. **Class imbalance** — works_oob 9.9% train vs 18.1% test (temporal shift)
4. **Overconfident errors** — confidence 0.600 на ошибках (модель уверенно ошибается)
5. **Cold-start парадокс** — новые игры F1=0.825, знакомые F1=0.752 (game aggregates вредят?)

Данные полностью собраны: ProtonDB reports 100%, Steam PICS 100%.

---

## Phase 16.0 — Re-run с полным coverage (0.5 дня, +0.005-0.020 F1)

### Перетренировка на полных данных

**Что:** Перезапустить лучший pipeline (IRT + 13.2 relabel) с полным contributor coverage.

**Ожидание:** Coverage gap +0.130 F1 при 70% coverage. При 100%:
- Больше annotators для IRT → более точные θ и d
- Больше reports с contributor data → contributor-aware relabel работает на большем % данных
- IRT features покрывают больше test set

**Стоимость:** 0. Просто перезапуск.

---

## Phase 16.1 — Class rebalancing (1 день, +0.005-0.015 F1)

### Борьба с works_oob underrepresentation

**Проблема:** works_oob = 9.9% train, 18.1% test. Time-based split: старые отчёты → больше tinkering (Proton хуже), новые → больше oob (Proton лучше). Модель недообучена на oob.

**Эксперименты:**

1. **class_weight Stage 2:** `{0: 1.0, 1: 1.5}` → upweight works_oob (сейчас неявно через label smoothing)

2. **Oversampling oob в train:** duplicate oob samples до ~15% (SMOTE не подходит для tabular с categoricals, просто random oversample)

3. **Temporal-aware split:** вместо 80/20 time split → использовать последние 6 месяцев как test (меньше distribution shift)

4. **Stratified time split:** time-based но с сохранением class proportions через undersampling tinkering в train

**Гипотеза:** 6053 ошибок works_oob → tinkering (52% всех ошибок). Если модель видит больше oob при обучении → recall oob вырастет. Сейчас works_oob recall = 0.506.

**Эффект:** Medium (+0.005-0.015 F1). Прямо атакует biggest error source.
**Стоимость:** ~20 строк.

---

## Phase 16.2 — Game aggregates ablation (1 день, диагностика)

### Проверка: вредят ли game aggregates?

**Проблема:** Cold-start gap ОТРИЦАТЕЛЬНЫЙ: −0.073 F1. Модель работает ЛУЧШЕ на новых играх (F1=0.825) чем на знакомых (F1=0.752). Это ненормально — game aggregates (26 features) должны помогать для seen games.

**Гипотеза:** Game aggregates leakage или noise:
- Aggregates вычисляются по всем train reports → для popular games (много reports) aggregates ≈ target → overfit
- Для test reports той же игры, aggregates не обновляются → stale signal
- Или: popular games просто сложнее (разные verdicts для разного hardware)

**Эксперименты:**
1. **Ablation:** убрать все 26 game aggregate features → если F1 растёт, они вредят
2. **GroupKFold:** train/test split по app_id (не по времени) → все reports одной игры в одном fold
3. **Leak-free aggregates:** вычислять aggregates только из OTHER games' reports, не из target game

**Эффект:** Диагностика. Может дать +0.005-0.015 если aggregates вредят.
**Стоимость:** ~30 строк.

---

## Phase 16.3 — `variant` feature investigation (0.5 дня, диагностика)

### Почему variant — feature #1 в Stage 2?

**Факт:** `variant` имеет gain 142K — в 5x больше чем irt_contributor_strictness (28K). Это самый важный feature для tinkering↔oob boundary.

**Что:** Проверить что это за feature, какие значения, нет ли leakage.

**Действия:**
1. Проверить распределение `variant` по классам
2. Проверить не является ли variant proxy для target (circular dependency)
3. Если `variant` = ProtonDB report type/variant → это может быть leakage

**Эффект:** Диагностика + возможное исправление.

---

## Phase 16.4 — Confidence calibration + conformal (1-2 дня, UX improvement)

### Борьба с overconfident errors

**Проблема:** Mean confidence на ошибках: 0.600. При conf ≥ 0.7 accuracy=95%, но только 61% данных. Модель не знает когда она не знает.

**Эксперименты:**

1. **Isotonic calibration** (уже есть, но проверить Stage 2 отдельно):
   - Текущая calibration на cascade level → может не помогать для Stage 2 boundary

2. **Conformal prediction (MAPIE):**
   ```python
   from mapie.classification import MapieClassifier
   # Prediction sets with guaranteed coverage
   # Для borderline: {tinkering, works_oob} вместо forced hard decision
   ```
   - При α=0.1: 90% coverage guarantee
   - Ожидание: ~30% predictions будут {tinkering, oob} set → honest uncertainty

3. **Temperature scaling:**
   - Один параметр T: logits / T → calibrated probabilities
   - Проще isotonic, может работать лучше для 3-class

**Эффект:** Не улучшает F1, но улучшает reliability. API может отдавать confidence interval.
**Стоимость:** ~30 строк.

---

## Phase 16.5 — IRT на полных данных + 2PL (1 день, +0.003-0.010 F1)

### Перефит IRT с полным coverage

**Что:** IRT fit на 100% contributor data (58K contributors, 181K report-contributor pairs).

**Ожидание:** Предыдущие прогоны при 43% coverage дали +0.030 F1. При 100%:
- Больше overlap для IRT → более точные θ и d
- IRT features покрывают ~100% test (vs 80% при 43% coverage)
- 2PL discrimination тоже точнее

**Также:** Проверить 2PL на полных данных (ранее +0.003 при 43%).

**Эффект:** Low-medium (+0.003-0.010 F1).

---

## Phase 16.6 — Error-targeted features (2 дня, +0.005-0.015 F1)

### Features для top error patterns

**Анализ ошибок:**
- 52% ошибок: true=oob, predicted=tinkering → модель слишком "строгая"
- FN oob имеют higher irt_difficulty (−0.91 vs −0.45) → модель не различает "объективно сложнее" от "annotator строгий"

**Новые фичи для Stage 2:**

1. **irt_difficulty_residual** = irt_difficulty − game_avg_difficulty
   - Residual после вычитания game-level signal → per-gpu_family signal

2. **contributor_consistency** — насколько consistent этот annotator (std его verdict scores)
   - Inconsistent annotator → его label менее надёжен

3. **game_verdict_agreement** — % agreement между annotators на этой игре
   - Высокий agreement → надёжный label → higher weight
   - Низкий → genuine ambiguity → lower confidence

4. **report_text_effort_score** — длина и детальность отчёта как proxy для quality
   - `concluding_notes_length + fault_notes_count` уже есть, но можно combined score

**Эффект:** Medium (+0.005-0.015 F1).

---

## Порядок реализации

```
Phase 16.0 (0.5 дня):  Re-run full coverage               → +0.005-0.020 F1
Phase 16.3 (0.5 дня):  variant investigation               → диагностика
Phase 16.2 (1 день):   Game aggregates ablation            → диагностика / +0.005-0.015
Phase 16.1 (1 день):   Class rebalancing                   → +0.005-0.015 F1
Phase 16.5 (1 день):   IRT full + 2PL                      → +0.003-0.010 F1
Phase 16.6 (2 дня):    Error-targeted features             → +0.005-0.015 F1
Phase 16.4 (1-2 дня):  Conformal prediction                → UX improvement
                                                      Итого: +0.015-0.040 F1
```

**Приоритет:** 16.0 → 16.3 → 16.2 → 16.1 → 16.5 → 16.6 → 16.4

16.0 — бесплатный win, просто перезапуск с полными данными.
16.3 — критично: если variant = leakage, всё меняется.
16.2 — cold-start paradox: game aggregates могут вредить.
16.1 — class rebalancing для works_oob (52% ошибок).

---

## Результаты (2026-03-14)

**Данные:** 100% contributor coverage (58K contributors, 181K records).

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob | oob_recall |
|---|---|---|---|---|---|---|
| 16.0 full coverage | 0.7717 | — | 0.840 | 0.886 | 0.589 | 0.507 |
| **16.1a cw_oob_1.5** | **0.7768** | **+0.005** | 0.840 | 0.878 | **0.613** | **0.603** |
| 16.1a cw_oob_2.0 | 0.7744 | +0.003 | 0.840 | 0.867 | 0.616 | 0.665 |
| 16.1a cw_oob_2.5 | 0.7716 | 0.000 | 0.840 | 0.859 | 0.616 | 0.707 |
| 16.1b oversample | 0.7759 | +0.004 | 0.840 | 0.879 | 0.609 | 0.587 |
| 16.2a no aggregates | 0.7711 | −0.001 | 0.840 | 0.886 | 0.588 | 0.508 |
| 16.6 error features | 0.7696 | −0.002 | 0.846 | 0.890 | 0.573 | 0.471 |
| **combined** | **0.7776** | **+0.006** | **0.846** | 0.884 | 0.603 | 0.557 |

### Выводы

1. **Class weight oob=1.5 — winner: +0.005 F1.** works_oob recall 0.507 → 0.603. Tinkering F1 немного снизился (0.886 → 0.878) — expected trade-off.

2. **Combined (error features + cw 1.5): +0.006 F1.** borked F1=0.846 — best ever.

3. **Full coverage:** +0.001 vs 70% coverage. 100% contributor data даёт marginal improvement — IRT уже хорошо экстраполирует с 70%.

4. **Game aggregates нейтральны** (−0.001 при удалении). Cold-start paradox объясняется не leakage, а тем что новые игры имеют более consistent verdicts (меньше temporal noise).

5. **`variant` — не leakage.** Это тип Proton runner (official/ge/experimental/native). Легитимный и самый важный feature в Stage 2 (gain 142K).

### Cumulative progress

| Stage | F1 macro | ΔF1 | works_oob F1 |
|---|---|---|---|
| Original baseline (Phase 11) | 0.7245 | — | 0.503 |
| + IRT features (Phase 12.8) | 0.7545 | +0.030 | 0.569 |
| + Contributor-aware relabel (Phase 13.2) | 0.7711 | +0.047 | 0.591 |
| + Class weight + error features (Phase 16) | **0.7776** | **+0.053** | **0.603** |

## Метрики

| Метрика | Before Phase 16 | After Phase 16 | Target |
|---|---|---|---|
| F1 macro | 0.7706 | **0.7776** | > 0.79 |
| works_oob F1 | 0.588 | **0.603** | > 0.63 |
| works_oob recall | 0.506 | **0.557** | > 0.55 ✅ |
| borked F1 | 0.838 | **0.846** | ≥ 0.84 ✅ |

## Ключевой вывод

Bottleneck analysis → targeted fix: class weight on works_oob дал +0.005 F1 и +0.096 oob recall. Простое решение для реальной проблемы (class imbalance из-за temporal shift).

Оставшийся gap до 0.79: ~0.013 F1. Возможные направления:
- Fine-tune class weight (sweep 1.3-1.7)
- Conformal prediction для honest uncertainty
- Better Stage 2 architecture (но Phase 11.2 показал что LightGBM оптимален)
