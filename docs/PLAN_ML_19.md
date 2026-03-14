# Phase 19: Data quality — filtering, LLM extraction, label reconstruction

## Контекст

F1=0.7801. Анализ показал фундаментальную проблему данных:

**60.8% training labels — inferred, не observed:**

| Label source | Count | % | Reliability |
|---|---|---|---|
| `tinkering_inferred` (verdict=yes, oob=NULL) | 211,769 | 60.8% | **Низкая** — assumed tinkering |
| `borked` (verdict=no) | 68,745 | 19.7% | Высокая |
| `works_oob` (verdict_oob=yes) | 38,449 | 11.0% | Высокая |
| `tinkering_explicit` (verdict_oob=no) | 29,573 | 8.5% | Высокая |

ProtonDB добавил поле `verdict_oob` ~3-4 года назад. Все старые отчёты с verdict=yes автоматически считаются tinkering, хотя многие из них — works_oob.

**В train set:** только 17% имеют verdict_oob. 83% Stage 2 labels — inferred.
**В test set:** 31% имеют verdict_oob. Модель тестируется на данных лучшего качества чем обучается.

**LLM extraction:** pipeline существует (`extract/extractor.py`, промпты готовы), но `extracted_data` = 0 records. Ни разу не запускался.

---

## Phase 19.1 — Train Stage 2 только на explicit labels (1 день, +0.010-0.030 F1)

### Убрать inferred tinkering из Stage 2 training

**Что:** Stage 2 обучается только на reports с explicit verdict_oob (68K из 348K). Stage 1 по-прежнему на всех данных (borked detection не зависит от oob).

**Алгоритм:**
```python
# Stage 2: filter to reports with verdict_oob IS NOT NULL
s2_mask = train_report_has_oob  # ~17% of train
X_train_s2 = X_train[s2_mask & (y_train > 0)]
y_train_s2 = y_train[s2_mask & (y_train > 0)]
# Train on ~12K real tinkering/oob labels instead of ~200K inferred
```

**Гипотеза:** 83% Stage 2 training data — noise (inferred tinkering). Убрав их:
- Чище labels → лучше boundary
- Меньше данных, но higher quality
- IRT будет точнее на clean labels

**Риск:** 12K samples vs 200K — может underfit. Mitigation: увеличить regularization, или use inferred labels с пониженным weight.

**Варианты:**
- (a) Только explicit labels (hard filter)
- (b) Explicit labels weight=1.0, inferred weight=0.3 (soft filter)
- (c) Explicit labels + IRT-relabeled inferred (trusted subset)

**Инференс:** Да — та же модель.
**Эффект:** High (+0.010-0.030 F1). Устраняет главный источник noise в Stage 2.
**Стоимость:** ~20 строк.

---

## Phase 19.2 — Temporal filtering (0.5 дня, +0.005-0.010 F1)

### Обучение только на последних 3-4 годах

**Что:** Отфильтровать отчёты старше 4 лет из train set. Эти отчёты:
- Не имеют verdict_oob (0% coverage)
- Написаны на Proton 3-5 (irrelevant для предсказания на Proton 8-9)
- borked rate 25-27% vs 14% в новых → temporal shift

**Фильтр:** `report_age_days < 1460` (~4 года). Убирает ~108K из 348K reports (31%).

**Комбинация с 19.1:** temporal filter + explicit oob labels only → Stage 2 обучается на ~40K чистых recent labels.

**Эффект:** Medium (+0.005-0.010 F1).
**Стоимость:** ~10 строк.

---

## Phase 19.3 — LLM extraction pipeline (2-3 дня, +0.010-0.020 F1)

### Запуск text extraction и использование для label reconstruction

**Состояние:** Pipeline существует, промпты готовы, но `extracted_data` = 0. Нужно запустить.

**Что извлекает LLM:**
```python
ExtractionResult:
    actions: [Action(type, value, effect, conditions)]  # env_var, protontricks, dll_override...
    observations: [Observation(symptom, description)]   # crash_on_launch, black_screen...
    useful: bool
```

**Новые фичи из extracted_data:**

1. `has_effective_actions` (binary) — есть ли effective customizations
2. `action_count` — количество actions
3. `action_types` — типы (env_var, protontricks, runner_selection...)
4. `has_crash_symptom` — наблюдал ли crash
5. `has_audio_issue` — audio problems
6. `effort_score` — combined score из количества и типов actions

**Label reconstruction (главная ценность):**

Для 211K inferred tinkering reports:
- LLM извлекает actions из текста
- Если actions = 0 или только runner_selection → **relabel to works_oob**
- Если actions include protontricks/env_var/dll_override → **confirm tinkering**
- Это расширение Phase 8 relabeling на 211K reports (сейчас Phase 8 работает только на regex, не на full text understanding)

**Конфигурация LLM:**
- Нужен local LLM (ollama/llama.cpp) или cloud API
- Batch processing: ~3-5 reports/call
- 112K reports с текстом × 1 call/3 reports ≈ ~37K LLM calls
- При 10 calls/sec (local) ≈ ~1 час
- При 1 call/sec (cloud) ≈ ~10 часов

**Стоимость:** 2-3 дня (запуск pipeline + feature engineering + label reconstruction).
**Эффект:** High (+0.010-0.020 F1). LLM-based relabeling на 211K reports >> regex relabeling на 29K.

---

## Phase 19.4 — LLM-based verdict inference for old reports (2 дня, +0.005-0.015 F1)

### LLM присваивает verdict_oob старым reports

**Что:** Для 211K reports без verdict_oob — использовать LLM для inference:

**Prompt:**
```
Given this ProtonDB report, did the user need to do significant
customization to make the game work, or did it work out of the box?

Report text: {concluding_notes}
Active customizations: {cust_winetricks}, {cust_protontricks}...
Launch flags: {flag_*}

Answer: "works_oob" or "needs_tinkering"
```

**Отличие от Phase 19.3:** Phase 19.3 извлекает structured actions. Phase 19.4 напрямую спрашивает LLM о verdict.

**Можно делать в два этапа:**
1. Сначала Phase 19.3 (extraction) → структурированные actions
2. Затем Phase 19.4 — rule-based verdict inference из extracted actions (без дополнительных LLM calls)

**Эффект:** Medium (+0.005-0.015 F1).

---

## Phase 19.5 — Enrichment доработки для LLM (1 день)

### Что нужно доработать в enrichment для поддержки LLM extraction

**Текущее состояние:**
- `protondb_settings/preprocessing/extract/` — полный pipeline (extractor, spotter, validator, models)
- `protondb_settings/preprocessing/llm/` — LLM client + prompts (GPU norm, CPU norm, launch options, text extraction)
- CLI: `protondb-settings preprocess llm extract` — команда существует

**Нужные доработки:**

1. **DB schema:** `extracted_data` таблица — проверить что migration создаёт её

2. **Progress tracking:** extraction на 112K reports = долго. Нужно:
   - Resume from interruption (уже есть через pipeline_runs)
   - Progress bar (уже есть через PipelineStep)
   - Batch size configuration (EXTRACT_BATCH_LOCAL=3)

3. **LLM configuration:** `.env` уже имеет:
   ```
   OPENAI_BASE_URL=http://localhost:11434/v1
   MODEL=qwen3.5:latest
   ```
   Нужен работающий LLM backend.

4. **Post-extraction merge:** результаты из `extracted_data` → фичи в `_build_feature_matrix`:
   - JOIN reports с extracted_data
   - Compute action-based features
   - Add to feature matrix

**Стоимость:** 1 день infrastructure.

---

## Phase 19.6 — Combined: filtered Stage 2 + LLM relabeling (1 день, +0.015-0.035 F1)

### Объединение всех data quality improvements

**Pipeline:**
```
1. LLM extraction → extracted_data (112K reports)
2. Label reconstruction:
   - 29K explicit tinkering: IRT contributor-aware relabel (Phase 13.2)
   - 211K inferred tinkering: LLM-based relabel (from extracted actions)
   - 38K works_oob: keep
   - 69K borked: keep
3. Temporal filter: drop reports > 4 years old
4. Stage 2: train on explicit + LLM-relabeled labels only
5. IRT re-fit on clean labels → better θ and d
```

**Ожидание:**
- Stage 2 boundary noise: 60.8% inferred → ~10% (после LLM relabeling)
- More works_oob in train (currently 9.9%) → better oob recall
- IRT на cleaner labels → more accurate difficulty scores

**Эффект:** High (+0.015-0.035 F1). Комбинация всех data quality improvements.

---

## Порядок реализации

```
Phase 19.1 (1 день):   Explicit-only Stage 2              → +0.010-0.030 F1
Phase 19.2 (0.5 дня):  Temporal filtering                  → +0.005-0.010 F1
Phase 19.5 (done):     Enrichment доработки для LLM        → infrastructure ✅
Phase 19.3 (TBD):      LLM extraction pipeline             → TBD
Phase 19.4 (done):     LLM verdict inference               → −0.001 F1 ❌
Phase 19.6 (1 день):   Combined                            → +0.015-0.035 F1
                                                      Итого: +0.020-0.050 F1
```

**Приоритет:** 19.1 → 19.2 → 19.1+19.2 experiment → 19.5 → 19.3 → 19.6

19.1 — моментальный эксперимент: убрать inferred labels из Stage 2.
19.2 — простой фильтр по возрасту.
19.3-19.6 — зависят от LLM backend (нужна настройка).

---

## Зависимости

- **19.1, 19.2:** Нет зависимостей, можно запускать сейчас
- **19.3-19.6:** Нужен работающий LLM (local ollama или cloud API)
- IRT (Phase 12.8) — для re-fit на чистых labels
- `extract/` pipeline — уже реализован, но не запущен

---

## Метрики

| Метрика | Current | Цель Phase 19 |
|---|---|---|
| F1 macro | 0.7801 | **> 0.82** |
| works_oob F1 | 0.614 | **> 0.70** |
| works_oob recall | 0.607 | **> 0.70** |
| borked F1 | 0.846 | **≥ 0.84** |

## Ключевой вывод

Phase 12-18 оптимизировали модель на noisy данных. Phase 19 чистит сами данные:

**60.8% Stage 2 labels — inferred noise.** Это не random noise и не annotator bias (IRT уже это решил). Это systematic bias: старые reports без verdict_oob = assumed tinkering. Многие из них — works_oob.

Два пути чистки:
1. **Фильтрация** (19.1, 19.2) — убрать noisy samples (меньше данных, чище labels)
2. **Реконструкция** (19.3, 19.4) — LLM восстанавливает verdict_oob из текста (больше данных, чище labels)

Оба пути аддитивны к IRT (Phase 12-13). IRT чистит annotator bias. Phase 19 чистит systematic label absence bias.
