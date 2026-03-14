# ML Pipeline — новые фичи из текстовых данных (фаза 5)

## Мотивация

Эксперименты фазы 4 показали что модель достигла ceiling ≈ 0.593 F1 macro при текущем feature set:
- Каскад (Stage 1 + Stage 2) дал +0.009 vs single model
- Cleanlab, consensus weighting, regression — не помогают
- Причина: tinkering/works_oob **неразделимы** при текущих фичах

**Единственный путь к улучшению** — новые фичи, несущие информацию о "степени работоспособности". В БД есть значительный объём неиспользованных текстовых данных.

> Фичи из LLM-предобработанных данных (`extracted_data`) вынесены в [PLAN_ML_5_LLM.md](PLAN_ML_5_LLM.md).

---

## Доступные данные (без LLM-предобработки)

### 1. Текстовые поля отчётов (raw)

| Поле | Покрытие | Средняя длина |
|------|----------|--------------|
| `concluding_notes` | 32% (106K) | 174 символов |
| `notes_verdict` | 74% | 54 символа |
| `notes_extra` | 10% | 196 символов |
| `notes_customizations_used` | 7% | 149 символов |
| `notes_*_faults` (6 типов) | 3-10% каждый | 86-131 символов |

### 2. Данные из enrichment (уже в БД)

- Steam store: жанры, категории, metacritic score
- PCGamingWiki: DRM, античит
- ProtonDB summary tier
- Steam Deck Verified status

---

## Результаты экспериментов

### Эксперимент 4: Группа D — текстовые мета-фичи ✅

| Фича | Описание |
|------|----------|
| `has_concluding_notes` | Есть ли concluding_notes |
| `concluding_notes_length` | Длина concluding_notes |
| `fault_notes_count` | Кол-во заполненных notes_*_faults |
| `has_customization_notes` | Есть ли notes_customizations |
| `total_notes_length` | Суммарная длина всех notes |

**Результат: F1 = 0.6696 (+0.077 vs baseline)**

Главный драйвер улучшения. `has_concluding_notes` и `total_notes_length` — top по feature importance.
Stage 1 logloss: 0.322 → 0.213 — огромное улучшение borked detection.

### Эксперимент 5: Группа E — keyword-фичи ✅

| Фича | Описание |
|------|----------|
| `mentions_crash` | crash/segfault/SIGSEGV/freeze/broken/unplayable |
| `mentions_fix` | fix/workaround/solved/protontricks/winetricks |
| `mentions_perfect` | perfect/flawless/no issues/just works |
| `mentions_proton_version` | proton N/ge-proton/proton experimental |
| `mentions_env_var` | паттерн `VAR=value` |
| `mentions_performance` | lag/stutter/fps/slow/choppy |
| `sentiment_negative_words` | кол-во негативных слов |
| `sentiment_positive_words` | кол-во позитивных слов |

**Результат: F1 = 0.6378 (+0.045 vs baseline)**

`mentions_perfect` — самая важная фича в Stage 1 (gain 1M+). Прямой сигнал.

### Эксперимент 5b: Группа D+E вместе ✅

**Результат: F1 = 0.6979 (+0.105 vs baseline)**

Комбинация D+E даёт почти всё улучшение.

### Эксперимент 6: Группа F — агрегированные текстовые фичи ✅

| Фича | Описание |
|------|----------|
| `pct_reports_with_notes` | Доля отчётов с concluding_notes |
| `avg_notes_length` | Средняя длина notes для игры |
| `pct_reports_mention_crash` | Доля отчётов с crash-keywords |
| `pct_reports_mention_fix` | Доля отчётов с fix-keywords |
| `pct_reports_with_faults` | Доля отчётов с notes_*_faults |
| `game_report_count` | Кол-во отчётов для игры |

**Результат: F1 = 0.5917 (−0.001 vs baseline) — бесполезно**

Per-game агрегация текста не несёт нового сигнала поверх существующих aggregated features.

### Эксперимент 6b: Все группы D+E+F ✅

**Результат: F1 = 0.6988 (+0.106 vs baseline)**

Группа F добавляет лишь +0.001 сверх D+E. Не стоит включать.

---

## Интеграция в production

**Интегрированы в `_build_feature_matrix` (train.py)**: Группы D + E (13 фич).
Группа F **не интегрирована** (бесполезна).

**Production cascade результат** (train_cascade_pipeline):
- **F1 macro: 0.692** (было 0.593, +0.099)
- **ECE: 0.014 → 0.006**
- **Confidence ≥ 0.7: 65%** данных, accuracy 89.6%
- **borked recall: 0.67** (было 0.41), **precision: 0.79**
- Features: 104 → 117

---

## Оставшиеся эксперименты

### Эксперимент 7: LLM-обработка текста (Группа G)

Задачи для batch LLM processing raw-текста:

| Фича | Источник | Задача для LLM | Ожидаемый сигнал |
|------|----------|----------------|------------------|
| `text_sentiment_score` | `concluding_notes` | Оценка тональности 0-1 | Прямой сигнал borked↔oob |
| `effort_level` | notes + customizations | Оценка усилий: none/low/medium/high | none → oob, high → tinkering |
| `issue_severity` | notes + faults | Серьёзность: none/minor/major/critical | critical → borked |
| `playability_score` | Все notes | Можно ли играть? 0-1 | Прямой сигнал для Stage 2 |
| `customization_complexity` | customizations | Сложность: none/simple/complex | complex → tinkering |

**Сложность**: высокая (batch LLM, стоимость ~106K записей)
**Ожидание**: +0.010–0.025

---

## Не реализуемо сейчас

- **Proton changelog**: не доступен в БД и через API Steam. ROI неясен — temporal signals покрыты `report_age_days`
- **Sentence embeddings** из concluding_notes: требует BERT/etc, непропорционально дорого
- **Cross-report фичи**: нет user_id в данных
