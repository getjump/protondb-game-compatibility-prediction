# Phase 7b: Дешёвые текстовые фичи (без LLM)

## Контекст

Stage 2 (tinkering vs works_oob) — bottleneck модели:
- Stage 1 AUC=0.983, LogLoss=0.098
- Stage 2 AUC=0.835, LogLoss=0.395

Текстовые фичи — второй по важности сигнал после game_emb:
- text_emb (32 SVD): ΔF1=−0.036 при удалении
- text_meta (5): ΔF1=−0.018
- keywords (8): ΔF1=−0.003

Текущие текстовые фичи (13 штук):
- 5 meta: has_concluding_notes, concluding_notes_length, fault_notes_count, has_customization_notes, total_notes_length
- 8 keywords: mentions_crash/fix/perfect/proton_version/env_var/performance + sentiment_negative/positive_words

## Данные: различия tinkering vs oob

### Статистика текста
| Метрика | tinkering | oob | borked |
|---|---|---|---|
| avg text length | 236 | 160 | 89 |
| avg word count | 38.9 | 28.0 | 16.7 |
| has_concluding_notes | 70.2% | 59.0% | 12.6% |
| has_customization_notes | 17.4% | 6.6% | 5.5% |
| avg_filled_note_fields | 2.04 | 1.68 | 1.58 |

### Сильные паттерны (отношение tinkering/oob)

**Маркеры tinkering (ratio > 1.5):**
| Паттерн | tinkering% | oob% | ratio |
|---|---|---|---|
| protontricks | 5.3% | 1.4% | 3.79 |
| winetricks | 1.0% | 0.3% | 3.98 |
| workaround | 1.3% | 0.4% | 3.09 |
| fsync | 0.5% | 0.1% | 3.77 |
| STEAM_COMPAT | 0.4% | 0.1% | 2.88 |
| wine | 7.4% | 3.1% | 2.36 |
| launch_options | 11.9% | 5.4% | 2.21 |
| env_var (=) | 12.3% | 5.8% | 2.13 |
| vkd3d | 0.8% | 0.4% | 2.10 |
| proton_ge | 6.0% | 3.0% | 2.00 |
| PROTON_* env | 51.5% | 28.8% | 1.79 |
| dxvk | 2.3% | 1.4% | 1.65 |
| proton_experimental | 14.0% | 8.5% | 1.66 |

**Маркеры oob (ratio < 0.8):**
| Паттерн | tinkering% | oob% | ratio |
|---|---|---|---|
| out_of_box | 3.1% | 12.0% | 0.26 |
| just_works | 0.3% | 1.2% | 0.27 |
| no_issues | 2.2% | 4.1% | 0.55 |
| flawless | 4.2% | 5.5% | 0.76 |
| perfectly | 8.1% | 10.4% | 0.78 |

## Предложения: новые дешёвые фичи

### Группа 1: Effort/tinkering маркеры (regex, ~10 фич)

Новые keyword бинарные фичи с высоким discrimination ratio:

```python
# Высокий tinkering signal
mentions_protontricks    # protontricks|winetricks  (ratio ~3.9)
mentions_workaround      # workaround|work.?around  (ratio 3.1)
mentions_launch_options  # launch.?option            (ratio 2.2)
mentions_wine_tools      # wine(?!dows)|lutris       (ratio 2.3)
mentions_dll_override    # winedlloverrides|dll      (ratio 1.7)
mentions_proton_ge       # GE.?Proton|proton.?ge     (ratio 2.0)

# Высокий oob signal
mentions_out_of_box      # out.of.the.box|just.works|no.issues  (ratio 0.3)
mentions_flawless        # flawless|perfectly|works.great       (ratio 0.77)
```

**Стоимость:** ~0.1ms на отчёт, regex compile один раз.
**Ожидаемый эффект:** Дополнят существующие 8 keyword фич, закрывая пробел в effort detection.

### Группа 2: Расширенные текстовые метрики (5 фич)

```python
customization_notes_length  # len(notes_customizations) — уже 2.6x разница tink/oob
word_count                  # Количество слов (39 vs 28 — 1.4x)
filled_note_fields_count    # Количество заполненных полей (2.04 vs 1.68)
has_launch_flags_notes      # notes_launch_flags заполнено (tinkering-маркер)
concluding_to_total_ratio   # Доля concluding_notes от общего текста
```

**Стоимость:** ~0.05ms на отчёт.
**Ожидаемый эффект:** filled_note_fields — сильный effort proxy (больше полей = больше тинкерил).

### Группа 3: Verb/action density (3 фичи)

```python
# Подсчёт "action" слов — маркер effort
action_word_count    # set|install|add|change|switch|download|enable|disable|use|run|try|configure
                     # Нормированный на word_count → action_word_density
technical_term_count # proton|steam|wine|vulkan|dx11|dx12|opengl|dxvk|vkd3d|mesa|nvidia|amd
                     # → technical_term_density (на word_count)
question_mark_count  # Количество "?" — может коррелировать с проблемами
```

**Стоимость:** ~0.2ms на отчёт.
**Ожидаемый эффект:** Captures tinkering effort даже без конкретных tool names.

### Группа 4: TF-IDF micro-embeddings (8-16 фич)

Вместо полных SVD text embeddings (32 dims на notes_verdict), добавить маленький TF-IDF на **всех** текстовых полях:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# На этапе обучения:
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                         min_df=5, max_df=0.5, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(all_texts)  # all_text = concat всех note полей
svd = TruncatedSVD(n_components=16)
X_reduced = svd.fit_transform(X_tfidf)
# → text_full_emb_0..text_full_emb_15
```

Отличие от текущих text_emb: те строятся только на notes_verdict, а тут — **весь текст** (customizations, faults, concluding).

**Стоимость:** fit ~5s на 350K отчётов, transform ~0.1ms на отчёт.
**Ожидаемый эффект:** Может заменить или дополнить text_emb, захватывая сигнал из notes_customizations.

### Группа 5: Per-field embeddings (опционально, 8-16 фич)

Отдельные мини-SVD для customization-специфичного текста:

```python
# notes_customizations + notes_launch_flags → "effort" embedding (8 dims)
# concluding_notes → "summary" embedding (8 dims)
```

**Стоимость:** ~10s fit, ~0.1ms transform.
**Ожидаемый эффект:** Раздельные embeddings для разных семантических аспектов.

## Приоритеты реализации

| # | Группа | Фич | Стоимость | Ожидание | Риск |
|---|---|---|---|---|---|
| 1 | Effort keywords | ~8 | Нулевая | Высокое | Низкий |
| 2 | Расширенные метрики | ~5 | Нулевая | Среднее | Низкий |
| 3 | Action/tech density | ~3 | Нулевая | Среднее | Низкий |
| 4 | TF-IDF full-text SVD | 8-16 | 5s train | Высокое | Средний (может конфликтовать с text_emb) |
| 5 | Per-field embeddings | 8-16 | 10s train | Среднее | Средний |

## План действий

1. **Шаг 1:** Добавить группы 1-3 (~16 фич), замерить F1
2. **Шаг 2:** Если улучшение < 0.005, добавить группу 4 (TF-IDF full-text)
3. **Шаг 3:** Ablation — убрать фичи с ΔF1 < 0.001
4. **Шаг 4:** Рассмотреть замену text_emb (notes_verdict SVD) на full-text SVD если покрытие лучше

## Inference-time соображения

Все фичи из групп 1-3 — чистый regex/подсчёт, работают на inference без модели.
Группы 4-5 требуют сохранённый TF-IDF vectorizer + SVD в артефактах (pickle, ~2MB).
Для новых отчётов при inference: transform через сохранённый pipeline.

**Для cold-start** (нет текста пользователя):
- Keyword/metric фичи = 0/None
- TF-IDF embeddings = нулевой вектор
- Модель должна быть обучена с ~10% missing text для робастности
