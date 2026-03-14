# ML Pipeline — план улучшений (фаза 3)

## Текущие результаты (2026-03-11)

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.6973 |
| F1 macro | 0.5843 |
| Features | 104 |
| Samples | 348 536 (278 828 train / 69 708 test) |
| Гиперпараметры | n_est=2000, lr=0.03, leaves=63, class_weight={0:2, 1:1, 2:3} |

### Per-class метрики

| Класс | Доля | Precision | Recall | F1 |
|-------|------|-----------|--------|-----|
| borked | 14.1% | 0.60 | 0.38 | 0.46 |
| needs_tinkering | 67.6% | 0.80 | 0.80 | 0.80 |
| works_oob | 18.1% (?) | 0.44 | 0.56 | 0.49 |

### Confusion matrix (нормализованная)

```
                   Predicted
              borked  tinkering  works_oob
Actual borked  37.6%    44.8%     17.6%
       tinker   4.2%    80.2%     15.6%
       oob      3.6%    40.7%     55.7%
```

---

## Диагностика — ключевые проблемы

### Проблема 1: `report_age_days` доминирует для works_oob (SHAP = 2.7)

`report_age_days` — топ-1 фича overall (SHAP 1.07 mean) и абсолютный доминант для класса works_oob (SHAP 2.7). Это **temporal bias**: модель выучила что старые отчёты → works_oob, новые → tinkering. Причина: в ранние периоды ProtonDB преобладали оптимисты и простые игры, позже пришли сложные игры и больше borked-отчётов.

**Почему это проблема**: при inference новые отчёты всегда будут "молодыми" → модель систематически занижает works_oob.

### Проблема 2: works_oob error rate 56.4%

Больше половины предсказаний works_oob ошибочны. Главный поток: tinkering→works_oob (7387 случаев). Модель слишком оптимистична — вероятно из-за `class_weight=3.0` для works_oob.

### Проблема 3: Недокалибровка works_oob

При predicted P(works_oob)=0.5 реальная доля ~0.25. Модель завышает вероятность works_oob в 2 раза в среднем диапазоне.

### Проблема 4: borked recall 37.6%

44.8% реально borked игр предсказываются как tinkering. Модель недостаточно уверенно идентифицирует сломанные игры.

### Проблема 5: Размытая граница tinkering↔works_oob

Probability distributions сильно перекрываются. Это может быть фундаментальным ограничением данных — один и тот же человек может поставить "works" или "tinkering" в зависимости от настроения.

### Проблема 6: `total_reports × game_emb_0` — сильнейшее взаимодействие (SHAP interaction 0.025)

SHAP interaction analysis показал что `total_reports` модулирует эффект `game_emb_0`: для игр с малым числом отчётов эмбеддинг ненадёжен (мало данных для SVD), но модель всё равно на него опирается. Это источник ошибок на редких играх.

### Проблема 7: works_oob recall деградирует на новых данных (0.80 → 0.45)

Temporal analysis по 10 бакетам теста подтвердил: на самых старых данных works_oob recall = 0.80, на новейших = 0.45. При этом accuracy и class distribution стабильны. Модель использует `report_age_days` как shortcut: старый отчёт = works_oob. На реальных данных при inference это не будет работать.

### Проблема 8: Ошибки tinkering↔works_oob не кластеризуются в feature space

UMAP error clusters показал: ошибки need→work и work→need сконцентрированы в одном плотном кластере, где все три класса перемешаны. Это не проблема отсутствующих фич — это фундаментальная неразделимость классов в этой зоне. Модель не может провести границу tinkering/works_oob для ~40% данных.

### Проблема 9: Зоны высокой uncertainty совпадают с зоной tinkering/oob

Confidence map показал чёткую структуру: изолированные кластеры (borked, специфичные игры) — высокая confidence. Большой центральный кластер — низкая confidence, высокая entropy. Модель "знает что не знает", но всё равно вынуждена выбирать.

### Проблема 10: `pct_stability_faults` — недоиспользованный сигнал для borked

Feature distributions показали: borked отчётливо смещён вправо по `pct_stability_faults` (SHAP 0.050). Но фича aggreрированная по всем отчётам (data leakage). Если фиксировать на train period — может стать ещё сильнее как честный сигнал.

---

## План экспериментов

### 1. Пересмотреть `report_age_days`

**Проблема**: temporal bias, доминирование одной фичи.

**Эксперименты**:
- **1a.** Заменить `report_age_days` на **`proton_era`** — дискретные эпохи по версии Proton (до 5.x, 5.x-6.x, 7.x, 8.x, 9.x). Это сохраняет полезный сигнал (новый Proton = лучше) без прямой привязки к календарной дате.
- **1b.** **Нормализовать относительно игры**: `report_age_relative = (max_ts_game - ts) / (max_ts_game - min_ts_game)`. Показывает "ранний или поздний отчёт для этой игры", а не абсолютный возраст.
- **1c.** **Убрать совсем** — проверить baseline без temporal фич. Если accuracy упадёт менее чем на 2%, значит фича добавляла больше bias чем сигнала.
- **1d.** **Clip/cap**: ограничить сверху, например 365 дней. Старые vs очень старые — одно и то же.

**Ожидание**: F1 может упасть на тесте (temporal bias "помогал" на тесте), но модель станет честнее при inference. Проверять на ручных примерах.

**Приоритет**: 🔴 Высокий — это архитектурная проблема, влияет на всё.

### 2. Тюнинг `class_weight` для works_oob

**Проблема**: error rate 56.4% для works_oob предсказаний. weight=3.0 порождает false positives.

**Эксперименты**:
- **2a.** Снизить `class_weight` works_oob: {0:2, 1:1, 2:**2.0**} и {0:2, 1:1, 2:**2.5**}
- **2b.** Оптимизировать веса по F1 macro через grid search: {0: [1.5, 2.0, 2.5], 1: [1.0], 2: [1.5, 2.0, 2.5, 3.0]}
- **2c.** Вместо `class_weight` попробовать **focal loss** (параметр γ=1..3) — штрафует лёгкие примеры, фокусируется на трудных

**Ожидание**: снижение false positive rate для works_oob при сохранении recall borked.

**Приоритет**: 🟡 Средний — быстрый эксперимент, прямое влияние на precision.

### 3. Post-hoc калибровка

**Проблема**: predicted probabilities != actual frequencies, особенно для works_oob и borked.

**Эксперименты**:
- **3a.** **Isotonic regression** на hold-out validation set (отщепить 10% из train)
- **3b.** **Platt scaling** (logistic regression на logits)
- **3c.** **Temperature scaling** — один параметр T, делим logits на T

**Ожидание**: не улучшит accuracy/F1 (hard predictions), но сделает вероятности достоверными для API (пользователь видит "75% вероятность что заработает" — это реально 75%).

**Приоритет**: 🟡 Средний — для production API важно, для метрик нет.

### 4. Feature engineering: альтернативные агрегации

**Проблема**: per-game aggregated features (pct_*_faults, total_reports) используют ВСЕ отчёты включая тестовые → потенциальный data leakage.

**Эксперименты**:
- **4a.** **Фиксировать агрегации на train period** — считать pct_* только по отчётам до split point. Для test-отчётов использовать только "историю" до момента этого отчёта.
- **4b.** **Заменить `total_reports` на log(total_reports)** — сейчас сильно скошен (медиана ~3, макс ~2500). Log-шкала может помочь.
- **4c.** **Добавить `reports_last_year`** — количество отчётов за последний год. Живая vs мёртвая игра — разный сигнал.

**Ожидание**: 4a может снизить метрики на тесте (убрали "подсказку"), но улучшит generalization. 4b — простой фикс, может дать +0.5% F1.

**Приоритет**: 🔴 Высокий (4a — data leakage), 🟢 Низкий (4b, 4c).

### 5. Шумность меток: фильтрация / перевзвешивание

**Проблема**: граница tinkering↔works_oob субъективна. Один человек ставит "works", другой — "tinkering" для одной игры.

**Эксперименты**:
- **5a.** **Confident Learning** (cleanlab): найти "ошибочные" метки, убрать или перевзвесить. Если 20%+ меток шумные — можно существенно улучшить.
- **5b.** **Soft labels**: вместо hard target (0/1/2) использовать per-game vote distribution [0.1, 0.6, 0.3]. Для игры где 60% tinkering и 40% works_oob — soft label честнее чем hard "tinkering". Потребует custom loss.
- **5c.** **Бинарная задача**: объединить tinkering+works_oob в "works" → borked vs works. F1 должен сильно вырасти. Потом внутри "works" — второй классификатор tinkering/works_oob.

**Ожидание**: 5a — потенциально +2-5% F1 если шума много. 5c — упрощение задачи, может дать insight.

**Приоритет**: 🟡 Средний — требует эксперимента, чтобы оценить масштаб проблемы.

### 6. Улучшение эмбеддингов

**Проблема**: `game_emb_0` — топ-1 фича для borked (SHAP 0.55), но эмбеддинги 16-мерные для 30K игр. Возможно недостаточно ёмкости.

**Эксперименты**:
- **6a.** **Увеличить размерность**: 32 или 48 компонент (сейчас auto=16, 91% variance). Проверить explained variance при 32 — если 96%+, не стоит.
- **6b.** **Раздельные game embeddings для CPU и GPU** — сейчас game_emb из GPU SVD только. CPU-game SVD может нести другой сигнал (CPU bottleneck vs GPU).
- **6c.** **Temporal SVD**: строить матрицу только на последних 2 года отчётов. Старые ко-встречаемости (2018) не релевантны для 2026.

**Ожидание**: 6a — маловероятно (+1% variance), 6b — может дать новый сигнал, 6c — убирает temporal noise.

**Приоритет**: 🟢 Низкий — эмбеддинги уже работают.

### 7. Deck-specific модель или фичи

**Проблема**: Steam Deck = 14% отчётов, отдельная экосистема. `deck_status` из Steam API пока не наполнен.

**Эксперименты**:
- **7a.** Завершить Steam enrichment, наполнить `deck_status` (Verified/Playable/Unsupported/Unknown)
- **7b.** Обучить **отдельную модель для Deck** на 49K отчётов. Deck-отчёты имеют стандартное железо → меньше вариативности → проще задача.
- **7c.** Добавить **interaction features**: `is_steam_deck × engine`, `is_steam_deck × anticheat`. Anti-cheat на Deck — почти гарантированный borked.

**Ожидание**: 7a+7c — быстрый эксперимент, 7b — если Deck-метрики значительно хуже общих.

**Приоритет**: 🟡 Средний — зависит от enrichment pipeline.

### 8. GitHub Issues как сигнал регрессий

**Проблема**: `github_issue_count` появилась в фичах (из enrichment), но пока object dtype и не используется моделью эффективно.

**Эксперименты**:
- **8a.** Привести `github_*` к числовым типам в train.py (сейчас object → NaN при coerce)
- **8b.** Добавить `github_has_regression` как бинарную фичу — если есть regression issue, вероятность borked выше
- **8c.** `github_open_ratio = open / (open + closed)` — доля нерешённых проблем

**Ожидание**: малый эффект (покрытие низкое), но github_has_regression может быть сильным сигналом для конкретных игр.

**Приоритет**: 🟢 Низкий — enrichment pipeline зависимость.

### 9. Interaction features на основе SHAP interaction analysis

**Проблема**: топ-взаимодействия (total_reports × game_emb_0, is_ge_proton × report_age_days, variant × game_emb_0) — сильные, но LightGBM ловит их неявно через splits. Explicit interaction features могут помочь.

**Эксперименты**:
- **9a.** `log_total_reports × game_emb_0` — явная фича. Нормализует влияние редких игр на эмбеддинг. Если у игры 3 отчёта — эмбеддинг ненадёжен, при 500 — надёжен.
- **9b.** `has_ge_proton × proton_major` — GE-Proton эффект зависит от мажорной версии (GE-Proton7 vs GE-Proton9 — разный уровень патчей).
- **9c.** `is_steam_deck × pct_stability_faults` — стабильность на Deck vs desktop — разные причины.

**Ожидание**: +0.5-1% F1 — LightGBM уже ловит часть, но explicit features ускорят обучение и могут поймать то, что splits не нашли.

**Приоритет**: 🟢 Низкий — дерево и так моделирует, но быстрый эксперимент.

### 10. Confidence-aware prediction: "uncertain" как 4-й класс

**Проблема**: модель вынуждена выбирать между tinkering и works_oob даже когда entropy > 0.9. Confidence map показал что ~30% тестовых примеров в зоне высокой неопределённости.

**Эксперименты**:
- **10a.** На inference возвращать `"uncertain"` если max_probability < threshold (0.5 или 0.6). Для API: "мы не уверены, вот распределение вероятностей". Не меняет модель, только post-processing.
- **10b.** **Abstention learning**: добавить penalty за неуверенные предсказания. Модель учится говорить "не знаю" вместо random guess.
- **10c.** Измерить: какая доля тестовых примеров попадает в P(max) < 0.5? Если >30% — модель честнее с uncertainty output.

**Ожидание**: не улучшит F1, но значительно улучшит user experience: "70% works, 25% tinkering, 5% borked" полезнее чем "works" с 56% error rate.

**Приоритет**: 🟡 Средний — для production API очень важно, для метрик нет.

### 11. Temporal SVD: эмбеддинги на свежих данных

**Проблема**: SVD строится на ВСЕХ отчётах (2018-2026). Ко-встречаемости GPU×Game из 2018 нерелевантны: старые GPU, старые версии Proton, другие паттерны совместимости.

**Эксперименты**:
- **11a.** Строить SVD только на отчётах за последние 2 года. Матрица будет меньше, но актуальнее.
- **11b.** **Weighted SVD**: взвешивать ко-встречаемости по recency (экспоненциальный decay, half-life = 1 год). Недавние отчёты вносят больший вклад.
- **11c.** Сравнить explained variance и SHAP importance: если temporal SVD даёт те же 91% variance на меньших данных — старые данные были шумом.

**Ожидание**: может улучшить game_emb качество, особенно для игр с долгой историей (CS2, GTA V) где старые отчёты про другой Proton.

**Приоритет**: 🟡 Средний — потенциально убирает temporal noise из эмбеддингов.

### 12. Двухступенчатый классификатор (hierarchical)

**Проблема**: error clusters показали — ошибки borked↔tinkering и tinkering↔works_oob живут в разных зонах feature space. Одна модель пытается провести обе границы одновременно.

**Эксперименты**:
- **12a.** **Stage 1**: borked vs not_borked (бинарная). Высокий recall borked — главная цель.
- **12b.** **Stage 2**: tinkering vs works_oob (бинарная, только на not_borked). Отдельная модель с другими весами и, возможно, другими фичами.
- **12c.** Сравнить: F1 macro каскада vs single model. Если borked recall вырастет до 0.60+ без потери F1 — это win.

**Ожидание**: borked recall может вырасти с 0.38 до 0.55+, потому что Stage 1 фокусируется только на одной границе. Общий F1 macro может вырасти на 2-4%.

**Приоритет**: 🔴 Высокий — архитектурное изменение, потенциально самый большой выигрыш.

### 13. Embedding reliability: маска для ненадёжных эмбеддингов

**Проблема**: SHAP interaction `total_reports × game_emb_0 = 0.025` (топ-1 interaction) показывает что для игр с малым числом отчётов эмбеддинг — шум. Модель не знает, что game_emb для игры с 2 отчётами ненадёжен.

**Эксперименты**:
- **13a.** Добавить `game_emb_confidence = min(total_reports_in_svd / 10, 1.0)` — степень надёжности эмбеддинга. Для игры с 2 отчётами = 0.2, с 50 = 1.0.
- **13b.** **Зануление**: game_emb = NaN если game имеет < 5 отчётов в SVD матрице. Пусть модель явно видит "нет эмбеддинга" vs "есть".
- **13c.** Аналогично для gpu_emb и cpu_emb — если семейство редкое (< 100 отчётов), эмбеддинг ненадёжен.

**Ожидание**: уменьшит ошибки на long tail (редкие игры). +0.5-1% F1 macro.

**Приоритет**: 🟡 Средний — простой эксперимент, прямо следует из interaction analysis.

---

## Рекомендуемый порядок

### Этап A: Честный baseline (убрать bias и leakage)
1. **Шаг 1** → report_age_days пересмотр (1a-1d) — убрать temporal bias
2. **Шаг 2** → data leakage fix (4a) — агрегации только по train period
3. **Шаг 3** → embedding reliability mask (13a-13b) — явная маркировка ненадёжных эмбеддингов

*Ожидание: метрики на тесте могут упасть (мы убрали "подсказки"), но модель станет честной. Это новый настоящий baseline.*

### Этап B: Архитектурные улучшения
4. **Шаг 4** → двухступенчатый классификатор (12a-12c) — borked vs rest, затем tinkering vs oob
5. **Шаг 5** → class_weight тюнинг (2a-2c) — отдельно для каждого stage
6. **Шаг 6** → cleanlab / label noise (5a) — оценить масштаб шума, убрать или перевзвесить

### Этап C: Дополнительные фичи
7. **Шаг 7** → temporal SVD (11a-11b) — свежие эмбеддинги
8. **Шаг 8** → interaction features (9a-9c) — из SHAP interaction analysis
9. **Шаг 9** → github features (8a-8c) + deck features (7a-7c)

### Этап D: Production readiness
10. **Шаг 10** → post-hoc calibration (3a-3c) — достоверные вероятности
11. **Шаг 11** → confidence-aware output (10a-10c) — "uncertain" при низкой confidence
12. **Шаг 12** → soft labels (5b) — если cleanlab показал >15% label noise

## Целевые метрики

| Метрика | Текущее | После этапа A | После этапа B | Цель (D) |
|---------|---------|---------------|---------------|----------|
| F1 macro | 0.584 | ~0.55 (честный) | 0.62+ | 0.65+ |
| borked recall | 0.38 | ~0.35 | 0.55+ | 0.60+ |
| works_oob precision | 0.44 | ~0.40 | 0.55+ | 0.60+ |
| works_oob error rate | 56% | ~50% | < 35% | < 30% |
| Calibration error (ECE) | ~0.15 | ~0.15 | ~0.10 | < 0.05 |
| works_oob recall (newest bucket) | 0.45 | ~0.50 (stable) | 0.55+ | 0.60+ |

## Визуализации (data/plots/)

| Файл | Содержание |
|------|-----------|
| `1_confusion_matrix.png` | Confusion matrix (absolute + normalized) |
| `2_probability_distributions.png` | P(class) distributions per true class |
| `3_calibration_curves.png` | Reliability diagrams per class |
| `4_shap_dependency.png` | SHAP dependency — top 9 numeric features |
| `5_shap_per_class.png` | Feature importance by class (SHAP bars) |
| `6_error_analysis.png` | Confidence, error rates, misclassification flows |
| `7_feature_space_projections.png` | t-SNE + UMAP: true labels, predicted, errors |
| `8_feature_correlations.png` | Correlation heatmap top 40 features |
| `9_shap_interactions.png` | Top 20 feature interactions (SHAP) |
| `10_shap_interaction_scatter.png` | Top 3 interaction scatter plots |
| `11_error_clusters.png` | Misclassification types in t-SNE/UMAP |
| `12_feature_distributions_by_class.png` | Feature value histograms per class |
| `13_confidence_map.png` | UMAP: prediction confidence + entropy |
| `14_temporal_analysis.png` | Accuracy/recall/confidence by time bucket |
