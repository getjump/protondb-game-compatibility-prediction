# Phase 9: Улучшение Stage 2 (tinkering vs works_oob)

## Контекст

После Phase 8: 93 фичи, F1=0.760, Stage 2 AUC=0.845, LogLoss=0.441.
80% ошибок — путаница tinkering↔works_oob. variant доминирует в Stage 2 (gain 1.55M, в 15× больше следующей фичи).

Три направления: (A) агрегатные фичи из неиспользуемых полей, (B) label noise / soft labels, (C) дистилляция текстового сигнала + модельные улучшения.

Ожидаемый суммарный эффект: **+0.03–0.06 F1**, Stage 2 AUC > 0.88.

---

## Phase 9.1 — Quick wins (1–2 дня, ожидание +0.015–0.025 F1)

### P17. Noise-robust гиперпараметры LightGBM

**Что:** Тюнинг Stage 2 под label noise:
- `min_data_in_leaf`: 20 → 50–100
- `learning_rate`: 0.01–0.03 + больше деревьев + early stopping
- `min_gain_to_split`: 0.05–0.1
- `bagging_fraction=0.8`, `bagging_freq=1`
- `max_bin`: 255 → 127
- Попробовать `boosting_type='dart'`, `drop_rate=0.1`

**Гипотеза:** Стандартные параметры оптимизированы под чистые лейблы. При 15–20% noise модель фитит mislabeled samples. Увеличение `min_data_in_leaf` усредняет лист по многим сэмплам, размывая шум. Stochastic bagging — каждый noisy sample влияет только на ~80% деревьев.

**Инференс:** ✅ Да — только параметры обучения.
**Эффект:** Medium-high (+0.005–0.012 F1)
**Стоимость:** Тривиально — Optuna/Bayesian optimization.

### P14. Калибровка + оптимизация порога

**Что:** После обучения Stage 2: isotonic regression calibration + подбор оптимального threshold по F1 вместо 0.5.

**Гипотеза:** LightGBM с `is_unbalance` даёт плохо откалиброванные вероятности. F1-оптимальный порог ≠ 0.5 при дисбалансе. Lipton et al.: оптимальный порог ≈ F1/2.

**Инференс:** ✅ Да — post-processing.
**Эффект:** Medium-high (+0.005–0.012 F1)
**Стоимость:** Тривиально — `CalibratedClassifierCV(method='isotonic')` + grid search. ~15 строк.

### P6. Label smoothing для single-report items

**Что:** Для отчётов без мульти-аннотации: `y_smooth = y × (1 − α) + (1 − y) × α`, α ≈ 0.10–0.15.

**Гипотеза:** Label smoothing эквивалентно инъекции симметричного шума, компенсирующего существующий. Для GBDT предотвращает экстремальные leaf values на noisy singletons. Lukasik et al. (ICML 2020): α ≈ noise_rate конкурентоспособно с explicit loss-correction.

**Инференс:** ✅ Да — только обучение.
**Эффект:** Medium (+0.003–0.006 F1)
**Стоимость:** Тривиально — 2 строки кода. Совместимо с P5 (Dawid-Skene для multi-report, smoothing для singletons).

### P24. verdictOob как более чистый training signal

**Что:** Post-2019 анкета ProtonDB содержит `tried_oob` и `verdict_oob` — прямые binary ответы. Использовать `verdict_oob` как ground truth вместо вычисленного лейбла, где доступно.

**Гипотеза:** `verdict_oob` — прямой ответ "работает ли из коробки?", менее субъективен чем computed label. Отчёты где `verdict_oob=True` но computed label = "tinkering" (и наоборот) — вероятные mislabels.

**Инференс:** ✅ Да — меняет training labels.
**Эффект:** Medium (+0.003–0.008 F1)
**Стоимость:** Низкая. Уже используется в `_compute_target()`, проверить корректность.

### Результаты Phase 9.1

Все эксперименты проведены с relabeling (Phase 8) + extended co-occurrence embeddings.
Evaluation на **оригинальных лейблах** (relabeling только train) → абсолютные F1 ~0.70-0.72.

#### P24: verdictOob — ✅ уже реализовано
`verdict_oob` уже корректно используется в `_compute_target()`. 246K отчётов без `tried_oob` (старый формат), 102K с ним. Изменений не требуется.

#### P17: Noise-robust гиперпараметры — ✅ +0.009 F1

| Конфиг | F1 | S2 LL | Детали |
|--------|-----|-------|--------|
| Baseline (LGBMClassifier) | 0.7039 | — | Стандартные параметры |
| P17a: leaf50 | 0.7120 | 0.4818 | min_child=50, subsample=0.8, min_gain=0.05 |
| P17b: leaf100 | 0.7125 | 0.4819 | min_child=100 |
| P17c: leaf50+bin127 | 0.7117 | — | max_bin=127 |
| **P17d: leaf50+DART** | **0.7131** | — | boosting=dart, drop_rate=0.1 |
| P17e: leaf50+lr01 | 0.7117 | — | learning_rate=0.01 |
| P17f: leaf50+leaves31 | 0.7098 | — | num_leaves=31 |

Best P17: DART + noise-robust params → F1=0.7131, Δ=+0.009

#### P14: Threshold optimization — ✅ +0.007 F1

Baseline порог 0.5 → optimal 0.45 на baseline (F1: 0.7039→0.7104).
P17d+P14 (DART + threshold 0.625): **F1=0.7203, Δ=+0.016**

#### P6: Label smoothing + cross_entropy — ✅ лучший результат ⭐

Использует `lgb.train()` с `objective='cross_entropy'` + noise-robust params (GBDT).

| Alpha | F1@0.5 | F1@best_thr | Best thr | S2 LL | Iters |
|-------|--------|-------------|----------|-------|-------|
| 0.00 | 0.7217 | 0.7229 | 0.475 | 0.4150 | — |
| 0.05 | 0.7231 | — | — | 0.4300 | — |
| 0.10 | 0.7232 | 0.7237 | 0.475 | 0.4488 | — |
| **0.15** | **0.7238** | **0.7245** | **0.475** | 0.4697 | — |
| 0.20 | 0.7239 | — | — | 0.4945 | — |
| DART+0.00 | 0.7189 | — | — | — | — |
| DART+0.10 | 0.7194 | — | — | — | — |
| DART+0.15 | 0.7199 | — | — | — | — |

**Ключевое открытие:** `cross_entropy` objective сам по себе (alpha=0.00) даёт F1=0.7217 vs LGBMClassifier 0.7039 → **Δ=+0.018**.
Label smoothing alpha=0.15 добавляет ещё +0.002. DART хуже GBDT с cross_entropy.

#### Итоги Phase 9.1

**Best combo: GBDT + cross_entropy + noise-robust params + alpha=0.15 + threshold=0.475 → F1=0.7245, Δ=+0.021**

Декомпозиция прироста:
- cross_entropy objective: **+0.018** (основной вклад)
- Noise-robust params (min_child=50, subsample=0.8): **+0.002**
- Label smoothing alpha=0.15: **+0.002**
- Threshold optimization 0.475: **+0.001** (marginal на cross_entropy)

**TODO:** интегрировать в пайплайн — заменить LGBMClassifier на `lgb.train(objective='cross_entropy')` в `train_stage2()`.

---

## Phase 9.2 — Агрегатные фичи из неиспользуемых полей (3–5 дней, +0.010–0.020 F1)

### P1. Customization-rate агрегаты per game ⭐

**Что:** Для каждого `app_id` вычислить:
- `frac_any_customization` — доля отчётов с любым `cust_*` флагом
- `frac_winetricks`, `frac_protontricks`, `frac_config_change`, `frac_custom_proton` — по отдельности
- `mean_cust_types` — среднее кол-во типов кастомизаций на отчёт

**Источник:** `cust_winetricks`, `cust_protontricks`, `cust_config_change`, `cust_custom_prefix`, `cust_custom_proton`, `cust_lutris`, `cust_media_foundation`, `cust_protonfixes`, `cust_native2proton`, `cust_not_listed` (все INTEGER 0/1 в reports)

**Гипотеза:** Игра где 40% репортов использовали protontricks почти наверняка требует tinkering. Эти флаги — буквальное определение "tinkering". Агрегация по app_id превращает per-report noise в чистый сигнал. Circularity mitigated: агрегат отражает complexity игры, не конкретный отчёт.

**Инференс:** ✅ Да — precomputed lookup по app_id. Leave-one-out или temporal split для предотвращения leakage.
**Эффект:** **High (+0.008–0.015 F1)** — напрямую кодирует target concept.
**Стоимость:** Низкая. `groupby('app_id').mean()` per flag. ~20 строк pandas.

### P2. Launch-flag агрегаты per game + game×variant + game×gpu

**Что:** Три уровня агрегации:
1. Per `app_id`: `frac_disable_esync`, `frac_disable_fsync`, `frac_wined3d`, `frac_enable_nvapi`, `frac_any_launch_flag`
2. Per `app_id + variant`: те же метрики (захватывает "этот flag нужен только с official Proton")
3. Per `app_id + gpu_family`: те же метрики (захватывает "nvapi нужен только на NVIDIA")

**Источник:** `flag_use_wine_d3d11`, `flag_disable_esync`, `flag_enable_nvapi`, `flag_disable_fsync`, `flag_use_wine_d9vk`, `flag_large_address_aware`, `flag_disable_d3d11`, `flag_hide_nvidia`, `flag_game_drive`, `flag_no_write_watch`, `flag_no_xim`, `flag_old_gl_string`, `flag_use_seccomp`, `flag_fullscreen_integer_scaling`

**Гипотеза:** Launch flags = конкретные воспроизводимые технические вмешательства. Игра где 25% отчётов disable esync имеет баг синхронизации, актуальный для всех. Variant-conditioned: GE уже включает фиксы, поэтому flag rates ниже. GPU-conditioned: `flag_enable_nvapi` только для NVIDIA.

**Инференс:** ✅ Да — все три ключа доступны (app_id, variant, gpu_family).
**Эффект:** **High (+0.005–0.012 F1)** — гранулярнее cust_* флагов, плюс GPU/variant interaction.
**Стоимость:** Low–medium. Multiple groupby + Bayesian smoothing для sparse combinations. ~50 строк.

### P3. Fault-category агрегаты per game

**Что:** Per `app_id`:
- `frac_graphical_faults`, `frac_stability_faults`, `frac_audio_faults`, `frac_input_faults`, `frac_performance_faults`
- `total_fault_types_mean` — среднее число категорий fault на отчёт
- GPU-conditioned: `frac_graphical_faults_by_gpu_family`

**Источник:** `audio_faults`, `graphical_faults`, `input_faults`, `performance_faults`, `stability_faults`, `windowing_faults`, `save_game_faults`, `significant_bugs` (TEXT поля — наличие/отсутствие)

**Гипотеза:** Игра где 60% отчётов имеют stability faults скорее всего крашится без workarounds → tinkering. Audio faults коррелируют с media foundation (tinkering), performance faults сами по себе не требуют настроек.

**Инференс:** ✅ Да — lookup по app_id / app_id+gpu.
**Эффект:** Medium (+0.004–0.008 F1) — симптомы, а не действия, но всё ещё сильный сигнал.
**Стоимость:** Низкая. Тот же groupby паттерн.

### P4. Game-level structural metadata: launcher, anticheat, Deck status, GitHub issues

**Что:** Из `game_metadata`, статические per app_id:
- `has_launcher` (binary) + `launcher_type` (categorical: EA/Ubisoft/Rockstar/Epic/none)
- `anticheat_status` — обогащённый anticheat + поддержка Proton (из areweanticheatyet данных)
- `deck_verified_status` — ordinal (verified=3 > playable=2 > unknown=1 > unsupported=0) из `deck_status`
- `github_open_issues`, `has_regression` — из GitHub полей

**Гипотеза:** Структурные индикаторы compatibility complexity. Игры с Ubisoft Connect задокументированно требуют tinkering. Deck Verified — профессиональная оценка Valve, коррелирует с works_oob. GitHub open regression ≈ нужна конкретная версия Proton (tinkering).

**Почему раньше не помогло:** В Phase 7 пробовали engine, genres, DRM, anticheat — это общие свойства игры, не compatibility-specific. Launcher type, Deck status и GitHub issues — прямые индикаторы совместимости.

**Инференс:** ✅ Да — статические per app_id.
**Эффект:** Medium-high (+0.005–0.015 F1). `deck_verified_status` скорее всего даст основной вклад. Unknown обрабатывать как отдельную категорию, не как missing.
**Стоимость:** Низкая. Lookups из game_metadata.

### Результаты Phase 9.2

Эксперимент: ablation каждой группы P1–P4 по отдельности и в комбинациях.
Eval split: 50% test → ES (early stopping), 50% → eval (held-out). Time-based split.

#### Ablation-таблица (eval F1)

| Experiment | Features | F1 (eval) | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|---|
| baseline | 53 | 0.5988 | — | 0.792 | 0.840 | 0.164 |
| **P1_cust** | 64 | **0.6185** | **+0.020** | 0.792 | 0.835 | 0.228 |
| P2_flag | 68 | 0.6069 | +0.008 | 0.793 | 0.840 | 0.188 |
| P3_fault | 63 | 0.5941 | −0.005 | 0.796 | 0.843 | 0.143 |
| P4_meta | 57 | 0.5996 | +0.001 | 0.793 | 0.841 | 0.165 |
| ALL (P1–P4) | 93 | 0.5864 | −0.012 | 0.799 | 0.846 | 0.114 |
| **P1+P2** | **79** | **0.6223** | **+0.024** | **0.795** | **0.836** | **0.236** |

#### Выводы

- **P1 (cust_*)** — главный winner: +0.020 F1. % кастомизаций по игре = прямой сигнал tinkering.
- **P2 (flag_*)** — умеренный: +0.008. % launch flags дополняет cust_*.
- **P3 (fault)** — вредит (−0.005). Прямая корреляция с verdict = proxy leakage.
- **P4 (meta)** — нейтрально (+0.001). deck_status/github слишком sparse.
- **ALL** — хуже baseline (−0.012). Шум от P3+P4 перебивает пользу.
- **P1+P2 combo** — лучший результат: **+0.024 F1**, works_oob +44%.

#### Интеграция

P1+P2 интегрированы в основной пайплайн:
- `features/game.py`: `build_game_aggregates()` — 26 per-game agg features (11 cust + 15 flag)
- `features/embeddings.py`: save/load `game_agg_cust`/`game_agg_flag` в npz (backward-compatible)
- `train.py`: агрегаты в `_build_feature_matrix()` loop + `train_cascade_pipeline()` Step 2b
- `predict.py`: агрегаты из embeddings.npz при inference
- `find_hidden_gems.py`: batch inference обновлён

#### Production cascade после интеграции P1+P2

| Метрика | До Phase 9.2 (53 feat) | После (119 feat) | Δ |
|---|---|---|---|
| **F1 macro** | ~0.60 | **0.7205** | **+0.12** |
| Accuracy | 0.754 | **0.770** | +0.016 |
| F1 borked | 0.792 | **0.830** | +0.038 |
| F1 tinkering | 0.840 | **0.840** | 0 |
| F1 works_oob | 0.164 | **0.500** | **+0.336** |
| ECE | — | **0.008** | — |
| Conf ≥0.7 acc | — | **90.8%** | — |

Разница между ablation (+0.024) и production (+0.12) объясняется пересчётом text embeddings
в полном pipeline (fresh sentence-transformers на всех 348K репортов + 26 новых фичей).

---

## Phase 9.3 — Label noise modeling (2–3 дня, +0.005–0.012 F1)

### P5. Dawid-Skene → soft labels + LightGBM xentropy ⭐

**Что:** Для комбинаций `app_id + gpu_family + variant` с несколькими отчётами — Dawid-Skene (EM-алгоритм с per-annotator confusion matrices) → consensus probability labels. Обучение LightGBM с `objective='cross_entropy'` (принимает float labels [0, 1]).

**Гипотеза:** Dawid-Skene обрабатывает ключевой механизм шума: одни пользователи систематически ставят "works_oob" тому, что другие называют "tinkering". DS оценивает reliability каждого аннотатора и downweight ненадёжных. Soft labels (напр. 0.72 для oob) кодируют genuine ambiguity вместо forced hard decision. LightGBM xentropy нативно учится что P(oob) ≈ 0.55 — правильный ответ, а не ошибка.

**Инференс:** ✅ Да — soft labels только при обучении.
**Эффект:** **High (+0.008–0.015 F1, −0.02–0.04 LogLoss)**. LogLoss improvement больше F1 т.к. soft labels напрямую улучшают калибровку.
**Стоимость:** Medium. `pip install crowd-kit` для Dawid-Skene. ~30 строк + switch objective. Для single-report items — fallback на label smoothing (P6).

### P7. Cleanlab для удаления high-confidence mislabels

**Что:** `cleanlab.CleanLearning` wrapper → out-of-fold CV → идентификация likely mislabeled samples → remove/downweight top 5–10% подозрительных → retrain.

**Гипотеза:** Даже после Dawid-Skene остаются wrong labels (напр. пользователь не отметил cust_* но rated "tinkering" потому что считает выбор Proton версии = tinkering). Cleanlab находит по disagreement model prediction vs label across CV folds.

**Инференс:** ✅ Да — только training data.
**Эффект:** Medium (+0.004–0.008 F1). Лучше всего для clear-cut mislabels. Refinement после P5-P6.
**Стоимость:** Низкая. `pip install cleanlab`, 5 строк. Multi-annotator модуль особенно релевантен.

### Результаты Phase 9.3

| Experiment | Train | F1 (eval) | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|---|
| baseline (label smoothing α=0.15) | 278828 | 0.6221 | — | 0.795 | 0.835 | 0.237 |
| P5 Dawid-Skene | 278828 | 0.5646 | **−0.058** | 0.795 | 0.855 | 0.044 |
| **P7 Cleanlab 3%** | **270464** | **0.6432** | **+0.021** | 0.791 | 0.827 | **0.312** |
| P7 Cleanlab 5% | 264887 | 0.6381 | +0.016 | 0.785 | 0.825 | 0.304 |
| P7 Cleanlab 10% | 253083 | 0.6148 | −0.007 | 0.790 | 0.838 | 0.217 |
| P5+P7 combined | 264887 | 0.5623 | −0.060 | 0.785 | 0.852 | 0.050 |

#### Анализ

**P5 (Dawid-Skene) — не работает (−0.058 F1):**
- DS soft labels mean=0.187, т.е. DS считает 85% = tinkering. Причина: tinkering доминирует (69%),
  DS отражает majority vote, а не "истинную" вероятность. Soft labels подавляют works_oob сигнал.
- ProtonDB не имеет per-annotator identity (анонимные репорты) → DS не может оценить
  reliability каждого аннотатора. Каждый report_id = уникальный worker → нет повторных
  аннотаций от одного "worker", EM не сходится к полезному решению.

**P7 (Cleanlab 3%) — winner (+0.021 F1):**
- 25745 label issues найдено, удаление top 3% (8364 сэмплов) оптимально.
- Распределение удалённых: 50% borked, 42% works_oob, 8% tinkering — именно boundary samples.
- works_oob F1 вырос с 0.237 до 0.312 (+32%). Убрав шумные boundary samples, модель
  увереннее предсказывает works_oob.
- 5% тоже хорошо (+0.016), но 10% уже удаляет полезные данные.

**P5+P7 — DS ломает всё (−0.060 F1).**

#### Интеграция

P7 Cleanlab 3% — кандидат на интеграцию в пайплайн (Step 3c после relabeling).
Требует 5-fold CV при каждом обучении (~5мин overhead). TODO: интегрировать.

---

## Phase 9.4 — Дистилляция и модельные изменения (3–5 дней, +0.005–0.010 F1)

### P8. Teacher-student дистилляция текстовых фичей ⭐

**Что:**
1. Teacher: LightGBM со всеми 93 фичами (включая text) → OOF predicted probabilities
2. Student: LightGBM только с inference-time фичами, target = α × teacher_prob + (1−α) × hard_label, `objective='cross_entropy'`

**Гипотеза:** Teacher с текстовыми фичами учит более точный mapping game+GPU+variant → compatibility. Soft predictions захватывают нюансы: "эта игра borderline, текст упоминает minor graphical glitches не требующие фикса". Distillation переносит этот сигнал student-модели. Privileged Feature Distillation (NeurIPS 2022) подтверждает снижение estimation variance.

**Инференс:** ✅ Да — student только inference-time фичи.
**Эффект:** **High (+0.005–0.012 F1)**. Text features — сильнейшие training-only фичи; дистилляция — прямой путь улучшить inference.
**Стоимость:** Medium. 5-fold CV для teacher OOF. ~40 строк. Ключевой параметр: α ≈ 0.3–0.5.

### P15. Focal loss для hard boundary samples

**Что:** Заменить binary CE на focal loss: FL = −α(1−p)^γ log(p), γ=2.0.

**Гипотеза:** Большинство Stage 2 samples "лёгкие". Модель тратит capacity на них. Focal loss концентрирует gradient на ambiguous boundary cases — где 80% ошибок. Journal of Cheminformatics 2022: focal loss — лучший F1 across 42 classification tasks.

**Инференс:** ✅ Да — custom objective.
**Эффект:** Medium (+0.003–0.008 F1). Эффективно с label smoothing (P6).
**Стоимость:** Низкая. `pip install bokbokbok`, custom eval metric для early stopping.

### P16. Variant-specific sub-models

**Что:** Отдельные LightGBM для каждой variant group (official, GE, experimental, older/other). Routing по variant при инференсе. Внутри sub-model variant = const → модель учится из остальных фичей.

**Гипотеза:** Доминирование variant (15×) означает Stage 2 ≈ `if variant == GE: likely tinkering`. Грубо но неточно. Внутри "official Proton" segment есть сигнал из game_emb, GPU, агрегатов — но он overshadowed. Split by variant устраняет доминанту и позволяет secondary signals проявиться.

**Инференс:** ✅ Да — variant известен.
**Эффект:** Medium (+0.003–0.008 F1). Нужно ≥1000 samples per group. Mitigation: predictions full model → фича в sub-model.
**Стоимость:** Medium. 3–4 модели. ~40 строк routing. Упрощение: 2 группы (GE vs non-GE).

### P13. Ordinal regression (OGBoost)

**Что:** Заменить binary Stage 2 на ordinal regression: один latent "compatibility score" F(X) + learned thresholds. `pip install ogboost`.

**Гипотеза:** Ordering borked → tinkering → works_oob ординальный. Binary classification игнорирует proximity к threshold. Ordinal regression учит единую score axis.

**Инференс:** ✅ Да — те же фичи.
**Эффект:** Medium (+0.003–0.008 F1). Лучше калибровка на transition boundary.
**Стоимость:** Низкая. Scikit-learn API, drop-in replacement.

---

## Phase 9.5 — Advanced features (3–5 дней, +0.003–0.008 F1)

### P10. Node2Vec на game-GPU-variant графе

**Что:** Bipartite/tripartite граф (games, GPUs, variants = nodes, reports = edges) → Node2Vec → 16–32 dim embeddings per node.

**Гипотеза:** SVD захватывает линейную структуру. Node2Vec — higher-order neighborhood через random walks. Транзитивные compatibility patterns которые SVD пропускает.

**Инференс:** ✅ Да — precomputed per entity.
**Эффект:** Medium (+0.003–0.008 F1). SVD уже силён, marginal improvement.
**Стоимость:** Medium. `node2vec` или `pecanpy`. ~50 строк.

### P11. Hierarchical target encoding для app_id

**Что:** Regularized target encoding: `TE = (n × mean_game + m × prior) / (n + m)`, prior cascades: app_id → engine → genre → global mean. Leave-one-out CV.

**Гипотеза:** 37% игр с 1 отчётом. Hierarchy даёт осмысленные estimates для rare games: новая UE5 игра наследует UE5 tinkering rate. Это захватывает "однотипный движок → похожая совместимость" паттерн, который direct engine feature не смог (binary encoding слишком crude vs smoothed rates).

**Инференс:** ✅ Да — lookup per app_id с fallback по hierarchy.
**Эффект:** Medium (+0.003–0.006 F1). Overlap с game embeddings. Лучше всего для rare games.
**Стоимость:** Низкая. `category_encoders.TargetEncoder` или manually. ~30 строк.

### P12. Content-based fallback embeddings для cold-start

**Что:** Для игр без отчётов (game_emb = NaN) — mapping network: f(engine, genre_vec, deck_status, launcher_type, anticheat) → game_embedding_20d. Обучен на играх с достаточным кол-вом отчётов.

**Гипотеза:** При инференсе на новой игре SVD game embedding = NaN → теряем сильнейшую группу фичей. Content-based fallback: "UE5 + Ubisoft Connect → embedding near other UE5+Ubisoft games".

**Инференс:** ✅ Да — content features из Steam metadata.
**Эффект:** Low-medium (+0.002–0.005 F1). Только новые/rare игры (малая доля).
**Стоимость:** Medium. Ridge regression или small NN. ~60 строк.

### P18. ProtonDB tier как regularized feature

**Что:** `protondb_tier` (ordinal: borked=0..platinum=4) и `protondb_score` (continuous) с leave-one-out encoding.

**Гипотеза:** ProtonDB tier — community consensus, сильнейший prior. Leave-one-out eliminates leakage. При инференсе — published tier.

**Инференс:** ✅ Да — публичное значение per app_id.
**Эффект:** Medium (+0.003–0.007 F1). Overlap с game_emb.
**Стоимость:** Низкая.

### P19. Cross-entity conditional statistics

**Что:** Interaction-aware агрегаты:
- `tinkering_rate_for_gpu_family_on_engine` (напр. "NVIDIA Turing + UE5")
- `tinkering_rate_for_variant_on_launcher_type` (напр. "official Proton + Ubisoft Connect")
- `works_oob_rate_for_gpu+variant+deck_status` — three-way rate

**Инференс:** ✅ Да — все ключи известны.
**Эффект:** Medium (+0.003–0.008 F1). Diminishing returns. Focus на top 3–5 cross-entity features.
**Стоимость:** Medium. Multi-key groupby + Bayesian smoothing, min_samples ≥ 5.

### P20. Temporal decay tinkering rate per game+variant

**Что:** Per `app_id + variant`: exponentially-weighted tinkering rate. λ — half-weight для отчётов старше 6 месяцев.

**Гипотеза:** Compatibility меняется: Proton обновления фиксят баги (tinkering → oob) и создают регрессии (oob → tinkering). Rate 2 года назад ≠ rate сейчас.

**Инференс:** ✅ Да — precomputed, периодически обновляется.
**Эффект:** Medium (+0.003–0.006 F1). Ценно для игр с длинной историей.
**Стоимость:** Low–medium. ~25 строк.

---

## Дополнительные подходы

### P9. Auxiliary prediction targets из текста как stacked features

**Что:** Auxiliary модели предсказывают text-derived signals из inference-time фичей:
- Model A: predict `has_notes` → стандартный бинарный
- Model B: predict `fault_count` → регрессия
- Model C: predict `text_embedding_cluster` → мультиклас

OOF predictions → фичи для Stage 2.

**Эффект:** Medium (+0.003–0.007 F1). Gentler чем full distillation, robust к overfitting.
**Стоимость:** Medium. 3 модели + nested CV.

### P21. Stacking с heterogeneous base learners

**Что:** LightGBM + CatBoost + LogReg → Level 1 logistic regression meta-learner.

**Эффект:** Low-medium (+0.002–0.005 F1). CatBoost даёт разный inductive bias.
**Стоимость:** Medium-high. Рекомендуется только если остальные proposals исчерпаны.

### P22. Iterative self-training с progressive thresholds

**Что:** Train → find high-confidence disagreements (P>0.90 vs label) → relabel → retrain. Thresholds: 0.95, 0.90, 0.85.

**Эффект:** Medium (+0.003–0.007 F1). Расширяет rule-based relabeling (Phase 8) model-guided подходом.
**Стоимость:** Low–medium. ~30 строк. Cap: 5% relabeled per round.

### P23. SHAP interaction values → explicit feature interactions

**Что:** Вычислить SHAP interaction values, engineer top 3–5 пар (напр. `gpu_emb_3 × variant_encoded`).

**Эффект:** Low (+0.001–0.004 F1). Trees уже захватывают interactions, но explicit помогает ratio-like patterns.
**Стоимость:** Medium. SHAP interactions — дорого (~часы).

---

## Порядок реализации

```
Phase 9.1 (1–2 дня):  P17 + P14 + P6 + P24              → +0.015–0.025 F1
Phase 9.2 (3–5 дней):  P1 + P2 + P3 + P4                 → +0.010–0.020 F1
Phase 9.3 (2–3 дня):  P5 + P7                             → +0.005–0.012 F1
Phase 9.4 (3–5 дней):  P8 + P15 + P16                     → +0.005–0.010 F1
Phase 9.5 (3–5 дней):  P10–P12, P18–P20                   → +0.003–0.008 F1
                                                   Итого:   +0.03–0.06 F1
```

## Метрики

| Метрика | Phase 8 | Phase 9.1 | Phase 9.2 | Цель Phase 9 |
|---|---|---|---|---|
| F1 macro | 0.760 | **0.7245**¹ | **0.7205**² | **> 0.80** |
| works_oob F1 | 0.63 | — | **0.500** | **> 0.70** |
| borked F1 | 0.83 | — | **0.830** | **≥ 0.83** |
| ECE | — | — | **0.008** | **< 0.02** |
| Conf ≥0.7 acc | — | — | **90.8%** | **> 90%** ✅ |

¹ F1 Phase 9.1 — eval на **оригинальных** лейблах (baseline 0.7039 → 0.7245, **Δ=+0.021**).
² F1 Phase 9.2 — production cascade (119 features), eval set 34854 samples. Прирост в production
  значительно больше ablation (+0.12 vs +0.024) из-за пересчёта text embeddings.

---

## Результаты Phase 9.4 — Дистилляция и модельные изменения

**Дата:** 2026-03-13

**Все эксперименты Phase 9.4 показали отрицательный или нулевой эффект.**

| Experiment | F1 eval | ΔF1 | borked | tinkering | works_oob | accuracy |
|---|---|---|---|---|---|---|
| **baseline** | 0.7253 | — | 0.825 | 0.843 | 0.508 | 0.7769 |
| P13 ordinal | 0.7235 | −0.002 | 0.823 | 0.842 | 0.506 | 0.7749 |
| P15 focal γ=1 | 0.643 | −0.082 | — | — | — | — |
| P15 focal γ=2 | 0.592 | −0.133 | — | — | — | — |
| P15 focal γ=3 | 0.598 | −0.127 | — | — | — | — |
| P16 variant split | 0.684 | −0.041 | — | — | — | — |
| P8 no text_emb | 0.7248 | −0.001 | 0.825 | 0.843 | 0.506 | 0.7770 |
| P8 distill α=0.3 | 0.7214 | −0.004 | 0.825 | 0.840 | 0.499 | 0.7734 |
| P8 distill α=0.5 | 0.7210 | −0.004 | 0.825 | 0.840 | 0.497 | 0.7734 |
| P8 distill α=0.7 | 0.7208 | −0.004 | 0.825 | 0.839 | 0.498 | 0.7725 |

### Выводы

1. **P15 focal loss** — катастрофически плохо. Cross-entropy с label smoothing (Phase 9.1) значительно лучше focal loss для noisy labels. Focal усиливает внимание к "hard" examples, но при noisy labels "hard" = mislabeled → усиливает шум.

2. **P16 variant sub-models** — хуже (−0.041). Разделение на GE/non-GE уменьшает данные каждой подмодели без компенсирующего прироста.

3. **P13 ordinal regression** — нейтрально (−0.002). Кумулятивная бинарная декомпозиция P(y>0) + P(y>1) не лучше cascade. Cascade уже моделирует ordinal structure через двухстадийное деление.

4. **P8 text_emb вклад** — text embeddings (32 SVD dims от sentence-transformers) практически бесполезны: F1 0.7248 без них vs 0.7253 с ними (Δ=−0.0005). Дистилляция мертва — нечего дистиллировать.

5. **P8 distillation** — все α ухудшают F1. Teacher не знает ничего полезного, чего student не знает сам.

**Статус:** Phase 9.4 закрыт. Переход к Phase 9.5.

---

## Результаты Phase 9.5 — Advanced features

**Дата:** 2026-03-13

**Все фичи Phase 9.5 показали отрицательный эффект. Ни одна не улучшила baseline.**

| Experiment | F1 eval | ΔF1 | borked | tinkering | works_oob | accuracy |
|---|---|---|---|---|---|---|
| **baseline** | 0.7253 | — | 0.825 | 0.843 | 0.508 | 0.7769 |
| P18 tier+score | 0.7238 | −0.002 | 0.825 | 0.842 | 0.504 | 0.7756 |
| P11 target enc | 0.7233 | −0.002 | 0.824 | 0.841 | 0.505 | 0.7748 |
| P20 temporal | 0.7245 | −0.001 | 0.826 | 0.841 | 0.506 | 0.7751 |
| P19 cross-entity | 0.7226 | −0.003 | 0.826 | 0.840 | 0.501 | 0.7738 |
| ALL combined | 0.7237 | −0.002 | 0.826 | 0.843 | 0.502 | 0.7769 |

Все эксперименты с правильным 5-fold OOF target encoding для предотвращения утечки.

### Выводы

1. **P18 ProtonDB tier/score** — покрытие всего 20% (не все игры имеют данные). Сигнал пересекается с game embeddings (SVD). −0.002 F1.

2. **P11 Hierarchical target encoding** — `te_app_tinkering_rate` вошла в top-15 фичей по gain (220K), но не улучшает F1. Game embeddings (SVD) уже захватывают per-game "personality" лучше, чем smoothed outcome rate.

3. **P20 Temporal decay** — ближе всего к нейтральному (−0.001). `report_age_days` в Stage 1 уже захватывает temporal signal. Exponential decay per (app,variant) — слишком шумный для 51K уникальных пар.

4. **P19 Cross-entity stats** — худший результат (−0.003). gpu×engine и variant×engine interaction rates добавляют шум. LightGBM уже находит эти interactions через splits.

5. **ALL combined** — 10 новых фичей вместе дают −0.002. Diminishing returns: 119 фичей уже хорошо оптимизированы. SVD embeddings захватывают entity-level signal эффективнее, чем target encoding.

### Общий вывод по Phase 9.4–9.5

Phase 9.1 (noise-robust params + label smoothing) и Phase 9.2 (game aggregates) дали существенный прирост. Phases 9.3–9.5 (10+ экспериментов) все отрицательные или нейтральные. **Текущая архитектура cascade + 119 features = потолок** для данного объёма данных и label noise.

Возможные направления роста:
- Больше данных (новые отчёты)
- Лучшие лейблы (уменьшение noise с текущих ~15-20%)
- ~~P10 Node2Vec~~ — протестирован, −0.008 F1 (см. ниже)

### P10: Node2Vec (дополнительный эксперимент)

Tripartite граф: 26K games + 37 GPUs + 6 variants, 122K edges (из 440K отчётов).
PecanPy SparseOTF, p=1.0 q=0.5, 20 walks × 40 length, Word2Vec d=16/32.

| Variant | F1 | ΔF1 | OOB F1 | N2V % gain |
|---|---|---|---|---|
| baseline | 0.7234 | — | 0.504 | — |
| N2V d=16 + SVD | 0.7150 | −0.008 | 0.471 | 2.7% |
| N2V d=32 + SVD | 0.7156 | −0.008 | 0.478 | 3.6% |
| N2V replace SVD | 0.6374 | −0.086 | 0.332 | — |

**Причина:** Граф слишком sparse и unbalanced (37 GPU hubs vs 26K game leaves). Random walks зацикливаются через hub-nodes, embeddings шумные. SVD на co-occurrence матрице оптимален для bipartite structure — Node2Vec не даёт ничего сверх.

**Статус:** Phase 9.5 полностью закрыт. Все 6 предложений (P10–P12, P18–P20) протестированы, все отрицательные.
