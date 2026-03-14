# Задача: исследование новых фичей для улучшения ML-модели предсказания совместимости игр с Linux/Proton

## Что мы предсказываем

Каскадная модель из двух LightGBM для предсказания совместимости PC-игры с Linux через Proton/Wine.
Входные данные — отчёт пользователя о запуске игры. Предсказываем один из трёх классов:

- **borked** (0) — игра не работает
- **tinkering** (1) — работает, но потребовались настройки (launch options, env vars, protontricks и т.д.)
- **works_oob** (2) — работает "из коробки" без настроек

### Архитектура каскада
- **Stage 1**: borked vs works (binary). AUC=0.977, LogLoss=0.124 — почти идеальный
- **Stage 2**: tinkering vs works_oob (binary, только среди "works"). AUC=0.845, LogLoss=0.441 — bottleneck

### Контекст использования
При инференсе у нас есть: app_id игры, GPU пользователя, Proton variant. У нас **НЕТ** текстов отчётов (notes_verdict и т.д.) — они доступны только при обучении. Модель должна предсказывать для новых пользователей, у которых ещё нет отчёта.

---

## Данные

### База: SQLite, 348K отчётов, 31K игр

### Таблица `reports` (348,536 строк)
Основные поля (используются):
- `app_id` (INTEGER) — Steam App ID
- `timestamp` (TEXT) — время отчёта
- `gpu` (TEXT) — сырая строка GPU (напр. "NVIDIA GeForce RTX 3070")
- `gpu_driver` (TEXT) — сырая строка драйвера
- `variant` (TEXT) — тип Proton: official, ge, experimental, native, notListed, older
- `verdict` (TEXT) — "yes"/"no" (работает/не работает)
- `verdict_oob` (TEXT) — "yes"/"no" (работает из коробки)
- `battery_performance` (TEXT) — только Steam Deck

Текстовые поля (только для обучения):
- `concluding_notes`, `notes_verdict`, `notes_extra`, `notes_customizations`
- `notes_audio_faults`, `notes_graphical_faults`, `notes_performance_faults`
- `notes_stability_faults`, `notes_windowing_faults`, `notes_input_faults`
- `notes_significant_bugs`, `notes_save_game_faults`, `notes_concluding_notes`

Структурированные поля (не используются сейчас):
- `cust_winetricks`, `cust_protontricks`, `cust_config_change`, `cust_custom_prefix`, `cust_custom_proton`, `cust_lutris`, `cust_media_foundation`, `cust_protonfixes`, `cust_native2proton`, `cust_not_listed` — бинарные флаги кастомизаций (INTEGER 0/1)
- `flag_use_wine_d3d11`, `flag_disable_esync`, `flag_enable_nvapi`, `flag_disable_fsync`, `flag_use_wine_d9vk`, `flag_large_address_aware`, `flag_disable_d3d11`, `flag_hide_nvidia`, `flag_game_drive`, `flag_no_write_watch`, `flag_no_xim`, `flag_old_gl_string`, `flag_use_seccomp`, `flag_fullscreen_integer_scaling` — launch flags (INTEGER 0/1)
- `launch_options` (TEXT) — сырые launch options
- `proton_version`, `custom_proton_version` (TEXT)
- `cpu` (TEXT), `ram` (TEXT), `ram_mb` (INTEGER), `os` (TEXT), `kernel` (TEXT)
- `installs`, `opens`, `starts_play` (TEXT) — "yes"/"no", шаги запуска
- `duration` (TEXT) — длительность игры
- `audio_faults`, `graphical_faults`, `input_faults`, `performance_faults`, `stability_faults`, `windowing_faults`, `save_game_faults`, `significant_bugs` (TEXT) — категории проблем (не notes, а сами категории)
- `launcher`, `secondary_launcher` (TEXT) — тип лаунчера
- `is_multiplayer_important`, `online_mp_attempted`, `online_mp_played`, `online_mp_appraisal` (TEXT) — мультиплеер
- `is_impacted_by_anticheat` (TEXT) — влияние античита
- `readability`, `control_layout`, `did_change_control_layout`, `frame_rate` (TEXT) — Steam Deck-специфичные
- `followup_*_json` (TEXT) — JSON с детальными followup-данными по каждой категории проблем

### Таблица `game_metadata` (30,968 игр)
- `app_id` (INTEGER PK)
- `developer`, `publisher` (TEXT)
- `genres` (TEXT) — через запятую, 56 уникальных
- `categories` (TEXT) — через запятую, 145 уникальных (Steam categories: Single-player, Multi-player, Co-op и т.д.)
- `release_date` (TEXT)
- `has_linux_native` (INTEGER)
- `engine` (TEXT) — ~50 уникальных (Unity, Unreal Engine 4, и т.д.)
- `graphics_apis` (TEXT) — DirectX 11, Vulkan, OpenGL и т.д.
- `drm` (TEXT) — DRM-система
- `anticheat`, `anticheat_status` (TEXT)
- `deck_status` (INTEGER) — Steam Deck verified status (0/1/2/3)
- `deck_tests_json` (TEXT) — JSON с тестами Valve для Deck
- `protondb_tier` (TEXT) — агрегированный tier от ProtonDB (borked/bronze/silver/gold/platinum)
- `protondb_score` (REAL), `protondb_confidence` (TEXT), `protondb_trending` (TEXT)
- `github_issue_count`, `github_open_count`, `github_closed_completed`, `github_closed_not_planned` (INTEGER) — issues из Proton GitHub
- `github_has_regression` (INTEGER), `github_latest_issue_date` (TEXT)

### Таблица `enrichment_cache` (~92K строк)
- `app_id`, `source` (steam/pcgamingwiki/deck), `data_json` (TEXT) — сырые JSON с данных из Steam API, PCGamingWiki и Deck

### Нормализация
- `gpu_normalization_heuristic` (35K строк): raw_string → vendor, family, model, is_apu, is_igpu, is_mobile
- `cpu_normalization_heuristic` (2.4K строк): raw_string → vendor, family, model, generation
- `gpu_driver_normalization` (4.9K строк): raw_string → driver_vendor, driver_version, driver_major, driver_minor

---

## Текущие фичи (93 штуки)

### A. Hardware (7 фичей)
- `gpu_family` — категориальная, label-encoded (топ-100). Напр. "GeForce RTX 30", "Radeon RX 6000"
- `nvidia_driver_version` — числовая, только для NVIDIA (major + minor/1000)
- `mesa_driver_version` — числовая, только для Mesa/AMD/Intel
- `is_apu`, `is_igpu`, `is_mobile` — бинарные
- `is_steam_deck` — бинарный

### B. Report meta (1 фича)
- `variant` — категориальная: official, ge, experimental, native, notListed, older

### C. Text meta (5 фичей, доступны только при обучении)
- `has_concluding_notes` — бинарная
- `concluding_notes_length` — числовая
- `fault_notes_count` — кол-во непустых fault_notes полей (0-8)
- `has_customization_notes` — бинарная
- `total_notes_length` — числовая

### D. Text keywords (8 фичей, доступны только при обучении)
- `mentions_crash`, `mentions_fix`, `mentions_perfect`, `mentions_proton_version`, `mentions_env_var`, `mentions_performance` — бинарные
- `sentiment_negative_words`, `sentiment_positive_words` — счётчики

### E. GPU embeddings (20 dims)
SVD из расширенной co-occurrence матрицы (gpu_family + variant + engine + deck) × game → avg_verdict_score.
665 осей × 31K games. Левые сингулярные вектора → GPU embeddings.
`gpu_emb_0..19`

### F. Game embeddings (20 dims)
Правые сингулярные вектора той же SVD. Захватывают "compatibility profile" игры.
`game_emb_0..19` — **самая важная группа фичей** (ΔF1=−0.060 при удалении).

### G. Text embeddings (32 dims, доступны только при обучении)
sentence-transformers (all-MiniLM-L6-v2) на notes_verdict → SVD 32 dims.
`text_emb_0..31`. Объясняют 59% variance. LightGBM обрабатывает NaN нативно при инференсе.

---

## Текущие результаты

### Общие метрики (Phase 8: A2 extended co-occurrence + Strict relabeling)
```
              precision    recall  f1-score   support
      borked       0.85      0.82      0.83      9849
   tinkering       0.85      0.79      0.82     42713
   works_oob       0.57      0.69      0.63     17146

F1 macro:  0.760
Accuracy:  0.769
```

### Per-class AUC и Brier
```
borked:     AUC=0.977  Brier=0.035
tinkering:  AUC=0.878  Brier=0.143
works_oob:  AUC=0.858  Brier=0.130
```

### Stage-level
```
Stage 1 (borked vs works):       AUC=0.977, LogLoss=0.124 — решённая задача
Stage 2 (tinkering vs works_oob): AUC=0.845, LogLoss=0.441 — bottleneck
```

### Error cascade (16,135 ошибок, 23.1%)
```
tinkering→works_oob:   8171 (50.6%) — главная проблема
works_oob→tinkering:   4754 (29.5%)
borked→tinkering:      1023 ( 6.3%)
tinkering→borked:       953 ( 5.9%)
borked→works_oob:       717 ( 4.4%)
works_oob→borked:       517 ( 3.2%)
```

80% ошибок — путаница между tinkering и works_oob в Stage 2.

### Feature importance

**Stage 1 (borked vs works) — топ-5:**
```
text_emb_2:             1,871K gain
text_emb_0:               963K
mentions_perfect:         451K
fault_notes_count:        447K
text_emb_4:               260K
```

**Stage 2 (tinkering vs works_oob) — топ-10:**
```
variant:              1,553K gain  (в 15× больше следующей!)
concluding_notes_len:   110K
nvidia_driver_version:   99K
mesa_driver_version:     89K
game_emb_0:              79K
total_notes_length:      67K
game_emb_3:              66K
game_emb_1:              65K
has_customization_notes: 64K
has_concluding_notes:    50K
```

### Confidence deployment
```
conf ≥ 0.7: coverage=63%, accuracy=89%, F1=0.858
conf ≥ 0.8: coverage=48%, accuracy=94%, F1=0.895
conf ≥ 0.9: coverage=33%, accuracy=98%, F1=0.935
```

### Per-variant performance
```
official:      acc=0.880, F1=0.675 (78.8% tinkering, 9.0% oob)
experimental:  acc=0.663, F1=0.662 (35.7% tinkering, 42.2% oob)
ge:            acc=0.631, F1=0.662 (43.3% tinkering, 40.4% oob)
native:        acc=0.788, F1=0.655 (74.5% tinkering, 16.2% oob)
notListed:     acc=0.622, F1=0.637 (41.5% tinkering, 46.7% oob)
older:         acc=0.669, F1=0.644 (35.4% tinkering, 51.2% oob)
```

---

## Что уже пробовали и не помогло
- **Game metadata как прямые фичи** (engine, genres, categories, graphics_api, drm, anticheat, release_year, has_linux_native, is_multiplayer): ablation Phase 7 показал ΔF1 < 0.002 для всех. Engine вообще вредил (+0.004 F1 при удалении).
- **Steam metadata embeddings** (genres + categories + engine → multi-hot → SVD 16 dims): дублирует game_emb, ΔF1 = −0.002.
- **CPU embeddings** (CPU family × game → SVD): слабый сигнал, ΔF1 = +0.001 без них. cpu_vendor + cpu_generation (сами по себе удалены — ΔF1 < 0.002) не помогают.
- **Удалённые hardware фичи**: gpu_vendor, gpu_tier, cpu_vendor, cpu_generation, ram_gb, kernel_major, os_family, proton_major, is_ge_proton, has_proton_version, driver_major — все ΔF1 < 0.002.

## Что помогло (Phase 8)
- **Extended co-occurrence** (gpu + variant + engine + deck axes): +0.008 F1
- **Strict relabeling** 51% tinkering→oob (если нет effort markers в тексте): +0.010 F1
- **Text embeddings** (sentence-transformers → SVD 32): были добавлены ранее, значительный вклад в Stage 1

---

## Задание

Проанализируй текущую модель, данные и результаты. Предложи конкретные новые фичи или подходы, которые могут улучшить модель, особенно Stage 2 (tinkering vs works_oob).

Учитывай ограничения:
1. **При инференсе нет текстов отчётов** — только app_id, GPU, variant. Текстовые фичи помогают при обучении, но при инференсе будут NaN (LightGBM обрабатывает нативно).
2. **Потолок — label noise**: граница tinkering/oob субъективна. Relabeling уже помог, но ~40% ошибок Stage 2 — это шум в данных.
3. **variant доминирует** в Stage 2 — это сильнейший сигнал.

Интересуют:
- Новые фичи из неиспользуемых полей reports (cust_*, flag_*, launch_options, installs/opens/starts_play, duration, fault categories, followup_*_json, launcher, multiplayer, anticheat, frame_rate, readability, control_layout)
- Новые фичи из game_metadata (deck_tests_json, github_*, protondb_tier/score)
- Новые embedding-подходы или трансформации существующих
- Новые подходы к обработке label noise
- Взаимодействия фичей (feature interactions)
- Любые другие идеи

Для каждого предложения укажи:
1. Что за фича и из каких данных
2. Гипотеза — почему поможет
3. Доступна ли при инференсе
4. Ожидаемый эффект (high/medium/low)
5. Стоимость реализации
