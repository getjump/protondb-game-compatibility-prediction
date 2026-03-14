# ML Features — описание и анализ

## Обзор

Модель: **LightGBM** (multi-class classification)
Целевая переменная: `borked` / `needs_tinkering` / `works_oob`
Количество фич: **92**
Источник нормализованных данных: `heuristic` (по умолчанию) или `llm` (переключается через `NORMALIZED_DATA_SOURCE` / `--normalized-data`)

## Результаты (2026-03-10)

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.7237 |
| F1 macro | 0.5081 |
| Samples | 348 536 |
| Train/Test | 278 828 / 69 708 |
| Best iteration | 498 |

### Распределение классов

| Класс | Доля | Precision | Recall | F1 |
|-------|------|-----------|--------|-----|
| borked | 19.7% | 0.68 | 0.34 | 0.45 |
| needs_tinkering | 69.2% | 0.73 | 0.96 | 0.83 |
| works_oob | 11.1% | 0.70 | 0.15 | 0.24 |

### Проблемы

- **Сильный дисбаланс классов**: `needs_tinkering` = 69% → модель предсказывает его почти всегда (recall 96%), игнорируя `borked` (34%) и `works_oob` (15%)
- Precision неплохой по всем классам (~0.70), но recall крайне низкий для миноритарных классов
- Нужен class balancing (sample weights / focal loss / oversampling)

---

## Фичи по группам

### 1. Hardware (из нормализованных таблиц)

Источник: `gpu_normalization_heuristic` / `cpu_normalization_heuristic` / `gpu_driver_normalization`

| # | Фича | Тип | Описание | SHAP |
|---|-------|-----|----------|------|
| 0 | `gpu_vendor` | cat | Вендор GPU: nvidia, amd, intel, virtual, unknown | 0.0220 |
| 1 | `gpu_family` | cat | Семейство GPU: rtx30, rdna2, gcn4, vega, xe, ... | 0.0955 |
| 2 | `gpu_tier` | cat | Уровень GPU: low / mid / high / flagship (из family) | — |
| 3 | `cpu_vendor` | cat | Вендор CPU: intel, amd, unknown | — |
| 4 | `cpu_generation` | num | Поколение CPU (число): Ryzen 5xxx→5, Intel i7-12700→12 | 0.0227 |
| 5 | `ram_gb` | num | Объём RAM в ГБ (из ram_mb или парсинг строки) | 0.0188 |
| 6 | `driver_major` | num | Major-версия GPU-драйвера (NVIDIA: 535, Mesa: 24) | 0.1862 |
| 7 | `kernel_major` | num | Версия ядра Linux (major.minor как float: 6.1, 5.15) | 0.2022 |
| 8 | `os_family` | cat | Семейство ОС: arch, ubuntu, fedora, steamos, ... | 0.0348 |

**Анализ**: `kernel_major` и `driver_major` — 3-й и 4-й по важности. Версии ядра и драйвера сильно влияют на совместимость. `gpu_family` (0.095) тоже значимый — разные GPU по-разному работают с Proton.

### 2. Game metadata (из enrichment: Steam, PCGamingWiki)

Источник: таблица `game_metadata`

| # | Фича | Тип | Описание | SHAP |
|---|-------|-----|----------|------|
| 9 | `engine` | cat | Игровой движок (Unity, Unreal, Source, ...) | — |
| 10 | `graphics_api_dx9` | bool | Использует DirectX 9 | — |
| 11 | `graphics_api_dx11` | bool | Использует DirectX 11 | — |
| 12 | `graphics_api_dx12` | bool | Использует DirectX 12 | — |
| 13 | `graphics_api_vulkan` | bool | Использует Vulkan (нативная поддержка) | — |
| 14 | `graphics_api_opengl` | bool | Использует OpenGL | — |
| 15 | `has_denuvo` | bool | Есть Denuvo DRM (из PCGamingWiki) | — |
| 16 | `drm_count` | num | Количество DRM-систем | — |
| 17 | `anticheat` | cat | Тип античита (EasyAntiCheat, BattlEye, ...) | — |
| 18 | `anticheat_status` | cat | Статус поддержки античита (supported, denied, ...) | — |
| 19 | `developer` | cat | Разработчик (top-50, остальные → "other") | — |
| 20 | `publisher` | cat | Издатель (top-50, остальные → "other") | — |
| 21 | `has_linux_native` | bool | Есть нативная Linux-версия | — |
| 22 | `genre` | cat | Первый жанр из Steam (action, rpg, ...) | — |
| 23 | `is_multiplayer` | bool | Мультиплеер (из Steam categories) | — |
| 24 | `release_year` | num | Год выхода игры | — |

### 3. Aggregated report statistics (per-game)

Агрегированные статистики по всем отчётам для каждой игры. Вычисляются SQL-запросом при обучении.

| # | Фича | Тип | Описание | SHAP |
|---|-------|-----|----------|------|
| 25 | `total_reports` | num | Общее количество отчётов для игры | 0.0485 |
| 26 | `avg_verdict_score` | num | Средний вердикт (1.0=works_oob, 0.5=works, 0.0=borked) | 0.4845 |
| 27 | `pct_works_oob` | num | Доля отчётов "works out of the box" | **1.0567** |
| 28 | `pct_needs_winetricks` | num | Доля использующих winetricks | — |
| 29 | `pct_needs_protontricks` | num | Доля использующих protontricks | — |
| 30 | `pct_needs_custom_proton` | num | Доля с кастомной версией Proton | — |
| 31 | `pct_uses_wine_d3d11` | num | Доля с флагом WINE_D3D11 | — |
| 32 | `pct_uses_d9vk` | num | Доля с D9VK | — |
| 33 | `pct_audio_faults` | num | Доля отчётов с проблемами звука | — |
| 34 | `pct_graphics_faults` | num | Доля с графическими глюками | — |
| 35 | `pct_input_faults` | num | Доля с проблемами ввода | — |
| 36 | `pct_performance_faults` | num | Доля с проблемами производительности | — |
| 37 | `pct_stability_faults` | num | Доля с крашами/зависаниями | — |
| 38 | `pct_deck_battery_ok` | num | Доля Deck-отчётов с норм. батареей | — |
| 39 | `pct_deck_readable` | num | Доля Deck-отчётов с читаемым UI | — |
| 40 | `has_mp_reports` | bool | Есть отчёты о мультиплеере | — |
| 41 | `avg_online_mp_score` | num | Средняя оценка онлайн MP (0..1) | — |
| 42 | `pct_steam_launcher` | num | Доля запусков через Steam | 0.0614 |
| 43 | `pct_lutris_launcher` | num | Доля запусков через Lutris | — |

**Анализ**: `pct_works_oob` (1.06) и `avg_verdict_score` (0.48) — **самые важные фичи**. Это target leakage: агрегаты вердиктов других пользователей по той же игре фактически содержат ответ. Имеет смысл для "что скажут другие", но снижает полезность модели для новых игр без отчётов.

### 4. SVD embeddings

Получены через усечённое SVD из матриц ко-встречаемости (hardware_family × game). Кодируют латентные паттерны: "какие GPU/CPU типичны для каких игр".

| # | Фичи | Описание | Размерность | SHAP (max) |
|---|-------|----------|-------------|------------|
| 44–59 | `gpu_emb_0..15` | GPU-family embedding (SVD правые сингулярные вектора) | 16 (91.2% дисперсии) | — |
| 60–75 | `game_emb_0..15` | Game embedding (SVD из GPU-game матрицы) | 16 | 0.1047 |
| 76–91 | `cpu_emb_0..15` | CPU-family embedding (SVD из CPU-game матрицы) | 16 (100% дисперсии) | — |

**Анализ**: `game_emb_0` (0.10) — 5-я по важности фича. Game embeddings кодируют "тип игры с точки зрения железа" — полезный скрытый сигнал. CPU embeddings менее значимы (100% дисперсии на 16 компонентах = мало данных, всего 17 CPU-семейств).

---

## Top-15 фич по SHAP

| Ранг | Фича | Mean |SHAP| | Группа | Комментарий |
|------|-------|------------|--------|------------|
| 1 | `pct_works_oob` | 1.0567 | aggregated | **Target leakage** — доля OOB по игре |
| 2 | `avg_verdict_score` | 0.4845 | aggregated | **Target leakage** — средний вердикт |
| 3 | `kernel_major` | 0.2022 | hardware | Версия ядра влияет на совместимость |
| 4 | `driver_major` | 0.1862 | hardware | Новый драйвер = лучше совместимость |
| 5 | `game_emb_0` | 0.1047 | embedding | Латентный тип игры |
| 6 | `gpu_family` | 0.0955 | hardware | AMD/NVIDIA семейство |
| 7 | `pct_steam_launcher` | 0.0614 | aggregated | Steam vs другие лаунчеры |
| 8 | `game_emb_1` | 0.0555 | embedding | 2-я компонента game embedding |
| 9 | `total_reports` | 0.0485 | aggregated | Популярность игры |
| 10 | `os_family` | 0.0348 | hardware | Дистрибутив Linux |
| 11 | `game_emb_2` | 0.0256 | embedding | 3-я компонента |
| 12 | `game_emb_3` | 0.0238 | embedding | 4-я компонента |
| 13 | `cpu_generation` | 0.0227 | hardware | Поколение CPU |
| 14 | `gpu_vendor` | 0.0220 | hardware | NVIDIA vs AMD vs Intel |
| 15 | `ram_gb` | 0.0188 | hardware | Объём памяти |

---

## Рекомендации по улучшению

### 1. Target leakage (приоритет: высокий)
`pct_works_oob` и `avg_verdict_score` — это агрегаты целевой переменной по другим отчётам. Модель учится предсказывать "что в среднем говорят другие" вместо "будет ли работать на этой конфигурации". Варианты:
- Исключить или оставить только для warm-start (игры с историей)
- При предсказании для нового железа — обнулять

### 2. Class balancing (приоритет: высокий)
Recall для `borked` (34%) и `works_oob` (15%) критически низкий. Варианты:
- `class_weight` в LightGBM (`is_unbalance=True` или `scale_pos_weight`)
- Focal loss через custom objective
- SMOTE / oversampling миноритарных классов

### 3. Временные фичи
Сейчас нет фич, связанных со временем отчёта. Добавить:
- `report_age_days` — возраст отчёта
- `proton_version` — нормализованная версия Proton (major.minor)

### 4. Enrichment coverage
Game metadata доступен не для всех игр → много None. Расширить enrichment покрытие.
