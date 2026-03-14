# ProtonDB Recommended Settings API

## Обзор

API-сервис, который по паре `(game, hardware)` возвращает оптимальные настройки для Linux-гейминга: версию Proton, launch options, env variables, known issues. Данные — из дампа ProtonDB (~1M+ реальных отчётов).

## Tech Stack

- **Python 3.12+** — единый язык для всего (API, worker, preprocessing, ML)
- **FastAPI** + **uvicorn** — HTTP API
- **SQLite** (`sqlite3` stdlib) — БД, zero infrastructure, WAL mode
- **LightGBM** + **scikit-learn** — ML
- **Click** — CLI (worker, preprocessing, ML train)
- **Pydantic** — валидация, request/response модели

## Вдохновение

- **[protondb-community-api](https://github.com/Trsnaqe/protondb-community-api)** — Go API поверх дампов ProtonDB (MongoDB).
  Авто-обновление данных каждые 31 день. Endpoints: поиск игр по ID/названию, отчёты с фильтрацией,
  версионированные структуры (v1/v2). Проект заброшен (апрель 2023), но полезен как референс
  по парсингу дампов и структуре API. Наши отличия: Python + SQLite, recommendation engine,
  hardware-aware рекомендации, ML-based predictions.

## Источники данных

1. **ProtonDB Data Export** (`github.com/bdefore/protondb-data`)
   - Месячные `.tar.gz` с JSON-отчётами (~350K записей, ~31K уникальных app_ids)
   - Структура: `{ app: { appId, title, store }, systemInfo: { gpu, gpuDriver, cpu, ram, os, kernel }, responses: { ... }, timestamp }`
   - `responses` содержит ~60+ полей: verdict, faults с followUp-детализацией, customizationsUsed (boolean), launchFlagsUsed (boolean), notes (per-field текст), Steam Deck поля, multiplayer и т.д.
   - **Качество данных**: `systemInfo.*` часто содержит мусор (GPU-поле может содержать OS name; ram/kernel — нечитаемые строки). `protonVersion` может быть "Default", "", строка с \n
   - Целевая схема: post-Feb-2022
2. **Steam Store API** (`store.steampowered.com/api/appdetails`)
   - Резолв appId ↔ название игры
   - Кешируется в таблицу `games`
   - `app.title` в дампе ненадёжен (пробелы, Unicode), Steam API — source of truth для имён
3. **Steam Deck Verified** (`store.steampowered.com/saleaction/ajaxgetdeckappcompatibilityreport`)
   - Официальный статус Valve: `resolved_category` = 0 (unknown), 1 (unsupported), 2 (playable), 3 (verified)
   - Детали тестов в `resolved_items` (контроллер, интерфейс, производительность)
   - Без авторизации, кешируется в `game_metadata.deck_status`
   - Используется: в API ответе (deck section) + как ground truth для валидации ML
4. **ProtonDB Summary API** (`protondb.com/api/v1/reports/summaries/{appid}.json`)
   - Агрегированный community-рейтинг: `tier` = platinum/gold/silver/bronze/borked, `score`, `confidence`
   - Кешируется в `game_metadata.protondb_tier`
   - Используется: как ground truth для валидации ML predictions

## Архитектура

### Разделение ответственности

Всё на Python, единый codebase. Три режима запуска, общая БД:

```
┌─────────────────────────────────────────────────────────┐
│ protondb-settings worker sync                            │
│   Скачивает дамп → парсит JSON → INSERT в reports/games │
│   Единственный, кто пишет в reports и games             │
└─────────────────────┬───────────────────────────────────┘
                      │ SQLite (WAL mode)
┌─────────────────────┴───────────────────────────────────┐
│ protondb-settings preprocess run                         │
│   Читает из reports → пишет в extracted_data,            │
│   game_metadata, gpu_normalization                       │
│   НЕ включает ML training (отдельная команда)           │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│ protondb-settings serve                                  │
│   FastAPI + uvicorn, read-only доступ к БД              │
│   Загружает model.pkl + embeddings.npz при старте       │
│   SHAP, LightGBM prediction — нативно в том же процессе │
└─────────────────────────────────────────────────────────┘
```

### Worker

Два режима:

**`protondb-settings worker check`** — быстрая проверка обновлений (секунды, без скачивания):
1. GitHub API: `GET /repos/bdefore/protondb-data/releases/latest` → `tag_name`
2. Сравнить с `meta.dump_release_tag`
3. Вывести: `"up to date"` или `"new dump available: monthly_2025_03 (current: monthly_2025_02)"`
4. Exit code: 0 = есть обновление, 1 = актуально (для cron/скриптов)

**`protondb-settings worker sync`** — импорт (вручную или cron):
1. Вызывает `check` → если актуально и нет `--force`, exit
2. Скачивает `.tar.gz`, считает SHA-256
3. Проверяет `meta.dump_sha256` — если совпадает, skip (дедупликация)
4. Распаковывает, парсит JSON-отчёты
5. Batch UPSERT в `reports` и `games` (транзакция каждые 5000 записей)
6. Записывает `dump_release_tag`, `dump_sha256`, `dump_imported_at` в `meta`
- Все поля сохраняются **as-is** — без очистки, трансформации, нормализации
- **Не** запускает preprocessing и не пересчитывает recommendations

### Preprocessing

Запускается после worker, отдельно. **Обязательный шаг** — без него engine не работает.

1. **Data cleaning** — очистка сырых данных: `ram` → `ram_mb`, `proton_version` trim/"Default"→NULL, `kernel` regex
2. **Enrichment** — заполняет `game_metadata` из внешних API (см. `preprocessing/PLAN_ENRICHMENT.md`)
3. **LLM normalization** — заполняет `gpu_normalization`, `cpu_normalization`, `launch_options_parsed`
4. **LLM extraction** — заполняет `extracted_data` (см. `preprocessing/PLAN_LLM.md`)

```bash
# Проверка обновлений (быстро, без скачивания):
protondb-settings worker check                # новый дамп? (1 HTTP запрос)
protondb-settings preprocess check            # статус всех шагов

# Полный pipeline (auto-resume — можно прервать и перезапустить):
protondb-settings worker sync
protondb-settings preprocess run              # cleaning + enrichment (не требует LLM)
protondb-settings preprocess llm all --model anthropic/claude-haiku  # LLM шаги (требует --model)
protondb-settings ml train                   # ML training (Phase 4)

# Или по шагам:
protondb-settings preprocess run --step cleaning
protondb-settings preprocess run --step enrichment
protondb-settings preprocess llm normalize-gpu
protondb-settings preprocess llm extract
protondb-settings ml train
```

### Server

- Запуск: `protondb-settings serve --port 8080 --db ./data/protondb.db`
- FastAPI + uvicorn, read-only доступ к БД (WAL mode позволяет читать во время записи)
- Загружает `model.pkl` + `embeddings.npz` в память при старте
- SHAP, LightGBM prediction — нативно в том же процессе (нет impedance mismatch)

## Схема БД (SQLite)

Это **единственный source of truth** для схемы. Все остальные планы ссылаются сюда.

```sql
-- === Core (заполняется worker) ===

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Используемые ключи:
--   dump_release_tag    — GitHub release tag последнего импортированного дампа ("monthly_2025_02")
--   dump_sha256         — SHA-256 скачанного файла (для дедупликации)
--   dump_imported_at    — ISO timestamp последнего импорта
--   awacy_etag          — ETag от AreWeAntiCheatYet games.json
--   awacy_fetched_at    — ISO timestamp последнего fetch

-- === Pipeline durability ===

CREATE TABLE pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        TEXT NOT NULL,       -- DB: underscores. CLI: hyphens (normalize-gpu → normalize_gpu)
                                     -- Preprocessing: 'cleaning', 'enrichment', 'normalize_gpu', 'normalize_cpu',
                                     -- 'parse_launch_options', 'extract'
                                     -- ML (отдельная команда): 'ml_train'
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,                -- NULL = in progress или прервано
    total_items INTEGER,             -- сколько всего нужно обработать
    processed   INTEGER DEFAULT 0,   -- сколько обработано (обновляется каждый batch commit)
    status      TEXT DEFAULT 'running',  -- 'running' | 'completed' | 'failed'
    error       TEXT,                -- последняя ошибка
    dump_tag    TEXT                  -- против какой версии дампа запущено
);
CREATE INDEX idx_pipeline_step_status ON pipeline_runs(step, status);

CREATE TABLE games (
    app_id     INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- reports.id берётся из JSON дампа: каждый отчёт имеет уникальный ID (строка).
-- Формат: случайный ID из ProtonDB (e.g. "YCxB2ZIp"). Worker сохраняет as-is.
-- Это обеспечивает стабильный ID для UPSERT при повторных импортах.
--
-- Worker выполняет маппинг camelCase (JSON дамп) → snake_case (SQL):
--   app.steam.appId          → app_id
--   systemInfo.gpu           → gpu
--   systemInfo.gpuDriver     → gpu_driver
--   responses.protonVersion  → proton_version
--   responses.customProtonVersion → custom_proton_version
--   responses.verdictOob     → verdict_oob
--   responses.triedOob       → tried_oob
--   responses.startsPlay     → starts_play
--   responses.audioFaults    → audio_faults
--   responses.concludingNotes → concluding_notes
--   responses.batteryPerformance → battery_performance
--   responses.controlLayout  → control_layout
--   responses.isImpactedByAntiCheat → is_impacted_by_anticheat
--   responses.isMultiplayerImportant → is_multiplayer_important
--   responses.customizationsUsed.configChange → cust_config_change
--   responses.launchFlagsUsed.useWineD3d11 → flag_use_wine_d3d11
--   responses.followUp.audioFaults → followup_audio_faults_json (сериализуется как JSON)
--   responses.notes.verdict  → notes_verdict
--   (и т.д. — все responses.* поля по тому же паттерну)
-- launch_options хранится as-is (без trim) для exact match с launch_options_parsed

CREATE TABLE reports (
    id               TEXT PRIMARY KEY,
    app_id           INTEGER NOT NULL REFERENCES games(app_id),
    timestamp        TEXT NOT NULL,

    -- === systemInfo (из дампа as-is, очистка — в preprocessing) ===
    gpu              TEXT,        -- 35K уникальных, бывает OS вместо GPU
    gpu_driver       TEXT,
    cpu              TEXT,        -- бывает мусор
    ram              TEXT,        -- сырая строка ("16 GB"). Worker сохраняет as-is
    os               TEXT,
    ram_mb           INTEGER,     -- заполняется cleaning: regex(\d+) из ram, NULL если мусор
    kernel           TEXT,        -- бывает мусор
    window_manager   TEXT,        -- KWin 55.8%, GNOME Shell 19.4%, Mutter и т.д. (полезно для Wayland/X11)

    -- === responses: основные поля ===
    type             TEXT,        -- "steamPlay" (23%) | "tinker" (6.6%). Присутствует в 29.6%
    variant          TEXT,        -- "official"|"experimental"|"ge"|"native"|"older"|"notListed". 70.4%
    proton_version   TEXT,        -- messy: 78% "Default", "", версии с \n. Реальных ~25K
    custom_proton_version TEXT,   -- 12%, 1087 уникальных. GE-Proton версии! Важный источник
    verdict          TEXT,        -- "yes" (80.3%) | "no" (19.7%). 100% покрытие
    tried_oob        TEXT,        -- "yes"|"no". 29.1%
    verdict_oob      TEXT,        -- "yes"|"no". 19.6%
    tinker_override  TEXT,        -- "yes" (1.6%) | "no" (31%). 32.6%

    -- === responses: воронка запуска (91-100% покрытие) ===
    installs         TEXT,        -- "yes" (99.7%) | "no". Устанавливается ли
    opens            TEXT,        -- "yes" (91.1%) | "no". Открывается ли
    starts_play      TEXT,        -- "yes" (96%) | "no". Начинает ли играться
    duration         TEXT,        -- 24.5%: severalHours|moreThanTenHours|lessThanAnHour|aboutAnHour|lessThanFifteenMinutes
    extra            TEXT,        -- "yes"|"no". 24.4% — флаг "есть доп. информация"

    -- === responses: fault fields (yes/no/blank) ===
    audio_faults     TEXT,
    graphical_faults TEXT,
    input_faults     TEXT,
    performance_faults TEXT,
    stability_faults TEXT,
    windowing_faults TEXT,
    save_game_faults TEXT,
    significant_bugs TEXT,

    -- === responses.followUp.* — детализация faults ===
    -- 32.1% отчётов имеют followUp. Хранятся как JSON (checkbox dict: key→True/False)
    followup_audio_faults_json      TEXT,  -- borked, crackling, lowQuality, missing, outOfSync, other
    followup_graphical_faults_json  TEXT,  -- heavyArtifacts, minorArtifacts, missingTextures, other
    followup_input_faults_json      TEXT,  -- bounding, controllerMapping, controllerNotDetected, drifting, inaccuracy, lag, other
    followup_performance_faults_json TEXT, -- НЕ checkbox, а enum: slightSlowdown (26.8K) | significantSlowdown (14.2K)
    followup_stability_faults_json  TEXT,  -- НЕ checkbox, а enum: occasionally (14.3K) | frequentCrashes (8.3K) | notListed (6.6K)
    followup_windowing_faults_json  TEXT,  -- activatingFullscreen, fullNotFull, switching, other
    followup_save_game_faults_json  TEXT,  -- errorLoading, errorSaving, other
    followup_anticheat_json         TEXT,  -- battleEye, easyAntiCheat, other (тип конкретного античита!)
    followup_control_cust_json      TEXT,  -- enableGripButtons, gyro, rightTrackpad, other (Steam Deck)

    -- === responses.customizationsUsed.* — структурированные boolean ===
    cust_winetricks       INTEGER,  -- 0/1
    cust_protontricks     INTEGER,
    cust_config_change    INTEGER,
    cust_custom_prefix    INTEGER,
    cust_custom_proton    INTEGER,
    cust_lutris           INTEGER,
    cust_media_foundation INTEGER,
    cust_protonfixes      INTEGER,
    cust_native2proton    INTEGER,
    cust_not_listed       INTEGER,

    -- === responses.launchFlagsUsed.* — структурированные boolean (3.4% отчётов) ===
    flag_use_wine_d3d11      INTEGER,  -- 4,384
    flag_disable_esync       INTEGER,  -- 3,305
    flag_enable_nvapi        INTEGER,  -- 2,803
    flag_disable_fsync       INTEGER,  -- 1,383
    flag_use_wine_d9vk       INTEGER,  -- 820
    flag_large_address_aware INTEGER,  -- 669
    flag_disable_d3d11       INTEGER,  -- 522
    flag_hide_nvidia         INTEGER,  -- 269
    flag_game_drive          INTEGER,  -- 258
    flag_no_write_watch      INTEGER,
    flag_no_xim              INTEGER,
    flag_old_gl_string       INTEGER,
    flag_use_seccomp         INTEGER,

    -- === responses: текстовые поля для LLM-экстракции ===
    launch_options   TEXT,        -- сырая строка (16K уникальных значений)
    concluding_notes TEXT,        -- свободный текст (106K уникальных)

    -- === responses.notes.* — per-field заметки пользователя ===
    notes_verdict          TEXT,
    notes_audio_faults     TEXT,
    notes_graphical_faults TEXT,
    notes_performance_faults TEXT,
    notes_stability_faults TEXT,
    notes_windowing_faults TEXT,
    notes_input_faults     TEXT,
    notes_significant_bugs TEXT,
    notes_save_game_faults TEXT,
    notes_extra            TEXT,
    notes_launch_flags     TEXT,
    notes_customizations   TEXT,
    notes_variant          TEXT,
    notes_proton_version   TEXT,
    notes_concluding_notes TEXT,

    -- === Steam Deck / портатив (13% отчётов — ~45K) ===
    battery_performance  TEXT,    -- "no" (83.9%) | "yes" (16.1%)
    readability          TEXT,    -- "no" (87.2%) | "yes" (12.8%)
    control_layout       TEXT,    -- community (30%)|official (22.6%)|keyboardAndMouse (18.5%)|gamepadWithMouseTrackpad|gamepad|gamepadWithGyro|gamepadWithJoystick|mouseOnly
    did_change_control_layout TEXT, -- "no" (81.1%) | "yes" (18.9%)
    control_layout_customization TEXT, -- "yes" (55.4%) | "no" (44.6%). 2.4%
    frame_rate           TEXT,    -- 0.1% (376 записей): gt60|30to60|20to30|lt20. Малополезно
    notes_control_layout TEXT,
    notes_control_layout_customization TEXT,
    notes_battery_performance TEXT,
    notes_readability    TEXT,

    -- === Launcher ===
    launcher             TEXT,    -- "steam" | "lutris" | "bottles" | "gamehub" | "notListed"
    secondary_launcher   TEXT,    -- "yes" | "no"
    notes_launcher       TEXT,
    notes_secondary_launcher TEXT,

    -- === Multiplayer ===
    is_multiplayer_important TEXT, -- "yes" (59%) | "no" (41%). 9.2%
    local_mp_attempted   TEXT,    -- "no" (91.8%) | "yes". 27.6%
    local_mp_played      TEXT,    -- "yes" (95.5%) | "no". 2.3%
    local_mp_appraisal   TEXT,    -- excellent (86%)|good|acceptable|weak|awful. 2.2%
    notes_local_mp       TEXT,
    online_mp_attempted  TEXT,    -- "no" (50.2%) | "yes" (49.8%). 27.6%
    online_mp_played     TEXT,    -- "yes" (94.3%) | "no". 13.8%
    online_mp_appraisal  TEXT,    -- excellent (76.8%)|good (13.8%)|acceptable|weak|awful. 13.4%
    notes_online_mp      TEXT,

    -- === Anti-cheat ===
    is_impacted_by_anticheat TEXT,  -- "no" (92.4%) | "yes" (7.6%). 9.2%
    notes_anticheat      TEXT,
    -- followup_anticheat_json содержит конкретный тип: battleEye, easyAntiCheat, other

    -- === Tinker override ===
    notes_tinker_override TEXT
);

CREATE INDEX idx_reports_app_id ON reports(app_id);
CREATE INDEX idx_reports_gpu ON reports(gpu);
CREATE INDEX idx_reports_timestamp ON reports(timestamp);
CREATE INDEX idx_reports_verdict ON reports(verdict);
CREATE INDEX idx_reports_launch_options ON reports(launch_options);  -- JOIN с launch_options_parsed

-- === Preprocessing: enrichment (заполняется preprocessing/enrichment) ===

CREATE TABLE game_metadata (
    app_id              INTEGER PRIMARY KEY REFERENCES games(app_id),
    developer           TEXT,
    publisher           TEXT,
    genres              TEXT,          -- JSON array
    categories          TEXT,          -- JSON array
    release_date        TEXT,
    has_linux_native    INTEGER,
    engine              TEXT,
    graphics_apis       TEXT,          -- JSON: ["DirectX 12", "Vulkan"]
    drm                 TEXT,          -- JSON: ["Denuvo Anti-Tamper", "Steam"] из PCGamingWiki Availability.Uses_DRM
    anticheat           TEXT,
    anticheat_status    TEXT,          -- from AreWeAntiCheatYet
    deck_status         INTEGER,       -- Steam Deck Verified: 0=unknown, 1=unsupported, 2=playable, 3=verified
    deck_tests_json     TEXT,          -- JSON: [{display_type, loc_token}] — детали тестов Valve
    protondb_tier       TEXT,          -- community tier: platinum/gold/silver/bronze/borked
    protondb_score      REAL,          -- numeric score 0..1
    protondb_confidence TEXT,          -- strong/good/weak
    protondb_trending   TEXT,          -- trending tier (API: trendingTier → protondb_trending)
    enriched_at         TEXT DEFAULT (datetime('now'))
);

-- === Preprocessing: GPU normalization ===

CREATE TABLE gpu_normalization (
    raw_string      TEXT PRIMARY KEY,   -- "NVIDIA GeForce GTX 1060 6GB"
    vendor          TEXT NOT NULL,      -- "nvidia" | "amd" | "intel" | "unknown"
    family          TEXT NOT NULL,      -- "gtx10" | "rdna2" | "custom_apu" | "unknown"
    model           TEXT NOT NULL,      -- "gtx1060" | "steam_deck_apu" | "unknown"
    normalized_name TEXT NOT NULL,      -- "NVIDIA GTX 1060" | "AMD Custom APU (Steam Deck)"
    is_apu          INTEGER DEFAULT 0,  -- кастомные APU (Steam Deck, ROG Ally, etc.)
    is_igpu         INTEGER DEFAULT 0,  -- встроенная графика
    is_virtual      INTEGER DEFAULT 0   -- llvmpipe, virgl, vmware
    -- vendor=unknown для мусорных строк; scoring engine даёт им weight=0
);

CREATE INDEX idx_gpu_norm_family ON gpu_normalization(family);
CREATE INDEX idx_gpu_norm_vendor ON gpu_normalization(vendor);

CREATE TABLE cpu_normalization (
    raw_string      TEXT PRIMARY KEY,   -- "AMD Ryzen 7 5800X"
    vendor          TEXT NOT NULL,      -- "intel" | "amd" | "unknown"
    family          TEXT NOT NULL,      -- "zen3" | "alder_lake" | "custom_apu" | "unknown"
    model           TEXT NOT NULL,      -- "ryzen7_5800x" | "unknown"
    normalized_name TEXT NOT NULL,
    generation      INTEGER,            -- поколение (12, 13 для Intel; Zen 3 → 3)
    is_apu          INTEGER DEFAULT 0
    -- vendor=unknown для мусора; не отбрасываем ничего
);

CREATE INDEX idx_cpu_norm_family ON cpu_normalization(family);
CREATE INDEX idx_cpu_norm_vendor ON cpu_normalization(vendor);

-- === Preprocessing: LLM launch options parsing ===

CREATE TABLE launch_options_parsed (
    raw_string       TEXT PRIMARY KEY,  -- уникальная launch options строка
    env_vars_json    TEXT,              -- [{"name":"KEY","value":"VALUE"}]
    wrappers_json    TEXT,              -- [{"tool":"gamescope","args":"-W 1920 -H 1080"}]
    game_args_json   TEXT,              -- ["-dx11","-skipintro"]
    unparsed         TEXT               -- что не удалось разобрать
);

-- === Preprocessing: LLM text extraction (см. preprocessing/PLAN_LLM.md) ===

CREATE TABLE extracted_data (
    report_id         TEXT PRIMARY KEY REFERENCES reports(id),
    app_id            INTEGER NOT NULL REFERENCES games(app_id),
    actions_json      TEXT,       -- [{type, value, detail, reported_effect, conditions, risk}]
    observations_json TEXT,       -- [{symptom, description, hardware_specific}]
    useful            INTEGER,    -- 0/1
    processed_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_extracted_app_id ON extracted_data(app_id);
CREATE INDEX idx_extracted_useful ON extracted_data(useful);
```

## Recommendation Engine

### Стратегия: on-the-fly с кешированием

Рекомендации вычисляются **при запросе** (on-the-fly), а не precomputed. Причины:
- Проще архитектура (нет отдельного шага пересчёта)
- Всегда актуальные данные
- Достаточно быстро (SQLite + индексы)

**Кеширование**: in-memory LRU cache (TTL = 1 час).
Key format: `app_id:gpu_family:cpu_family` для POST, `app_id:__no_hw__` для GET (без hardware) — разные ключи, нет коллизий.
Инвалидация: при cache miss сервер проверяет `meta.dump_imported_at` — если изменился с последнего check, сбрасывает весь кеш. Если ML-артефакты отсутствуют при старте — server fail-fast с понятной ошибкой.

### Engine (ML-based)

Engine основан на ML моделях, не на ручных эвристиках. Preprocessing — обязательный шаг перед запуском сервера.

**Данные, используемые engine:**
- `gpu_normalization`, `cpu_normalization` — hardware matching через embeddings
- `extracted_data.actions_json` — действия с `reported_effect` (effective/ineffective)
- `extracted_data.observations_json` — симптомы → known_issues
- `launch_options_parsed` — структурированный разбор launch options
- `game_metadata` — engine, anticheat, DRM, developer → features для ML + контекст в ответе
- Structured fields из `reports` — `customizationsUsed.*`, `launchFlagsUsed.*`, `followUp.*`, `variant`, `proton_version`

**ML модели (одна модель + embeddings):**
1. **Hardware embeddings** (SVD) — learned similarity для GPU/CPU; используются как features для LightGBM + для cosine similarity search
2. **LightGBM classifier** — предсказывает P(works_oob), P(needs_tinkering), P(borked) для (game, hardware), используя embedding-фичи + categorical + game_metadata

```
POST /recommendations { app_id, hardware: { gpu, cpu, ram_gb } }
    │
    ▼
[1. Compatibility prediction] ─── LightGBM: P(works), P(tinkering), P(borked)
    │
    ▼
[2. Hardware similarity] ─── GPU/CPU embeddings → cosine similarity → top-N отчётов
    │
    ▼
[3. Settings aggregation] ─── см. ниже
    │
    ▼
[4. Response] ─── prediction + settings + game_metadata + known_issues
    │
    ▼
[5. Cache (LRU, TTL=1h)]
```

Подробности ML pipeline — в `PLAN_ML.md`.

### Settings Aggregation (шаг 3)

Ключевой алгоритм: как из сотен отчётов получить одну чистую рекомендацию.

**Источники данных (три потока, merge в единый пул actions):**
- `extracted_data.actions_json` — LLM-экстракция из свободного текста
- `launch_options_parsed` — LLM-парсинг launch options строк
- Structured fields из `reports` — `launchFlagsUsed.*`, `customizationsUsed.*`

Structured fields конвертируются в тот же формат action при загрузке:
```
flag_enable_nvapi=1 → {type: "env_var", value: "PROTON_ENABLE_NVAPI=1", source: "structured"}
cust_custom_proton=1 → {type: "runner_selection", value: custom_proton_version, source: "structured"}
```

**`reported_effect` для structured fields** — выводится из `verdict` того же отчёта:
```
flag_enable_nvapi=1 + verdict="yes" → reported_effect="effective"
flag_enable_nvapi=1 + verdict="no"  → reported_effect="ineffective"
```
Structured field сам по себе говорит только "использовал", а `verdict` — помогло ли.

Это позволяет объединить все три источника в один пул и считать по ним единую статистику.

#### Алгоритм

```
1. COLLECT
   Все actions для данного app_id из отчётов, отобранных по hardware similarity (top-N).
   Каждый action имеет: type, value, reported_effect, conditions, risk, source, report_timestamp.

2. GROUP — по canonical key = (type, normalized_value)
   Нормализация value:
     - env_var: uppercase key, trim spaces ("PROTON_ENABLE_NVAPI = 1" → "PROTON_ENABLE_NVAPI=1")
     - runner_selection: normalize version ("GE-Proton8-4" → "GE-Proton")
     - game_arg: lowercase, trim
   Одинаковые actions из разных источников (LLM + structured) группируются вместе.

3. SCORE — для каждой группы:
   effective_count   = count(reported_effect = "effective")
   ineffective_count = count(reported_effect = "ineffective")
   unclear_count     = count(reported_effect = "unclear")

   effectiveness = effective_count / (effective_count + ineffective_count)   # [0..1], unclear не учитывается
                                                                            # если оба 0 (только unclear) → effectiveness = 0.5 (neutral, не penalize)
   confidence    = log2(effective_count + ineffective_count + 1)             # больше данных = выше уверенность
   recency_boost = avg(1.0 / (1 + years_since(report_timestamp)))           # свежие отчёты весят больше
   score         = effectiveness × confidence × recency_boost

4. FILTER
   - risk = "safe" (исключить risky: sudo, rm, /etc)
   - effectiveness > 0.6 (больше помогает чем нет)
   - effective_count >= 2 (минимум 2 подтверждения)
   - conditions match user hardware (conditions — list of {kind, value} из LLM):
       any(c.kind=="gpu_vendor") → c.value == user.gpu_vendor (нет такого condition = подходит всем)
       аналогично для distro, proton_version (если user не указал — condition игнорируется)

5. CONFLICT RESOLUTION
   Взаимоисключающие actions (один и тот же type+key, разные values):
     env_var:PROTON_USE_WINED3D=1  vs  env_var:PROTON_USE_WINED3D=0 → берём с высшим score
     game_arg:-dx11  vs  game_arg:-dx12 → берём с высшим score
   Совместимые actions (разные keys) — все включаются.
   runner_selection (Proton) — **отдельная логика**, см. "Proton Version Selection".

6. RANK
   Сортировка по score desc. Top-K actions (K=10) → в ответ.

7. COMPOSE
   Merge actions → финальные поля ответа:
     env_vars с type="env_var"      → response.env_variables + response.launch_options
     type="wrapper_config"          → response.launch_options (gamescope, mangohud)
     type="game_arg"                → response.launch_options (%command% args)
     type="runner_selection"        → response.proton (обрабатывается отдельно)
     type="protontricks_verb"       → response.protontricks[]
     type="dll_override"            → response.env_variables (WINEDLLOVERRIDES)
   launch_options собирается: "ENV1=VAL1 ENV2=VAL2 [wrappers] %command% [game_args]"
```

#### Proton Version Selection

Proton — **особый случай**. Версии линейно упорядочены, и новая обычно superset старой (баги фиксятся, совместимость растёт). Поэтому `max(score)` неправильный подход — нужна специальная логика.

**Источники данных о Proton (три, merge):**
- `reports.variant` — `official` / `experimental` / `ge` / `native` / `older`
- `reports.proton_version` — конкретная версия (`"Proton 9.0-1"`, `"Proton Experimental"`)
- `reports.custom_proton_version` — GE-Proton версии (`"GE-Proton8-4"`)
- `extracted_data.actions` с `type="runner_selection"` — из текста отчётов

**Нормализация версий:**
```
"Proton 9.0-1"       → family=official,    semver=(9,0,1)
"Proton 8.0-5"       → family=official,    semver=(8,0,5)
"Proton Experimental"→ family=experimental, semver=MAX  (всегда "новейший")
"GE-Proton8-4"       → family=ge,          semver=(8,4)
"GE-Proton9-7"       → family=ge,          semver=(9,7)
"native"             → family=native,      semver=N/A
```

**Алгоритм:**

```
1. Собрать все (version, verdict) из похожих отчётов для данного app_id
   verdict: effective (works/works_oob) / ineffective (borked)

2. Для каждого family (official, experimental, ge) отдельно:

   versions_effective = версии с verdict=effective, отсортированные по semver
   versions_broken    = версии с verdict=ineffective

   IF versions_effective пусто:
     family не рекомендуется

   ELSE:
     max_effective = max(semver) среди effective
     min_effective = min(semver) среди effective

     # Ключевая эвристика:
     # Если эффективные версии разбросаны (включая новые) →
     #   новые версии работают, можно брать новейшую
     # Если эффективные версии только старые, а новые broken →
     #   регрессия, рекомендовать конкретную старую

     has_recent_effective = max_effective.semver >= (8, 0)  # Proton 8.0+ считается "recent"
     has_recent_broken    = any(v in versions_broken where v.semver > max_effective.semver)

     IF has_recent_effective AND NOT has_recent_broken:
       # Новые версии работают, регрессии нет → рекомендуем "latest" или max_effective
       recommend = "Proton Experimental" (if family=official/experimental)
                 | max_effective (if family=ge)
       pinned = false

     ELSE IF has_recent_broken:
       # Регрессия: новее max_effective есть broken → pin к конкретной версии
       recommend = max_effective  # конкретная версия, не "latest"
       pinned = true

     ELSE:
       # Только старые отчёты, непонятно → рекомендуем max_effective
       recommend = max_effective
       pinned = false  # не уверены что это регрессия

3. Выбор между families:
   Приоритет: official > experimental > ge (если score одинаковый)
   Но если GE-Proton имеет значительно больше effective отчётов → рекомендуем GE

4. Формирование ответа:
   response.proton = {
     recommended: "Proton Experimental",     # или "Proton 8.0-5" если pinned
     pinned: false,                          # true = не обновлять!
     alternatives: ["GE-Proton9-7"],         # другие рабочие families
     avoid: ["Proton 9.0-1"],               # версии с broken отчётами
     note: "Recent reports confirm latest Proton works"  # или "Pin to 8.0-5, regression in 9.x"
   }
```

**Примеры:**

```
# Игра работает на новых версиях → рекомендуем latest
Effective: [7.0-1, 8.0-2, 8.0-5, 9.0-1, Experimental]
Broken:    [7.0-1 (старый)]
→ recommended: "Proton Experimental", pinned: false

# Регрессия: 9.x сломал игру
Effective: [7.0-1, 8.0-2, 8.0-5]
Broken:    [9.0-1, 9.0-2]
→ recommended: "Proton 8.0-5", pinned: true, avoid: ["9.0-1", "9.0-2"]

# Только GE-Proton работает (нативный Proton сломан)
Official effective: []
GE effective: [GE-Proton8-4, GE-Proton9-7]
→ recommended: "GE-Proton9-7", alternatives: []

# native (Linux порт) лучше Proton
Native effective: 15 reports
Official effective: 3 reports (с issues)
→ recommended: "Native Linux version", alternatives: ["Proton Experimental"]
```

#### Пример (полный)

```
Cyberpunk 2077 + RTX 3070:

47 похожих отчётов → 200 actions

После GROUP:
  env_var:PROTON_ENABLE_NVAPI=1     23 effective, 4 ineffective  → eff=0.85, score=2.8
  env_var:DXVK_ASYNC=1              15 effective, 1 ineffective  → eff=0.94, score=2.6
  game_arg:-dx11                     3 effective, 0 ineffective  → eff=1.00, score=1.1
  env_var:VKD3D_CONFIG=dxr11         2 effective, 5 ineffective  → eff=0.29  ← отфильтрован (eff<0.6)

Proton selection (отдельно):
  Official: effective=[8.0-5, 9.0-1, Experimental], broken=[]
  GE:       effective=[GE-Proton8-4, GE-Proton9-7], broken=[]
  → recommended: "Proton Experimental", pinned: false
  → alternatives: ["GE-Proton9-7"]

После FILTER (hardware=nvidia, risk=safe, eff>0.6, count>=2):
  PROTON_ENABLE_NVAPI=1  ✓ (conditions.gpu_vendor=nvidia, совпадает)
  DXVK_ASYNC=1           ✓ (no conditions)
  -dx11                  ✓
  VKD3D_CONFIG=dxr11     ✗ (eff=0.29)

COMPOSE:
  launch_options: "PROTON_ENABLE_NVAPI=1 DXVK_ASYNC=1 %command% -dx11"
  proton: { recommended: "Proton Experimental", pinned: false,
            alternatives: ["GE-Proton9-7"], avoid: [] }
```

#### known_issues (из observations)

`extracted_data.observations_json` агрегируется аналогично:
1. GROUP по `(symptom, description)` — нормализация текста
2. Частота: сколько отчётов упоминают этот issue
3. `hardware_specific=true` → фильтр по hardware
4. Top-K по частоте → `response.known_issues[]`

#### Merge structured + LLM

Structured fields (`launchFlagsUsed.enableNvapi=1`) и LLM extraction (`"Set PROTON_ENABLE_NVAPI=1"`) описывают одно и то же. При группировке они попадают в одну группу по canonical key, что:
- Увеличивает `effective_count` (два источника подтверждают друг друга)
- Повышает confidence рекомендации
- Покрывает отчёты без текста (structured) и отчёты без checkbox'ов (LLM)

### GPU / CPU normalization (LLM по уникальным значениям)

Таблицы `gpu_normalization` и `cpu_normalization` заполняются на этапе preprocessing:
1. Собираем уникальные строки из `reports.gpu` (~35K) и `reports.cpu`
2. LLM нормализует batch-ом: vendor, family, model, is_apu, is_igpu
3. Мусорные строки (OS name в поле GPU и т.п.) → `vendor=unknown`
4. **Ничего не отбрасываем** — кастомные APU (Steam Deck, ROG Ally), iGPU, серверные CPU — всё валидно

**Роль normalization vs embeddings:**
- `gpu_normalization` — справочник: LLM-разбор raw string → vendor/family/model. Заполняется в preprocessing
- **Hardware embeddings** (Phase 4) — learned similarity: обучаются на матрице (GPU×Game→verdict). Используют `family` из normalization как ключ
- При API запросе: user GPU string → lookup в `gpu_normalization` → `family` → embedding → cosine similarity с отчётами
- `vendor=unknown` → нет embedding, не участвует в similarity matching (отчёты с мусорным hardware игнорируются при поиске похожих)

Покрывает в том числе:
```
"NVIDIA GeForce GTX 1060 6GB"    → nvidia, gtx10, gtx1060, apu=0
"AMD Custom GPU 0405"             → amd, custom_apu, steam_deck_apu, apu=1
"Intel HD Graphics 630"           → intel, hd600, hd630, igpu=1
"Arch Linux"                      → unknown, unknown, unknown (мусор)
```

### ML pipeline

См. `PLAN_ML.md` — hardware embeddings + LightGBM classification (одна модель). ML — ядро engine, не опциональное улучшение.

## API Endpoints

```
GET  /health
     → { "status": "ok", "version": "0.1.0", "reports_count": N, "last_import": "...", "dump_tag": "..." }

GET  /games/search?q={name}
     → [{ "app_id": 1091500, "name": "Cyberpunk 2077" }]

GET  /games/{app_id}
     → { "app_id": 1091500, "name": "Cyberpunk 2077", "total_reports": 2518,
         "game_metadata": { "engine": "REDengine", "graphics_apis": ["DirectX 12","Vulkan"],
                             "drm": ["DRM-free"], "anticheat": null, "anticheat_status": null,
                             "developer": "CD PROJEKT RED", "publisher": "CD PROJEKT RED",
                             "release_date": "2020-12-10", "has_linux_native": false } }

POST /recommendations
     ← { "app_id": 1091500, "hardware": { "gpu": "...", "cpu": "...", "ram_gb": 32 },
          "device": "desktop" }
     // device: "desktop" (default) | "steam_deck"
     → {
         "prediction": {
           "experience": "works_out_of_box",
           "works_probability": 0.91,
           "confidence": 0.87,
           "based_on": 47,
           "factors": [                              // SHAP top-3 (почему такой prediction)
             { "feature": "anticheat", "value": "none", "impact": +0.15 },
             { "feature": "engine", "value": "REDengine", "impact": +0.08 },
             { "feature": "has_denuvo", "value": false, "impact": +0.06 }
           ]
         },
         "proton": { "recommended": "Proton Experimental", "pinned": false,
                     "alternatives": ["GE-Proton9-7"], "avoid": [] },
         "launch_options": "PROTON_ENABLE_NVAPI=1 %command%",
         "env_variables": { "PROTON_ENABLE_NVAPI": "1" },
         "known_issues": [{ "description": "...", "frequency": 0.15 }],
         "game_metadata": { "engine": "REDengine", "drm": ["DRM-free"],
                            "anticheat": null, "anticheat_status": null },
         // Только для device="steam_deck":
         "deck": {
           "valve_status": "verified",              // verified/playable/unsupported/unknown
           "battery_ok_pct": 0.72,                  // % отчётов с battery_performance=yes
           "readable_pct": 0.85,                    // % readability=yes
           "recommended_layout": "community"        // самый популярный control_layout
         }
       }

GET  /recommendations/{app_id}
     → То же, но без hardware-фильтра:
       - prediction: hardware features = NaN/missing (LightGBM нативно поддерживает missing values)
       - factors[]: исключать hardware features из SHAP top-3 (значение NaN бессмысленно для пользователя)
       - settings aggregation по ВСЕМ отчётам (без hardware similarity filtering)
       - proton selection по всем отчётам
```

## Структура проекта

```
protondb-recommended-settings/
├── protondb_settings/              # единый Python-пакет
│   ├── __init__.py
│   ├── cli.py                      # Click CLI: serve, worker, preprocess, ml
│   ├── config.py                   # пути к БД, порт, rate limits
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py           # SQLite подключение, WAL mode, PRAGMA
│   │   ├── migrations.py           # единственное место со схемой
│   │   ├── games.py
│   │   ├── reports.py
│   │   └── extracted.py            # чтение extracted_data
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── protondb.py             # парсинг дампа
│   │   └── steam.py                # Steam API для имён игр
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # PipelineStep context manager
│   │   ├── cleaning.py
│   │   ├── enrichment/             # см. preprocessing/PLAN_ENRICHMENT.md
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── models.py
│   │   │   ├── merger.py
│   │   │   └── sources/
│   │   │       ├── steam.py        # Steam Store + Deck Verified
│   │   │       ├── protondb.py     # ProtonDB Summary API
│   │   │       ├── pcgamingwiki.py # PCGamingWiki Cargo API
│   │   │       └── anticheat.py    # AreWeAntiCheatYet
│   │   ├── llm/
│   │   │   ├── client.py           # OpenAI-compatible клиент
│   │   │   └── prompts/
│   │   ├── normalize/
│   │   │   ├── gpu.py
│   │   │   ├── cpu.py
│   │   │   └── launch_options.py
│   │   ├── extract/
│   │   │   ├── spotter.py          # regex pre-extraction
│   │   │   ├── extractor.py        # LLM extraction
│   │   │   ├── validator.py        # post-validation
│   │   │   ├── filter.py           # фильтрация отчётов для extraction
│   │   │   └── models.py           # pydantic: Action, Observation
│   │   └── store.py                # UPSERT helpers для SQLite
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features/
│   │   │   ├── hardware.py         # hardware features из gpu/cpu_normalization
│   │   │   ├── game.py             # game-level features из game_metadata
│   │   │   ├── embeddings.py       # SVD embeddings (GPU/CPU/Game)
│   │   │   └── encoding.py         # categorical encoding, gpu_tier маппинг
│   │   ├── models/
│   │   │   └── classifier.py       # LightGBM train/predict
│   │   ├── train.py                # CLI: обучение
│   │   ├── evaluate.py             # метрики, SHAP, валидация
│   │   └── export.py               # model.pkl + embeddings.npz
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── engine.py               # основной orchestrator
│   │   ├── predictor.py            # LightGBM prediction + SHAP (нативно!)
│   │   ├── similarity.py           # embeddings → cosine similarity → top-N
│   │   ├── aggregator.py           # settings aggregation (7 шагов)
│   │   ├── proton.py               # Proton version selection + regression detection
│   │   ├── issues.py               # observations → known_issues
│   │   └── compose.py              # merge actions → launch_options string, env_variables
│   └── api/
│       ├── __init__.py
│       ├── app.py                  # FastAPI app factory
│       ├── routes/
│       │   ├── health.py
│       │   ├── games.py
│       │   └── recommendations.py
│       ├── models.py               # Pydantic request/response DTOs
│       └── cache.py                # LRU cache для recommendations
├── data/                           # .gitignore'd, runtime
│   ├── protondb.db
│   ├── model.pkl                   # trained LightGBM
│   └── embeddings.npz              # GPU/CPU/Game embeddings + family→index maps
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Фазы реализации

### Phase 0 — Разведка (до начала кода)
- [x] Скачать последний дамп с `github.com/bdefore/protondb-data`
- [x] Проверить реальную структуру JSON (поля, типы, edge cases)
- [x] Прогнать записи — понять качество данных
- [x] Проверить Steam Store API на нескольких appId
- [x] Проверить PCGamingWiki Cargo API — реальные ответы
- [x] Задокументировать расхождения со схемой в плане

**Находки Phase 0 (348,683 отчётов, 30,968 app_ids):**
- `systemInfo.gpu` — 35K уникальных значений, часто мусор (OS name вместо GPU)
- `responses.protonVersion` — "Default", пустые строки, строки с \n
- `app.title` — ведущие пробелы, Unicode, не-английский текст
- Обнаружены rich structured поля: `followUp.*`, `customizationsUsed.*`, `launchFlagsUsed.*` — снижают потребность в LLM для этих аспектов
- Steam Deck поля: `batteryPerformance`, `readability`, `controlLayout`
- Multiplayer поля: `localMultiplayerAppraisal/Attempted/Played`, `onlineMultiplayer*`
- `launcher`: steam/lutris/bottles/gamehub/notListed

### Phase 1 — Foundation (Python)
- [ ] `pyproject.toml` — зависимости (fastapi, uvicorn, click, lightgbm, scikit-learn, pydantic, httpx, shap, scipy, rich, openai, python-dateutil, numpy)
- [ ] `protondb_settings/db/` — SQLite подключение, WAL mode, миграции (единственный source of truth для схемы)
- [ ] `protondb_settings/cli.py` — Click CLI (группы: serve, worker, preprocess, ml)
- [ ] `protondb_settings/api/app.py` — FastAPI app с health endpoint

### Phase 2 — Worker / Ingestion
- [ ] `protondb_settings/worker/` — CLI: `protondb-settings worker check|sync`
- [ ] Скачивание и парсинг ProtonDB дампа
- [ ] Batch insert отчётов (включая все fault-поля)
- [ ] Steam API клиент для кеша игр
- [ ] Запись `dump_release_tag`, `dump_sha256`, `dump_imported_at` в `meta`

### Phase 3 — Preprocessing
Preprocessing — **обязательный шаг** перед запуском API. Без него engine не работает.
- [ ] Enrichment: `game_metadata` из Steam/PCGamingWiki/AreWeAntiCheatYet (см. `preprocessing/PLAN_ENRICHMENT.md`)
- [ ] GPU/CPU normalization: LLM по уникальным строкам → `gpu_normalization`, `cpu_normalization`
- [ ] Launch options: LLM парсинг всех уникальных строк → `launch_options_parsed`
- [ ] Text extraction: LLM → `extracted_data` (actions, observations)
- Подробности — в `preprocessing/PLAN_LLM.md`

### Phase 4 — ML Training
ML модели — ядро recommendation engine. См. `PLAN_ML.md`.
- [ ] Hardware embeddings: SVD на матрице (GPU/CPU × Game → verdict) → feature vectors + similarity search
- [ ] Feature engineering: hardware + embeddings + game-level (из game_metadata) + aggregated reports
- [ ] LightGBM classifier: одна модель, предсказание (works_oob / needs_tinkering / borked)
- [ ] Export: `model.pkl`, `embeddings.npz`

### Phase 5 — API + Engine
- [ ] `protondb_settings/engine/` — predictor, similarity, aggregator, proton, issues, compose
- [ ] `GET /games/search`, `GET /games/{app_id}`
- [ ] `POST /recommendations` — ML-based predictions + settings aggregation + SHAP factors
- [ ] `GET /recommendations/{app_id}` — без hardware-фильтра
- [ ] LRU cache (key=`app_id:gpu_family:cpu_family`, TTL=1h)
- [ ] Загрузка ML моделей при старте (model.pkl + embeddings.npz — нативно)

### Phase 6 — Polish
- [ ] Input validation
- [ ] Dockerfile (Python + dependencies)
- [ ] README с примерами
- [ ] `docker-compose.yml` (server + optional llama-server)
- [ ] CI: lint (ruff), test (pytest)
