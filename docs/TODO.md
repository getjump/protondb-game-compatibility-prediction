# TODO — ProtonDB Recommended Settings

Задачи распилены из планов. Порядок = порядок зависимостей.

---

## Phase 0 — Разведка
- [x] Скачать последний дамп ProtonDB
- [x] Проверить реальную структуру JSON (поля, типы, edge cases)
- [x] Проанализировать качество данных → `preprocessing/ANALYSES.md`
- [x] Задокументировать расхождения со схемой
- [x] Проверить Steam Store API на нескольких appId
- [x] Проверить PCGamingWiki Cargo API — реальные ответы

---

## Phase 1 — Foundation (Python) ✓
- [x] `pyproject.toml` — зависимости (fastapi, uvicorn, click, lightgbm, scikit-learn, pydantic, httpx, shap, scipy, rich, openai, python-dateutil, numpy)
- [x] `protondb_settings/config.py` — конфиг (пути к БД, порт, rate limits)
- [x] `protondb_settings/db/connection.py` — подключение к SQLite, WAL mode, PRAGMA
- [x] `protondb_settings/db/migrations.py` — создание всех таблиц из схемы PLAN.md
  - [x] `meta`, `pipeline_runs`, `games`, `reports` (~90 колонок)
  - [x] `game_metadata`, `gpu_normalization`, `cpu_normalization`
  - [x] `launch_options_parsed`, `extracted_data`
  - [x] Все индексы
- [x] `protondb_settings/cli.py` — Click CLI (группы: serve, worker, preprocess, ml)
- [x] `protondb_settings/api/app.py` — FastAPI app factory
- [x] `protondb_settings/api/routes/health.py` — `GET /health`

---

## Phase 2 — Worker / Ingestion (Python) ✓
- [x] `protondb_settings/worker/` — CLI: `protondb-settings worker check|sync`
  - [x] Subcommands: `check`, `sync`
- [x] `protondb-settings worker check` — быстрая проверка обновлений
  - [x] GitHub API: `GET /repos/bdefore/protondb-data/releases/latest`
  - [x] Сравнение `tag_name` с `meta.dump_release_tag`
  - [x] Exit code: 0 = есть обновление, 1 = актуально
- [x] `protondb-settings worker sync` — импорт дампа
  - [x] Вызов check → skip если актуально (если нет `--force`)
  - [x] Скачивание `.tar.gz`, SHA-256 хеш → проверка `meta.dump_sha256`
  - [x] Парсинг JSONL дампа
  - [x] camelCase → snake_case маппинг (см. комментарий в PLAN.md schema)
  - [x] Все поля сохраняются as-is (без очистки/трансформации)
  - [x] Сериализация `followUp.*` в JSON-строки
  - [x] UPSERT в `reports` и `games` (транзакция каждые 5000 записей)
  - [x] Запись `dump_release_tag`, `dump_sha256`, `dump_imported_at` в `meta`
- [x] `protondb_settings/worker/steam.py` — Steam API клиент для `games` (name resolution)
  - [x] Rate limiting (1 req/sec)
  - [x] Кеширование в таблицу `games`
- [ ] Тестирование: импорт дампа из `github.com/bdefore/protondb-data` → проверка counts, полей, UPSERT idempotency

---

## Phase 3 — Preprocessing (Python) ✓

Обязательный шаг. Без preprocessing engine не работает.
Все шаги **автоматически resumable** — обрабатывают только необработанные данные.

### 3a — Pipeline Infrastructure ✓
- [x] `protondb_settings/preprocessing/pipeline.py` — `PipelineStep` context manager
  - [x] Rich progress bar: step name, bar, N/M, ETA
  - [x] Запись в `pipeline_runs`: начало/завершение/ошибка
  - [x] Обнаружение прерванных runs при старте → resume
  - [x] Обновление `processed` counter + progress bar каждый batch commit
  - [x] `advance(n)` — обновляет и progress bar и pipeline_runs
- [x] `protondb-settings preprocess check`
  - [x] Статус всех шагов (pending items, last run, прерванные)
  - [x] Проверка ProtonDB dump через GitHub API (1 запрос)
  - [x] Проверка AreWeAntiCheatYet через HTTP HEAD + ETag
  - [x] Stale check для enrichment (enriched_at > 30 дней)
- [x] `protondb-settings preprocess run`
  - [x] Запуск всех шагов последовательно, с auto-resume
  - [x] `--force <step>` — перезапуск конкретного шага с нуля
  - [x] `--step <step>` — запуск только конкретного шага

- [x] `protondb_settings/preprocessing/store.py` — UPSERT helpers для SQLite

### 3b — Data Cleaning ✓
- [x] `protondb_settings/preprocessing/cleaning.py` — очистка сырых данных в `reports`
  - [x] Implicit checkpoint: `WHERE ram IS NOT NULL AND ram_mb IS NULL`
  - [x] `ram` → `ram_mb` (regex `\d+`, NULL если мусор)
  - [x] `proton_version`: trim, `"Default"|""` → NULL
  - [x] `kernel`: regex `(\d+\.\d+[\.\d]*)`, нет match → NULL
  - [x] Batch commits каждые 500 записей
  - [x] Результаты пишутся в доп. колонки (`ram_mb`) или UPDATE in-place

### 3c — Enrichment ✓
- [x] `protondb_settings/preprocessing/enrichment/` — `__init__.py`, `models.py` (pydantic модели)
- [x] `protondb_settings/preprocessing/enrichment/sources/steam.py` — Steam Store API + Deck Verified
  - [x] Rate limiting ~200 req/5 min
  - [x] Парсинг Store: developer, publisher, genres, categories, release_date, has_linux_native
  - [x] Парсинг Deck: `ajaxgetdeckappcompatibilityreport` → deck_status, deck_tests_json
- [x] `protondb_settings/preprocessing/enrichment/sources/protondb.py` — ProtonDB Summary API
  - [x] Rate limiting ~1 req/sec
  - [x] Парсинг: tier, score, confidence, trending_tier
- [x] `protondb_settings/preprocessing/enrichment/sources/pcgamingwiki.py` — PCGamingWiki Cargo API
  - [x] Batch по 10 app_ids через OR в WHERE
  - [x] Парсинг: engine, graphics_apis, anticheat, DRM (из таблицы `Availability.Uses_DRM`)
- [x] `protondb_settings/preprocessing/enrichment/sources/anticheat.py` — AreWeAntiCheatYet
  - [x] Fetch `games.json` с GitHub, ETag caching в `meta.awacy_etag`
  - [x] HTTP HEAD для проверки обновлений без скачивания
  - [x] Парсинг: anticheats, status
- [x] `protondb_settings/preprocessing/enrichment/merger.py` — объединение данных из всех источников
- [x] `protondb_settings/preprocessing/enrichment/main.py` — логика с `--min-reports`, `--source`
  - [x] Implicit checkpoint: `WHERE app_id NOT IN (SELECT app_id FROM game_metadata)`
  - [x] UPSERT в `game_metadata`, batch commits каждые 100 app_ids
  - [x] `--force` — удалить game_metadata и начать сначала
  - [x] Stale refresh: `--refresh-older-than 30d` для переобогащения

### 3d — LLM Client ✓
- [x] `protondb_settings/preprocessing/llm/client.py` — единый OpenAI-compatible клиент
  - [x] Работает с любым провайдером: local (llama.cpp), OpenRouter, OpenAI, etc.
  - [x] Конфигурация: `--base-url`, `--model`, `--api-key` / env vars
  - [x] Retry с exponential backoff
  - [x] Concurrent requests (configurable)

### 3e — GPU/CPU Normalization (LLM) ✓
- [x] `protondb_settings/preprocessing/normalize/gpu.py` — GPU normalization pipeline
  - [x] Implicit checkpoint: `DISTINCT gpu FROM reports WHERE gpu NOT IN (SELECT raw_string FROM gpu_normalization)`
  - [x] Промпт: vendor, family, model, normalized_name, is_apu, is_igpu, is_virtual
  - [x] Batch стратегия: 20-50 строк/запрос (cloud), 1 (local)
  - [x] UPSERT → `gpu_normalization`, batch commits каждые 100
  - [x] `protondb_settings/preprocessing/llm/prompts/gpu_normalize.py` — промпт для GPU
- [x] `protondb_settings/preprocessing/normalize/cpu.py` — CPU normalization pipeline
  - [x] Аналогично GPU → `cpu_normalization`
  - [x] `protondb_settings/preprocessing/llm/prompts/cpu_normalize.py` — промпт для CPU

### 3f — Launch Options Parsing (LLM) ✓
- [x] `protondb_settings/preprocessing/normalize/launch_options.py` — все 16K уникальных строк через LLM
  - [x] Implicit checkpoint: `DISTINCT launch_options ... NOT IN (SELECT raw_string FROM launch_options_parsed)`
  - [x] Batch стратегия: 10-20 строк/запрос (cloud), 1-5 (local)
  - [x] UPSERT → `launch_options_parsed`, batch commits каждые 100
  - [x] `protondb_settings/preprocessing/llm/prompts/launch_parse.py` — промпт для launch options

### 3g — Text Extraction (LLM) ✓
- [x] `protondb_settings/preprocessing/extract/spotter.py` — Слой 1: regex pre-extraction
  - [x] Паттерны: env_var, proton_version, wine_version, wrapper_tool, game_arg, file_path, package, dll_override
  - [x] Результат передаётся в промпт как hints
- [x] `protondb_settings/preprocessing/extract/extractor.py` — Слой 2: LLM extraction
  - [x] Implicit checkpoint: `WHERE id NOT IN (SELECT report_id FROM extracted_data)` + text filter
  - [x] Промпт с game_metadata контекстом, structured fields, pre-extracted entities
  - [x] 13 action types + observations (см. PLAN_LLM.md)
  - [x] `reported_effect`: effective / ineffective / unclear
  - [x] `conditions`: gpu_vendor, symptom, display_server, distro, proton_version
  - [x] UPSERT → `extracted_data`, batch commits каждые 100
- [x] `protondb_settings/preprocessing/extract/validator.py` — Слой 3: post-validation
  - [x] Pydantic validation
  - [x] Risk override (sudo, /etc, rm -rf → force risky)
  - [x] Scope validation (file_patch пути)
  - [x] Sanitization (env_var values, path traversal)
- [x] `protondb_settings/preprocessing/extract/models.py` — pydantic: Action, Observation
- [x] `protondb_settings/preprocessing/extract/filter.py` — фильтрация отчётов для extraction
  - [x] Skip: verdict_oob=yes + no faults + no text
  - [x] Skip: all notes < 10 chars
  - [x] Skip: already in extracted_data (implicit checkpoint)
- [x] `protondb_settings/preprocessing/llm/prompts/text_extract.py` — формирование промпта
- [ ] GBNF грамматика для local LLM (см. PLAN_LLM.md appendix)

### 3h — CLI Entry Points (Click) ✓
- [x] `protondb-settings preprocess llm` — CLI для LLM задач
  - [x] Subcommands: `normalize-gpu`, `normalize-cpu`, `parse-launch-options`, `extract`, `all`
  - [x] `--base-url`, `--model`, `--api-key`
  - [x] `--force` — перезапуск с нуля
- [x] `protondb-settings preprocess run` — запуск всего pipeline последовательно
  - [x] Общий progress (N/6 steps) + текущий шаг с progress bar
  - [x] Галочки для завершённых шагов, спиннер для текущего
- [x] `protondb-settings preprocess check` — статус всех шагов + проверка обновлений

### 3i — Validation
- [ ] Gold set: разметить вручную 100 отчётов для валидации
- [ ] Прогон local vs cloud, сравнение quality

---

## Phase 4 — ML Training (Python) ✓

ML — ядро recommendation engine. См. `PLAN_ML.md`.
Модель обучена: Accuracy 0.72, F1 macro 0.50 (без enrichment данных; ожидается ~0.80 после enrichment).

### 4a — Embeddings (feature engineering) ✓
- [x] `protondb_settings/ml/features/embeddings.py` — SVD на матрице (GPU_family × Game → verdict)
  - [x] Left singular vectors → GPU embeddings
  - [x] Right singular vectors → Game embeddings — бесплатно из того же SVD
  - [x] Отдельный SVD для CPU → CPU embeddings
  - [x] Размерность: автоподбор по explained variance (90%, clip 16..64)
  - [x] vendor=unknown → исключаются из матрицы
  - [x] Новая игра без embedding → fallback на game_metadata features

### 4b — Feature Engineering ✓
- [x] `protondb_settings/ml/features/hardware.py` — hardware features из `gpu/cpu_normalization`
- [x] `protondb_settings/ml/features/game.py` — game-level features из `game_metadata`
- [x] `protondb_settings/ml/features/encoding.py` — categorical encoding, label maps
- [x] Embedding vectors (GPU + CPU + Game) как числовые features

### 4c — LightGBM Classification ✓
- [x] `protondb_settings/ml/models/classifier.py` — LightGBM train/predict (одна модель)
- [x] Time-based train/test split
- [x] Target: verdict_oob/verdict → borked/needs_tinkering/works_oob
- [x] `protondb_settings/ml/evaluate.py` — accuracy, F1, confusion matrix
  - [x] SHAP TreeExplainer — top-K features per prediction для API response `factors[]`
  - [ ] Валидация vs ProtonDB tiers: agreement rate для игр с 50+ отчётов
  - [ ] Валидация vs Steam Deck Verified: agreement rate для deck predictions

### 4d — Export & CLI ✓
- [x] `protondb_settings/ml/export.py` — export: `model.pkl` + `embeddings.npz`
- [x] `protondb_settings/ml/train.py` — CLI: `protondb-settings ml train`
  - [x] Регистрация в `pipeline_runs` (step=`ml_train`) с progress bar
- [x] `protondb_settings/ml/evaluate.py` — CLI: `protondb-settings ml evaluate`
  - [x] Accuracy, F1, confusion matrix, SHAP

---

## Phase 5 — API + ML Engine (Python / FastAPI)

### 5a — ML Integration
- [ ] `protondb_settings/engine/engine.py` — основной orchestrator (ML-based)
- [ ] `protondb_settings/engine/predictor.py` — загрузка `model.pkl` + `embeddings.npz`, prediction + SHAP (нативно!)
- [ ] `protondb_settings/engine/similarity.py` — загрузка `embeddings.npz` → cosine similarity → top-N похожих отчётов
  - [ ] `vendor=unknown` → исключить из similarity search
- [ ] `protondb_settings/db/extracted.py` — чтение `extracted_data`, `launch_options_parsed`, structured fields
- [ ] `protondb_settings/engine/aggregator.py` — Settings Aggregation (см. PLAN.md):
  - [ ] Merge трёх источников: extracted_data.actions + launch_options_parsed + structured fields (launchFlagsUsed/customizationsUsed → action format)
  - [ ] GROUP по canonical key `(type, normalized_value)` — нормализация env vars, versions, args
  - [ ] SCORE: `effectiveness × confidence × recency_boost`
  - [ ] FILTER: risk=safe, effectiveness>0.6, effective_count>=2, conditions match hardware
  - [ ] CONFLICT RESOLUTION: взаимоисключающие actions → highest score wins
  - [ ] RANK: сортировка по score desc, top-K actions (K=10)
- [ ] `protondb_settings/engine/compose.py` — COMPOSE: merge actions → launch_options string, env_variables
- [ ] `protondb_settings/engine/proton.py` — Proton Version Selection (см. PLAN.md):
  - [ ] Нормализация версий → (family, semver): official/experimental/ge/native
  - [ ] Группировка (version, verdict) по family
  - [ ] Regression detection: effective только на старых + broken на новых → pin к конкретной версии
  - [ ] Если effective разбросаны включая новые → рекомендовать latest
  - [ ] Выбор между families: official > experimental > ge (при прочих равных)
  - [ ] Ответ: recommended, pinned, alternatives, avoid
- [ ] `protondb_settings/engine/issues.py` — observations → known_issues (GROUP по symptom, частота, hardware filter)
- [ ] Инвалидация кеша при обновлении данных

### 5b — API Endpoints
- [ ] `protondb_settings/api/models.py` — Pydantic request/response DTOs (включая prediction)
- [ ] `protondb_settings/api/routes/games.py`
  - [ ] `GET /games/search?q={name}` — поиск по имени
  - [ ] `GET /games/{app_id}` — инфо + `game_metadata`
- [ ] `protondb_settings/api/routes/recommendations.py`
  - [ ] `POST /recommendations` — ML prediction + settings aggregation
  - [ ] `GET /recommendations/{app_id}` — без hardware-фильтра
  - [ ] Параметр `device`: `desktop` (default) / `steam_deck`
  - [ ] `prediction.factors[]` — SHAP top-3 features с impact
  - [ ] `deck{}` section в ответе (valve_status, battery_ok_pct, readable_pct, recommended_layout)
  - [ ] Включать `prediction`, `game_metadata` в ответ
- [ ] `protondb_settings/api/cache.py` — LRU cache (key=`app_id:gpu_family:cpu_family`, TTL=1h)
- [ ] Input validation (Pydantic)
- [ ] Error handling, proper HTTP status codes

---

## Phase 6 — Polish
- [ ] Input validation (app_id, hardware fields) — Pydantic
- [ ] Dockerfile (Python + dependencies)
- [ ] README с примерами запросов
- [ ] `docker-compose.yml` (server + optional llama-server)
- [ ] CI: lint (ruff), test (pytest)
- [ ] Удалить `preprocessing/LLM.md.archived`

---

## Operational — Запуск preprocessing (после реализации)
- [ ] Создать venv: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- [ ] Запустить enrichment: `protondb-settings preprocess run --step enrichment --min-reports 10`
- [ ] Запустить LLM preprocessing: `protondb-settings preprocess llm all --model ...`
- [ ] Переобучить ML после enrichment: `protondb-settings ml train`
