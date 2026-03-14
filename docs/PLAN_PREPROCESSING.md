# Preprocessing Pipeline — обзор

Phase 3 в `PLAN.md`. **Обязательный шаг** между worker (Phase 2) и ML training (Phase 4). Без preprocessing engine не работает.

Preprocessing читает из `reports` в SQLite (заполненной worker'ом) и обогащает данные.

## Порядок шагов

```
reports table (SQLite, заполнена worker'ом as-is)
    │
    ▼
[1. Data cleaning] — очистка сырых данных (ram→ram_mb, proton_version trim, kernel regex)
    │
    ▼
[2. Enrichment] — game_metadata из внешних API (см. PLAN_ENRICHMENT.md)
    │
    ▼
[3. LLM preprocessing] — нормализация + экстракция (см. PLAN_LLM.md)
    │
    ├── [3a. GPU/CPU normalization] — LLM по уникальным строкам → gpu/cpu_normalization
    ├── [3b. Launch options parsing] — LLM по всем уникальным строкам → launch_options_parsed
    └── [3c. Text extraction] — свободный текст → extracted_data (actions, observations)
```

### LLM Backend

Единый OpenAI-compatible клиент. Работает с любым провайдером:
- **Local (llama.cpp)**: бесплатный, ~28-54 часа на 1 GPU
- **Cloud (OpenRouter)**: любая модель (Claude Haiku, GPT-4o-mini, Gemini Flash и т.д.), ~$11-16
- **Прямой провайдер**: OpenAI, Anthropic, Google и др.

Конфигурация через `--base-url`, `--model`, `--api-key` или env vars `OPENAI_BASE_URL`, `MODEL`.

### Запуск полного pipeline

```bash
# 0. Worker импортировал дамп (as-is, без очистки)
protondb-settings worker sync

# 1. Все шаги последовательно (auto-resume):
protondb-settings preprocess run

# 2. Или по шагам:
protondb-settings preprocess run --step cleaning
protondb-settings preprocess run --step enrichment --min-reports 10

# 3. LLM preprocessing (все задачи)
# Cloud через OpenRouter:
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=sk-or-... \
  protondb-settings preprocess llm all --model anthropic/claude-haiku

# Или local (llama.cpp):
protondb-settings preprocess llm all \
  --base-url http://localhost:8090/v1 --model qwen2.5-7b-instruct

# Отдельные LLM-задачи:
protondb-settings preprocess llm normalize-gpu
protondb-settings preprocess llm normalize-cpu
protondb-settings preprocess llm parse-launch-options
protondb-settings preprocess llm extract

# Resume (автоматический — просто запустить снова):
protondb-settings preprocess llm extract
```

---

## Data cleaning

Отдельный шаг preprocessing (не в worker). Worker сохраняет всё as-is. Подробный анализ — в `ANALYSES.md`.

| Поле | Проблема (из анализа 348K отчётов) | Метод | Обоснование |
|---|---|---|---|
| `systemInfo.gpu` | 35K уникальных. 99.1% валидных, 3.1K невалидных. Но есть валидные edge cases: nouveau, llvmpipe/virgl, кастомные APU | **LLM по уникальным строкам** → `gpu_normalization`. Мусор → `vendor=unknown` | Regex не покроет будущие GPU, nouveau, APU без ручных обновлений |
| `systemInfo.cpu` | 99.8% валидных, 546 мусорных | **LLM по уникальным строкам** → `cpu_normalization`. Мусор → `vendor=unknown` | Аналогично GPU |
| `systemInfo.ram` | 99.93% валидных, 231 мусорных | **Regex**: `(\d+)\s*[GgMm]?[Bb]?` → int. Не число → NULL | Формат стабилен |
| `systemInfo.kernel` | 99.8% валидных, 711 мусорных | **Regex**: `(\d+\.\d+[\.\d]*)`. Нет match → NULL | Формат ядра стабилен |
| `proton_version` | 78% "Default", 22% реальных | **Regex**: trim + `"Default"\|""` → NULL. Паттерн `(\d[\d.\-]+\d)` | Формат версий стабилен |
| `app.title` | Пробелы, Unicode | **Trim**. Steam API — source of truth | — |
| `steamRuntimeVersion` | lspci output | **Игнорируем** | — |

**Принципы**: ничего не отбрасываем (мусор → `unknown`/NULL); никаких хардкод-списков — regex по стабильным форматам или LLM для open-ended строк.

---

## Входные данные reports: два типа

### Уже структурированные поля (НЕ нужен LLM)

Engine использует напрямую:
- `customizationsUsed.*` — boolean: winetricks, protontricks, configChange, customProton и т.д. (11% отчётов)
- `launchFlagsUsed.*` — boolean: useWineD3d11, disableEsync, enableNvapi и т.д. (3.4% отчётов)
- `followUp.*` — детализация faults: конкретные sub-categories (32.1% отчётов)
- `installs`/`opens`/`startsPlay` — воронка запуска (91-100%)
- `duration` — сколько играл (24.5%)
- `variant`, `proton_version`, `custom_proton_version` — версии Proton

### Текстовые поля для LLM-экстракции

LLM извлекает то, что **не покрыто structured fields**:
- `concluding_notes` — 32.2%, ср. 174 chars
- `notes.*` — 24 sub-keys, покрытие 3-74%

Подробности — в `PLAN_LLM.md`.

---

## Структура кода

Preprocessing — часть единого Python-пакета `protondb_settings`. Все зависимости — в корневом `pyproject.toml`.

```
protondb_settings/preprocessing/
├── __init__.py
├── pipeline.py              # PipelineStep: progress bar (rich) + pipeline_runs + batch commits
├── cleaning.py              # data cleaning (ram→ram_mb, proton_version trim, etc.)
├── llm/                     # LLM client и промпты
│   ├── client.py            # OpenAI-compatible (local/cloud/OpenRouter)
│   └── prompts/
├── normalize/               # GPU, CPU, launch_options normalization
│   ├── gpu.py
│   ├── cpu.py
│   └── launch_options.py
├── extract/                 # text extraction pipeline
│   ├── spotter.py           # regex pre-extraction
│   ├── extractor.py         # LLM extraction
│   ├── validator.py         # post-validation, risk override
│   ├── filter.py            # фильтрация отчётов для extraction
│   └── models.py            # pydantic models
├── enrichment/              # см. PLAN_ENRICHMENT.md
│   ├── main.py
│   ├── sources/
│   │   ├── steam.py         # Steam Store + Deck Verified
│   │   ├── protondb.py      # ProtonDB Summary API
│   │   ├── pcgamingwiki.py
│   │   └── anticheat.py
│   ├── merger.py
│   └── models.py
└── store.py                 # UPSERT helpers
```

## Durability & Resume

Каждый шаг preprocessing **автоматически продолжает с места остановки**. Перезапуск с нуля — только по явному `--force`.

### Принципы

1. **Implicit checkpointing** — сами данные = checkpoint. Шаг определяет "что ещё не обработано" запросом к БД:
   - Data cleaning: `SELECT id FROM reports WHERE ram IS NOT NULL AND ram_mb IS NULL` (необработанные)
   - GPU normalization: `SELECT DISTINCT gpu FROM reports WHERE gpu NOT IN (SELECT raw_string FROM gpu_normalization)`
   - CPU normalization: аналогично
   - Launch options: `SELECT DISTINCT launch_options FROM reports WHERE launch_options IS NOT NULL AND launch_options NOT IN (SELECT raw_string FROM launch_options_parsed)`
   - Text extraction: `SELECT id FROM reports WHERE id NOT IN (SELECT report_id FROM extracted_data)` + фильтр по наличию текста
   - Enrichment: `SELECT app_id FROM games WHERE app_id NOT IN (SELECT app_id FROM game_metadata)`

2. **Batch commits** — транзакция коммитится каждые N элементов (100-500), не в конце. Прерывание теряет максимум один batch.

3. **UPSERT** — все записи используют `INSERT OR REPLACE` / `ON CONFLICT ... DO UPDATE`. Повторный запуск для уже обработанного элемента безопасен (idempotent).

4. **`pipeline_runs` table** — трекинг прогресса (см. `PLAN.md` schema):
   - При старте: `INSERT INTO pipeline_runs (step, status, total_items, dump_tag)`
   - Каждый batch: `UPDATE pipeline_runs SET processed = ? WHERE id = ?`
   - При завершении: `UPDATE pipeline_runs SET status='completed', finished_at=...`
   - При ошибке: `UPDATE pipeline_runs SET status='failed', error=...`
   - При старте проверяем: есть ли незавершённый run для этого step → resume (лог: "resuming from item N/M")

### Progress UI (rich)

Все шаги выводят live progress bar через **`rich`**:

```
GPU normalization ━━━━━━━━━━━━━━━╸━━━━━━━━━━  42% 14,748/35,114  0:12:30  ETA 0:17:15
```

Реализация через `preprocessing/pipeline.py` — обёртка `PipelineStep`:

```python
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

class PipelineStep:
    """Контекстный менеджер для шага pipeline с progress bar и pipeline_runs."""

    def __init__(self, db, step_name: str, total: int):
        self.db = db
        self.step_name = step_name
        self.total = total
        self.run_id = None

    def __enter__(self):
        # 1. Записать pipeline_run (или найти незавершённый)
        # 2. Создать rich Progress
        self.progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self.progress.start()
        self.task = self.progress.add_task(self.step_name, total=self.total)
        return self

    def advance(self, n: int = 1):
        """Вызывается после каждого обработанного элемента."""
        self.progress.update(self.task, advance=n)
        # pipeline_runs.processed обновляется каждый batch commit

    def __exit__(self, *exc):
        self.progress.stop()
        # Обновить pipeline_runs: completed или failed
```

Использование в каждом шаге:

```python
pending = get_pending_gpu_strings(db)
with PipelineStep(db, "GPU normalization", total=len(pending)) as step:
    for batch in chunked(pending, batch_size):
        results = llm_client.normalize_gpu(batch)
        upsert_gpu_normalization(db, results)
        db.commit()
        step.advance(len(batch))
```

При запуске `protondb-settings preprocess run` — **только** cleaning + enrichment (не требует LLM):

```
Pipeline ━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━  1/2 steps
  ✓ Data cleaning          348,683 / 348,683  done in 0:02:14
  ⠋ Enrichment              12,500 /  30,968  ETA 1:15:00
```

При запуске `protondb-settings preprocess llm all` — LLM шаги (требует `--model`):

```
LLM Pipeline ━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━  1/4 steps
  ⠋ GPU normalization       14,748 /  35,114  ETA 0:17:15
    CPU normalization            — /       —  pending
    Launch options               — /       —  pending
    Text extraction              — /       —  pending
```

5. **`--force`** — удаляет данные для этого шага и начинает сначала:
   - `--force cleaning` → `UPDATE reports SET ram_mb = NULL`
   - `--force normalize-gpu` → `DELETE FROM gpu_normalization`
   - `--force extract` → `DELETE FROM extracted_data`
   - etc.

### Проверка обновлений

**`protondb-settings preprocess check`** — показывает статус всех источников:

```
ProtonDB dump:        up to date (monthly_2025_02, imported 2025-02-15)
Data cleaning:        3,241 reports pending (of 348,683)
GPU normalization:    done (35,114 / 35,114)
CPU normalization:    done (28,903 / 28,903)
Launch options:       1,204 pending (of 16,822)
Text extraction:      running (145,230 / 198,000) — interrupted 2h ago
Enrichment:           2,341 games pending (of 30,968)
  AreWeAntiCheatYet:  stale (last fetch: 14 days ago)
  Steam Store API:    ok
  PCGamingWiki:       ok
```

Реализация:
- Для ProtonDB: `meta.dump_release_tag` vs GitHub API latest release
- Для AreWeAntiCheatYet: `meta.awacy_etag` + HTTP HEAD с `If-None-Match`
- Для enrichment API: нет "обновления" — только новые/непокрытые app_ids + stale check (`enriched_at` > 30 дней)
- Для каждого шага: `COUNT(*)` pending items из запросов выше + `pipeline_runs` для статуса

### Пример flow

```bash
# 1. Быстрая проверка — есть ли что обновлять?
protondb-settings worker check            # проверка дампа (1 HTTP запрос)
protondb-settings preprocess check        # проверка всех шагов

# 2. Если есть новый дамп:
protondb-settings worker sync             # скачает и импортирует

# 3. Запуск preprocessing — автоматически обработает только новое:
protondb-settings preprocess run          # все шаги последовательно, resume

# 4. Если прервалось — просто запустить снова:
protondb-settings preprocess run          # продолжит с места остановки

# 5. Перезапуск конкретного шага с нуля:
protondb-settings preprocess llm normalize-gpu --force
```

## Инкрементальное обновление

1. Worker импортирует новый дамп → новые записи в `reports`
2. Data cleaning обрабатывает новые записи (ram_mb IS NULL)
3. Enrichment обогащает новые app_ids
4. LLM: нормализует новые уникальные GPU/CPU/launch_options строки
5. LLM: экстрагирует из новых отчётов с текстом
6. ML: переобучение моделей
7. Server автоматически использует новые данные

## Связанные документы

- **`PLAN_LLM.md`** — детали LLM preprocessing: промпты, таксономия actions, GBNF грамматика, стоимость
- **`PLAN_ENRICHMENT.md`** — enrichment из внешних API (Steam, PCGamingWiki, AreWeAntiCheatYet)
- **`ANALYSES.md`** — полный анализ качества данных из дампа
- **`../PLAN.md`** — единый source of truth для схемы БД и архитектуры
- **`../PLAN_ML.md`** — ML pipeline (Phase 4, зависит от preprocessing данных)
