# LLM Preprocessing — нормализация и экстракция

Все задачи, где используется LLM в preprocessing pipeline. Запускается **после enrichment** (см. `PLAN_ENRICHMENT.md`).

## Обзор задач

| Задача | Вход | Уникальных | Выход | Приоритет |
|---|---|---|---|---|
| **GPU normalization** | `reports.gpu` | ~35K строк | `gpu_normalization` table | P0 |
| **CPU normalization** | `reports.cpu` | ~29K строк | `cpu_normalization` table | P0 |
| **Launch options parsing** | `reports.launch_options` | ~16K строк | `launch_options_parsed` table | P1 |
| **Text extraction** | `concluding_notes`, `notes.*` | ~100-200K отчётов | `extracted_data` table | P1 |

**Ключевая оптимизация**: GPU, CPU и launch_options обрабатываются по **уникальным значениям**, а не per-report. Это 35K+16K строк вместо 350K отчётов.

---

## Задача 1: GPU normalization

### Вход

Уникальные значения `reports.gpu` (~35K строк). Примеры:

```
"NVIDIA GeForce RTX 3070"           → стандартная дискретная
"nouveau NVC8"                       → open-source NVIDIA driver
"AMD Custom GPU 0405"                → кастомный APU (Steam Deck)
"Intel HD Graphics 630"              → iGPU
"VMware, Inc. llvmpipe"              → виртуальный GPU
"Arch Linux (64 bit)"                → мусор (OS в поле GPU)
```

### Промпт

```
Normalize this GPU string from a Linux gaming compatibility report.
Return JSON only.

Input: "{raw_gpu_string}"

{
  "vendor": "nvidia|amd|intel|unknown",
  "family": "string",       // e.g. "rtx30", "rdna2", "hd600", "custom_apu", "unknown"
  "model": "string",        // e.g. "rtx3070", "rx6800xt", "steam_deck_apu", "unknown"
  "normalized_name": "string", // human-readable: "NVIDIA RTX 3070", "AMD Custom APU (Steam Deck)"
  "is_apu": true/false,
  "is_igpu": true/false,
  "is_virtual": true/false   // llvmpipe, virgl, vmware
}

Rules:
- "nouveau NVxx" = NVIDIA (open-source driver). Parse chip code to family.
- "llvmpipe", "virgl", "VMware" = virtual GPU, vendor from context or "unknown".
- If the string is clearly not a GPU (OS name, random text) → all fields "unknown".
- Preserve all info. Never discard unusual hardware (APUs, embedded, server GPUs).
```

### Batch стратегия

- Группируем по 20-50 строк в один запрос (для cloud)
- Для local: по 1 строке (маленькие модели лучше с одним)
- ~35K строк × ~50 tokens/ответ = ~1.75M output tokens
- **Cloud (Haiku)**: ~$0.50, ~30 минут
- **Local (7B)**: ~2-3 часа

### Выход

Таблица `gpu_normalization` — см. `PLAN.md`.

---

## Задача 2: CPU normalization

Аналогично GPU. ~546 мусорных из ~348K. Уникальных строк значительно меньше.

### Промпт

```
Normalize this CPU string from a Linux gaming report.
Return JSON only.

Input: "{raw_cpu_string}"

{
  "vendor": "intel|amd|unknown",
  "family": "string",       // "zen3", "alder_lake", "custom_apu", "unknown"
  "model": "string",        // "ryzen7_5800x", "i7_12700k", "unknown"
  "normalized_name": "string",
  "generation": null|int,   // Intel: 12, 13, 14. AMD Zen: 3, 4, 5.
  "is_apu": true/false
}

Rules:
- "VirtualApple" → vendor="unknown" (virtual/emulated).
- Random garbage ("0x0", "Spicy Silicon") → all "unknown".
- Custom APUs (Steam Deck, ROG Ally) → is_apu=true, appropriate family.
```

### Оценка

- **Cloud**: ~$0.10-0.20
- **Local**: ~30 минут-1 час

---

## Задача 3: Launch options parsing (LLM)

### Вход

Все уникальные значения `reports.launch_options` (~16K строк, 48K отчётов).

Примеры:
```
"gamemoderun %command%"                                    → wrapper
"PROTON_USE_WINED3D=1 %command%"                           → env var
"DXVK_ASYNC=1 MANGOHUD=1 %command% -windowed"             → env vars + arg
"gamescope -W 1920 -H 1080 -f -- %command%"                → wrapper с параметрами
"PROTON_ENABLE_NVAPI=1 gamescope -r 60 -- %command% -dx11" → всё вместе
"SteamDeck=0 %command% -skipintro -nointro"                → env + game args
```

### Подход: LLM для всех строк

Все 16K уникальных строк обрабатываются LLM. Это обеспечивает:
- Единообразный парсинг (нет расхождений regex vs LLM)
- Корректную обработку edge cases (нестандартные форматы, опечатки, нет %command%)
- Семантическое понимание (gamescope флаги, неизвестные wrappers)

### Batch стратегия

- Группируем по 10-20 строк в один запрос (для cloud)
- Для local: по 1-5 строк
- ~16K строк × ~100 tokens/ответ = ~1.6M output tokens
- **Cloud (Haiku)**: ~$0.40, ~15 минут
- **Local (7B)**: ~1-2 часа

### Промпт

```
Parse these Steam launch options strings into structured components.
Return JSON array — one result per input.

Inputs:
1. "{launch_options_string_1}"
2. "{launch_options_string_2}"
...

For each input, return:
{
  "env_vars": [{"name": "KEY", "value": "VALUE"}],
  "wrappers": [{"tool": "gamescope|mangohud|gamemoderun|prime-run|other", "args": "-W 1920 -H 1080 -f"}],
  "game_args": ["-dx11", "-skipintro"],
  "unparsed": "anything that doesn't fit above"
}

Rules:
- Everything before %command% that matches KEY=VALUE is an env var.
- Known wrapper tools: gamescope, mangohud, gamemoderun, prime-run, taskset, obs-gamecapture.
- Everything after %command% (or after -- for gamescope) is a game argument.
- If no %command%, infer structure from known patterns.
- Wrappers have their own args (gamescope -W 1920 -H 1080 -f) — separate from game_args.
```

### Выход

Таблица `launch_options_parsed` — см. `PLAN.md` (единый source of truth для схемы).

### Оценка

- **Cloud (Haiku)**: ~$0.40, ~15 мин
- **Local (7B)**: ~1-2 часа

---

## Задача 4: Text extraction из отчётов

Самая объёмная задача. Извлекает структурированные действия из свободного текста.

### Вход

Отчёты, у которых есть полезный текст:
- `concluding_notes` — 32.2%, ср. длина 174 chars
- `notes.extra` — 10.0%, ср. длина 196 chars
- `notes.customizationsUsed` — 7.1%, ср. длина 149 chars
- `notes.verdict` — 74.3%, ср. длина 54 chars (короткие, но массовые)
- `notes.{fault}` — 3-10%, ср. длина 86-131 chars

После фильтрации: ~100-200K отчётов.

### Архитектура: трёхслойный pipeline

```
Отчёт (текстовые поля)
    │
    ▼
[Слой 1: Детерминированный споттинг]
    │   regex: env vars, versions, paths, tools, packages
    │   → pre-extracted entities передаются в промпт как подсказка
    ▼
[Слой 2: LLM extraction]
    │   → actions[], observations[], reported_effect
    ▼
[Слой 3: Post-validation]
    │   → risk classification, scope validation, dedup
    ▼
extracted_data table
```

### Слой 1: Детерминированный споттинг

Regex-парсер выделяет из текста сущности **до** LLM-вызова. Результат передаётся в промпт как hints, снижая нагрузку на LLM:

```python
PATTERNS = {
    "env_var":        r'\b([A-Z][A-Z0-9_]{2,})=(\S+)',
    "proton_version": r'(?:GE-)?Proton[\s-]*[\d.]+(?:-GE-\d+)?|Proton\s+Experimental',
    "wine_version":   r'Wine[\s-]*\d+\.\d+',
    "wrapper_tool":   r'\b(gamescope|mangohud|gamemoderun|prime-run|protontricks|winetricks)\b',
    "game_arg":       r'(?<=\s)-(?:dx\d+|vulkan|windowed|fullscreen|skipintro|nointro|nobattleye|force-[\w]+)\b',
    "file_path":      r'[~/][\w./\\-]+\.(?:ini|cfg|conf|json|xml|dll|exe|so|reg)',
    "package":        r'\b(?:vcrun\d+|dotnet\d+|d3dcompiler_\d+|dxvk|vkd3d|mf|faudio)\b',
    "dll_override":   r'\b\w+\.dll\b',
}
```

### Слой 2: LLM extraction

#### Таксономия action types

Набор action types:

| Action type | Описание | Автоматизируемость | Пример |
|---|---|---|---|
| `env_var` | Переменная окружения | Полная | `PROTON_ENABLE_NVAPI=1` |
| `game_arg` | Аргумент exe игры | Полная | `-dx11`, `-skipintro` |
| `wrapper_config` | gamescope/mangohud/gamemoderun | Полная | `gamescope -W 1920 -H 1080` |
| `runner_selection` | Версия Proton/Wine | Полная | `GE-Proton9-25` |
| `protontricks_verb` | protontricks/winetricks | Полная | `vcrun2019`, `dotnet48` |
| `dll_override` | WINEDLLOVERRIDES | Полная | `d3d9=n,b` |
| `prefix_action` | Действие над wine prefix | Частичная | "delete prefix", "clear shader cache" |
| `file_patch` | Изменение конфигов | Частичная | "set fullscreen=false in .ini" |
| `registry_patch` | Wine regedit | Частичная | "add key to HKCU" |
| `executable_override` | Запуск другого exe | Частичная | "launch Win64/game.exe instead of launcher" |
| `dependency_install` | Системный пакет | Нет (sudo) | `lib32-vulkan-radeon` |
| `session_requirement` | Wayland/X11/compositor | Нет | "works only on X11" |
| `system_tweak` | sysctl/grub/kernel | Нет (sudo, risky) | "increase vm.max_map_count" |
| `observation` | Факт без действия | — | "crashes after splash screen" |

#### Промпт

```
Extract actionable Linux gaming compatibility information from this ProtonDB report.

Game: {title}
Engine: {engine} | Graphics API: {graphics_apis} | Anti-cheat: {anticheat}
Hardware: {gpu}, {cpu}, {os}, kernel {kernel}
Proton: {proton_version} ({variant}) | Custom: {custom_proton_version}
Launcher: {launcher}
Structured data already known:
  - Customizations: {active_customizations}
  - Launch flags: {active_launch_flags}
  - Faults: {fault_summary}
Detected entities (regex): {pre_extracted_entities}

User text:
---
{combined_text}
---

Return JSON only:
{
  "actions": [
    {
      "type": "env_var|game_arg|wrapper_config|runner_selection|protontricks_verb|dll_override|prefix_action|file_patch|registry_patch|executable_override|dependency_install|session_requirement|system_tweak",
      "value": "exact value from text",
      "detail": "additional context if needed",
      "reported_effect": "effective|ineffective|unclear",
      "conditions": [{"kind": "gpu_vendor|symptom|display_server|distro|proton_version", "value": "..."}],
      "risk": "safe|risky"
    }
  ],
  "observations": [
    {
      "symptom": "crash_on_launch|black_screen|stutter|no_audio|controller_issue|launcher_crash|anti_cheat_fail|other",
      "description": "short text",
      "hardware_specific": true/false
    }
  ],
  "useful": true/false
}

Rules:
- Extract ONLY what is explicitly stated. Never infer actions not mentioned.
- Distinguish "X helped" (effective) from "X didn't help" (ineffective) — CRITICAL.
- Don't duplicate what's already in "Structured data already known".
- env_var: exact KEY=VALUE. Common: PROTON_ENABLE_NVAPI, PROTON_USE_WINED3D, DXVK_ASYNC, VKD3D_CONFIG, WINE_FULLSCREEN_FSR, PROTON_NO_ESYNC/FSYNC.
- game_arg: flags for game exe (-dx11, -vulkan, -windowed, -skipintro), NOT env vars.
- wrapper_config: full gamescope/mangohud command with flags.
- runner_selection: exact version string (GE-Proton9-25, Proton Experimental).
- protontricks_verb: exact verb name (vcrun2019, dotnet48, d3dcompiler_47).
- file_patch: include file path and what to change.
- risk=risky: anything requiring sudo, system-wide changes, /etc, GRUB, kernel params, disabling security.
- "works fine" / "no issues" with no details → useful=false, empty actions.
- Observations are gold — capture symptoms even without fix.
```

#### reported_effect

Критическое поле. Позволяет engine различать:
- `"Switching to GE-Proton fixed it"` → `effective`
- `"Tried GE-Proton but still crashes"` → `ineffective`
- `"Using GE-Proton"` (без оценки) → `unclear`

Без этого поля engine будет рекомендовать действия, которые пользователи явно пометили как нерабочие.

### Слой 3: Post-validation

Детерминированная проверка после LLM:

1. **Формат**: pydantic validation (JSON structure, enum values)
2. **Risk override**: если action содержит паттерны → force `risk=risky`:
   - Пути: `/etc/`, `/boot/`, `~/.ssh/`, `/usr/lib/`
   - Команды: `sudo`, `rm -rf`, `curl|wget ... | bash`, `chmod -R 777`
   - Scope: `sysctl`, `grub`, `modprobe`, `udev`
3. **Dedup**: одинаковые action для одного app_id из разных отчётов → merge с подсчётом частоты
4. **Scope validation**: `file_patch` путь должен быть внутри game dir или `~/.steam/`; иначе → `risk=risky`
5. **Sanitization**: значения env_var должны быть `[A-Za-z0-9_=,.\-/]+`; пути не содержат `..`

### Выход

Таблица `extracted_data` — см. `PLAN.md` (единый source of truth для схемы).

---

## Backend: единый OpenAI-compatible клиент

Оба backend (local и cloud) используют **OpenAI-compatible API**. Один клиент, разные `base_url` и `model`.

### Конфигурация

```python
# Local (llama.cpp)
OPENAI_BASE_URL=http://localhost:8090/v1
OPENAI_API_KEY=not-needed
MODEL=qwen2.5-7b-instruct

# Cloud (OpenRouter — любая модель)
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
MODEL=anthropic/claude-haiku  # или google/gemini-flash, meta-llama/llama-3.1-8b, etc.

# Cloud (прямой провайдер)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
```

### Local LLM

**Qwen2.5-7B-Instruct** (Q4_K_M) — рекомендуемая модель для local inference.

| Модель | VRAM | Скорость* | Качество JSON |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** | ~6 GB | ~50 tok/s | Отличное |
| Llama-3.1-8B-Instruct | ~6 GB | ~45 tok/s | Хорошее |
| Qwen2.5-3B-Instruct | ~3 GB | ~90 tok/s | Среднее |

\* Для RTX 3070/4070 уровня.

```bash
llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --port 8090 --ctx-size 2048 --n-gpu-layers 99 --parallel 4
```

JSON grammar mode (GBNF) гарантирует валидный JSON — см. приложение.

- **1 отчёт на запрос** (маленькие модели хуже с batch)
- `ThreadPoolExecutor(max_workers=4)` + `--parallel 4` на llama-server

### Cloud (OpenRouter / любой провайдер)

Любая модель через OpenAI-compatible API. Рекомендации:
- **Claude Haiku** — дешёвый, отличное качество JSON extraction
- **GPT-4o-mini** — альтернатива
- **Gemini Flash** — самый дешёвый вариант

Batching: 10-20 отчётов в одном промпте (экономия токенов).
Параллельность: 10-20 concurrent requests с exponential backoff.

---

## Оценка стоимости и времени

| Задача | Объём | Local (7B, 1 GPU) | Cloud (Haiku) |
|---|---|---|---|
| GPU normalization | ~35K уникальных | ~2-3 часа | ~$0.50, ~30 мин |
| CPU normalization | ~29K уникальных | ~30 мин-1 час | ~$0.10-0.20 |
| Launch options | ~16K уникальных строк | ~1-2 часа | ~$0.40, ~15 мин |
| Text extraction | ~100-200K отчётов | **~24-48 часов** (50 tok/s, ~500+200 tok/req, parallel 4) | ~$10-15 |
| **Итого** | | **~28-54 часа** | **~$11-16** |

> **Примечание**: local estimate консервативный. При ~50 tok/s и ~700 tokens/request (вход+выход), 4 parallel slots дают ~1K req/час → 100-200K за 100-200 часов на 1 slot, ~25-50 часов с parallel 4. Cloud значительно быстрее (часы, не дни).

---

## Структура кода

```
protondb_settings/preprocessing/
├── llm/
│   ├── __init__.py
│   ├── client.py            # OpenAI-compatible клиент (local/cloud/OpenRouter)
│   ├── prompts/
│   │   ├── gpu_normalize.py
│   │   ├── cpu_normalize.py
│   │   ├── launch_parse.py
│   │   └── text_extract.py
│   └── grammar.gbnf         # GBNF для llama.cpp
├── normalize/
│   ├── __init__.py
│   ├── gpu.py               # GPU normalization pipeline
│   ├── cpu.py               # CPU normalization pipeline
│   └── launch_options.py    # LLM parsing всех уникальных строк
├── extract/
│   ├── __init__.py
│   ├── spotter.py           # Слой 1: regex pre-extraction
│   ├── extractor.py         # Слой 2: LLM extraction
│   ├── validator.py         # Слой 3: post-validation, risk override
│   ├── filter.py            # фильтрация отчётов для text extraction
│   └── models.py            # pydantic models (Action, Observation)
├── store.py                 # запись результатов в SQLite
└── pipeline.py              # PipelineStep: progress bar + pipeline_runs + resume
```

## Запуск

Единый OpenAI-compatible клиент. Провайдер задаётся через `--base-url`/`--model` или env vars.

```bash
# Cloud через OpenRouter (любая модель):
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=sk-or-... \
  protondb-settings preprocess llm all --model anthropic/claude-haiku

# Local (llama.cpp):
protondb-settings preprocess llm all \
  --base-url http://localhost:8090/v1 --model qwen2.5-7b-instruct

# Отдельные задачи:
protondb-settings preprocess llm normalize-gpu
protondb-settings preprocess llm normalize-cpu
protondb-settings preprocess llm parse-launch-options
protondb-settings preprocess llm extract

# Resume (автоматический — просто запустить снова):
protondb-settings preprocess llm extract
```

## Валидация качества

1. **Gold set**: размечаем вручную 100 отчётов (разнообразных: простые, сложные, мусор, edge cases)
2. **Сравнение моделей**: прогоняем gold set через разные модели (local 7B vs cloud), сравниваем
3. **Метрики**:
   - Precision/recall по action types
   - `reported_effect` accuracy (отличает effective от ineffective)
   - Risk classification accuracy (не пропускает risky как safe)
   - % false positives (выдуманные actions)
4. **Порог**: local < 85% от cloud quality → модель побольше или другой промпт
5. С grammar mode: 100% валидный JSON, проверяем только семантику

## Durability & инкрементальное обновление

Все шаги автоматически обрабатывают **только необработанные данные** (см. `preprocessing/PLAN.md` → Durability).

- GPU/CPU normalization: `DISTINCT gpu/cpu FROM reports WHERE ... NOT IN (SELECT raw_string FROM gpu/cpu_normalization)`
- Launch options: аналогично
- Text extraction: `WHERE id NOT IN (SELECT report_id FROM extracted_data)` + фильтр по тексту
- Batch commits каждые 100-500 элементов
- UPSERT (`INSERT OR REPLACE`) — idempotent
- `pipeline_runs` — трекинг прогресса, обнаружение прерванных runs
- `--force` — перезапуск с нуля

## Приложение: GBNF-грамматика

Для задачи text extraction (Задача 4). Обеспечивает валидный JSON от local LLM:

```gbnf
root ::= "{" ws
  "\"actions\":" ws actions "," ws
  "\"observations\":" ws observations "," ws
  "\"useful\":" ws boolean
  ws "}"

boolean ::= "true" | "false"
nullable-string ::= "null" | string

actions ::= "[]" | "[" ws action ("," ws action)* ws "]"
action ::= "{" ws
  "\"type\":" ws action-type "," ws
  "\"value\":" ws string "," ws
  "\"detail\":" ws nullable-string "," ws
  "\"reported_effect\":" ws effect "," ws
  "\"conditions\":" ws conditions "," ws
  "\"risk\":" ws risk
  ws "}"
action-type ::= "\"env_var\"" | "\"game_arg\"" | "\"wrapper_config\"" | "\"runner_selection\"" | "\"protontricks_verb\"" | "\"dll_override\"" | "\"prefix_action\"" | "\"file_patch\"" | "\"registry_patch\"" | "\"executable_override\"" | "\"dependency_install\"" | "\"session_requirement\"" | "\"system_tweak\""
effect ::= "\"effective\"" | "\"ineffective\"" | "\"unclear\""
risk ::= "\"safe\"" | "\"risky\""

conditions ::= "[]" | "[" ws condition ("," ws condition)* ws "]"
condition ::= "{" ws
  "\"kind\":" ws condition-kind "," ws
  "\"value\":" ws string
  ws "}"
condition-kind ::= "\"gpu_vendor\"" | "\"symptom\"" | "\"display_server\"" | "\"distro\"" | "\"proton_version\""

observations ::= "[]" | "[" ws observation ("," ws observation)* ws "]"
observation ::= "{" ws
  "\"symptom\":" ws symptom-type "," ws
  "\"description\":" ws string "," ws
  "\"hardware_specific\":" ws boolean
  ws "}"
symptom-type ::= "\"crash_on_launch\"" | "\"black_screen\"" | "\"stutter\"" | "\"no_audio\"" | "\"controller_issue\"" | "\"launcher_crash\"" | "\"anti_cheat_fail\"" | "\"other\""

string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
```
