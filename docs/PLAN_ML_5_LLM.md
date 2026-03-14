# ML Pipeline — фичи из LLM-извлечённых данных (фаза 5, часть LLM)

> Вынесено из [PLAN_ML_5.md](PLAN_ML_5.md) — фичи, зависящие от LLM-предобработки (`extracted_data`).

## Источники данных

### 1. Извлечённые действия (`extracted_data.actions_json`)

LLM уже извлёк структурированные actions из ~100-200K отчётов:

```json
{
  "type": "env_var",           // 13 типов
  "value": "PROTON_USE_WINED3D=1",
  "detail": "...",
  "reported_effect": "fixes black screen",
  "conditions": "nvidia only",
  "risk": "low"                // low/medium/high
}
```

**Типы действий** (13):
- Полностью автоматизируемые: `env_var`, `game_arg`, `wrapper_config`, `runner_selection`, `protontricks_verb`, `dll_override`
- Частично: `prefix_action`, `file_patch`, `registry_patch`, `executable_override`
- Не автоматизируемые: `dependency_install`, `session_requirement`, `system_tweak`

**Покрытие**: ~100-200K отчётов с extracted данными

**Текущее использование**: только в engine (подбор настроек), **НЕ в ML**.

### 2. Распарсенные launch options (`launch_options_parsed`)

LLM парсит ~16K уникальных строк launch options → структурированные данные:
- `env_vars_json`: переменные окружения
- `wrappers_json`: gamescope, mangohud, gamemode и т.д.
- `game_args_json`: аргументы игры

**Покрытие**: 14% отчётов (48K) имеют launch options

**Парсинг**: `protondb_settings/preprocessing/normalize/launch_options.py` — каждая уникальная строка отправляется в LLM, результат кэшируется в `launch_options_parsed`.

### 3. Наблюдения (`extracted_data.observations_json`)

```json
{
  "symptom": "crash",          // 8 типов симптомов
  "description": "crashes on startup",
  "hardware_specific": true
}
```

**Типы симптомов**: crash, black_screen, low_fps, audio_issue, input_issue, graphical_glitch, save_issue, network_issue

---

## Предлагаемые фичи

### Группа A: Агрегированные action-фичи (per-game)

Высокий ROI — данные уже извлечены, нужна только агрегация.

| Фича | Описание | Ожидаемый сигнал |
|------|----------|------------------|
| `action_count_per_game` | Среднее кол-во действий на отчёт для игры | Больше действий → скорее tinkering |
| `pct_automatable_actions` | Доля автоматизируемых действий | Высокая → tinkering (но решаемый) |
| `pct_non_automatable_actions` | Доля неавтоматизируемых | Высокая → сложный tinkering |
| `action_type_diversity` | Кол-во уникальных типов действий | Высокая → сложная настройка |
| `pct_reports_with_actions` | Доля отчётов с действиями для игры | Высокая → tinkering |
| `pct_effective_actions` | Доля действий с позитивным reported_effect | — |
| `avg_action_risk` | Средний risk level (low=0, med=0.5, high=1) | Высокий → сложный tinkering |

### Группа B: Launch options фичи (per-report)

Данные уже распарсены LLM в `launch_options_parsed`, нужен только JOIN.

| Фича | Описание | Ожидаемый сигнал |
|------|----------|------------------|
| `has_launch_options` | Есть ли launch options | Да → скорее tinkering |
| `env_var_count` | Кол-во переменных окружения | Больше → tinkering |
| `wrapper_count` | Кол-во wrappers | — |
| `has_gamescope` | Использует gamescope | — |
| `has_mangohud` | Использует mangohud | — |
| `has_gamemode` | Использует gamemode | — |
| `has_wine_prefix` | Кастомный wine prefix | → tinkering |

### Группа C: Symptom-фичи (per-game, агрегированные)

| Фича | Описание | Ожидаемый сигнал |
|------|----------|------------------|
| `symptom_count_per_report` | Среднее кол-во симптомов | Больше → ближе к borked |
| `pct_crash_reports` | Доля отчётов с crash | Высокая → borked |
| `pct_performance_reports` | Доля отчётов с low_fps | → tinkering (работает, но плохо) |
| `pct_graphical_reports` | Доля с графическими артефактами | — |
| `symptom_diversity` | Разнообразие симптомов | — |
| `pct_hw_specific_symptoms` | Доля HW-зависимых проблем | Высокая → зависит от железа |

---

## План экспериментов

### Эксперимент 1: Action-фичи (Группа A)

**Что делаем**:
1. SQL-агрегация из `extracted_data` по app_id
2. Добавить 7 фич в `_build_feature_matrix`
3. Обучить cascade, сравнить с baseline

**Сложность**: низкая (SQL + minor code changes)
**Ожидание**: +0.005–0.015 cascade F1

### Эксперимент 2: Launch options фичи (Группа B)

**Что делаем**:
1. Join `launch_options_parsed` в feature matrix
2. Добавить 7 фич (per-report)
3. Обучить cascade

**Сложность**: низкая (данные уже распарсены LLM)
**Ожидание**: +0.002–0.008 (покрытие всего 14%)

### Эксперимент 3: Symptom-фичи (Группа C)

**Что делаем**:
1. Парсить `observations_json` из `extracted_data`
2. Агрегировать по app_id
3. Добавить 6 фич

**Сложность**: средняя (нужен JSON parsing)
**Ожидание**: +0.005–0.010

---

## Приоритет

1. **Эксперимент 1 (actions)** — самый высокий ROI, данные уже структурированы
2. **Эксперимент 2 (launch options)** — данные уже распарсены, нужен только JOIN
3. **Эксперимент 3 (symptoms)** — симптомы = прямой сигнал о работоспособности
