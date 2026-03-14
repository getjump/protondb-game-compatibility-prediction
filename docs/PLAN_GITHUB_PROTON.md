# GitHub Proton Issues — план enrichment

## Источник

[ValveSoftware/Proton/issues](https://github.com/ValveSoftware/Proton/issues) — баг-репорты и compatibility reports от пользователей.

## Текущее состояние

- **9035 issues** (open + closed) через `gh issue list --json --state all`
- **4579 уникальных игр** (app_id извлечён из title/body)
- **92% overlap** с нашим датасетом (4223 из 4579)
- **13.6% покрытие** наших 31K игр — разреженный но целевой сигнал

### Распределение по резолюции

| State | Count |
|-------|-------|
| open | 4238 |
| closed/completed | 1359 |
| closed/not_planned | 535 |
| closed/duplicate | 266 |

### Лейблы (значимые)

| Label | Смысл |
|-------|-------|
| `Regression` | Было рабочее, сломалось (409 игр) |
| `Game compatibility` | Официальный баг-репорт |
| `Game compatibility - Unofficial` | Community-репорт |
| `.NET` / `XAudio2` | Подсистема-виновник |
| `NVIDIA drivers` / `Mesa drivers` / `AMD RADV` | Драйвер-специфичная проблема |
| `Feature Request` | Не баг |

---

## План

### 1. Keyword-based severity classification

Issues сильно разнятся по серьёзности. Классификация по ключевым словам в title + body (первые 500 символов):

| Severity | Score | Ключевые слова / условия |
|----------|-------|--------------------------|
| critical | 3 | `crash`, `won't launch`, `doesn't start`, `black screen`, `segfault`, `can't start`, `fails to launch`, `won't start` |
| major | 2 | `broken`, `freeze`, `hang`, `unplayable`, `regression`, `not working`, `can't play`; label `Regression` |
| minor | 1 | `glitch`, `audio`, `stutter`, `flickering`, `font`, `artifact`, `visual`, `minor` |
| not_bug | 0 | label `Feature Request`; keywords `feature request`, `question`, `wiki`, `suggestion` |

Приоритет: label `Feature Request` → not_bug; label `Regression` → min major; иначе keyword match; default = minor (1).

### 2. Per-game агрегация

Из classified issues → per-game метрики:

```
max_severity        — максимальный severity среди всех issues игры
critical_count      — количество critical issues
major_count         — количество major issues
has_open_critical   — есть ли открытый critical issue
has_regression      — label Regression (уже есть)
fixed_ratio         — closed_completed / issue_count (доля пофиксенных)
```

### 3. Интеграция

- **Enrichment**: данные уже сохраняются в `game_metadata` (github_* колонки)
- **Engine layer**: severity-агрегаты для рекомендации proton-версий и предупреждений о багах
- **ML**: пока не используется (13.6% покрытие, game embeddings покрывают per-game сигнал). Пересмотреть при росте покрытия

### 4. Инкрементальное обновление

При повторном запуске — фетчить только обновлённые issues:

```bash
gh issue list --state all --search "updated:>2026-03-10" --json ...
```

Сохранять `last_fetched_at` в `meta` таблицу.

### 5. Proton version extraction → compatibility timeline

Issues привязаны к конкретным версиям Proton. Из body/title извлекаем версию (regex из `_parse_proton_features` уже есть):

```
Body: "Proton version: 9.0-4"        → proton="9.0-4", family=official
Body: "Proton version: GE-Proton9-27" → proton="GE-Proton9-27", family=ge
Body: "Proton version: Experimental"   → proton="experimental", family=experimental
```

Результат — тройки `(app_id, proton_version, severity, state)`:

```
(3131680, "9.0-4", major, open)     — regression на 9.0-4
(477740,  "8.0-4", major, open)     — .NET issue на 8.0-4
(1778820, "9.0-2", minor, closed/completed) — пофикшено
```

### 6. Per-game compatibility timeline (engine layer)

Из троек строим per-game knowledge base — **не ML-модель**, а structured lookup для `engine/proton.py`:

```
Game 3131680 (Stardust Skate):
  9.0-4: 1 open major [Regression] → AVOID
  9.0-3: 0 issues                  → OK
  → recommend: 9.0-3

Game 477740 (Zero Escape):
  8.0-4: 1 open major [.NET]       → AVOID
  GE-Proton9: 0 issues             → OK
  → recommend: GE-Proton9, warn: .NET issues on official
```

**Почему не отдельная ML-модель:**
- 4.5K игр с issues, версия указана не во всех — слишком мало для обучения
- Задача детерминированная: «есть открытый critical issue на версии X» → avoid X
- Regression detection — это lookup, не prediction

**Комбинация с ProtonDB reports:**
Engine layer объединяет два источника regression detection:
1. **ProtonDB reports**: массовый сигнал (348K отчётов), per-version borked rate
2. **GitHub issues**: экспертный сигнал (Valve triage, labels, резолюции)

GitHub issues усиливают ProtonDB: если на версии X и borked rate высокий в отчётах, и есть открытый GitHub issue — confidence в «avoid X» выше.

### 7. ML-интеграция: conditional features (game × proton_version)

Сырые counts (issue_count, open_count) не дали эффекта — game embeddings уже кодируют per-game сигнал. Но embeddings **не знают** про связку (game × proton_version). GitHub issues привязаны к конкретным версиям — это даёт новый сигнал.

**Идея:** модель уже принимает `proton_major` и `variant` как вход. Создаём interaction features — «есть ли issue на эту игру именно на той версии Proton, которую юзер использует».

Пример:
```
Report: game=477740, proton_major=8, variant=official
GitHub: issue #6999 на game=477740, proton="8.0-4" [Regression, .NET]
→ github_issue_this_version=1, github_severity_this_version=2, github_regression_this_version=1

Report: game=477740, proton_major=9, variant=ge
GitHub: нет issues на эту комбинацию
→ github_issue_this_version=0, github_severity_this_version=0, github_regression_this_version=0
```

#### Version-specific фичи (interaction game × proton):

| Фича | Тип | Описание |
|------|-----|----------|
| `github_issue_this_version` | binary | Есть issue на эту (game, proton_major, family) |
| `github_severity_this_version` | int 0-3 | Max severity issues на эту версию |
| `github_open_this_version` | binary | Есть открытый issue на эту версию |

#### Version-independent фичи (per-game, но не raw counts):

| Фича | Тип | Описание |
|------|-----|----------|
| `github_has_any_issue` | binary | Есть хотя бы один issue |
| `github_max_severity` | int 0-3 | Max severity по всем issues игры |
| `github_fixed_ratio` | float | completed / total — proxy «Valve внимание» |

#### Реализация

Зависит от шагов 1 и 5 (severity classification + version extraction).

1. Извлечь proton version из issue body (regex из `_parse_proton_features`)
2. Classify severity (keyword-based)
3. Построить index: `(app_id, proton_major, family)` → `{max_severity, has_open}`
4. В feature extraction — lookup по `proton_major`/`variant` из текущего report
5. LightGBM нативно обрабатывает NULL (86% reports без github данных)

#### Порядок реализации

1. Шаг 1 — severity classification
2. Шаг 5 — proton version extraction из issue body
3. Шаг 7 — conditional features + тренировка
4. Шаг 6 — engine layer (lookup, не ML)

---

## Результаты экспериментов (ML)

### Данные

- **9037 issues** сфетчены через `gh issue list`
- **6398** matched to app_id → **4579 unique games**
- **66.4%** покрытие отчётов (популярные игры)
- Proton version извлечена из **78.9%** issues (5749 уникальных ключей game×major×family)
- Version-specific match: **13.6%** отчётов (остальные — NaN)

### Эксперимент 1: Текстовые эмбеддинги + raw severity

| Вариант | F1 macro | Δ vs baseline |
|---|---|---|
| Baseline (текущие фичи) | 0.7318 | — |
| C: title+body embeddings SVD16 | 0.7313 | −0.0005 |
| A: raw severity counts | 0.7310 | −0.0008 |
| B: title embeddings SVD16 | 0.7307 | −0.0011 |
| D: severity + embeddings | 0.7304 | −0.0014 |

**Вывод**: text embeddings и raw severity counts бесполезны — game embeddings уже кодируют per-game «репутацию».

### Эксперимент 2: Version-specific interaction features

| Вариант | F1 macro | Δ vs baseline |
|---|---|---|
| **C: combined** | **0.7329** | **+0.0014** |
| A: version-independent | 0.7320 | +0.0005 |
| Baseline | 0.7315 | — |
| B: version-specific only | 0.7314 | −0.0001 |

**Feature importance** (combined model):

Stage 2 (tinkering vs works_oob):
- `gh_severity_this_version`: **13788** — топ-1, версионно-специфичный severity
- `gh_fixed_ratio`: 4147 — «Valve обратило внимание»
- `gh_issue_this_version`: 3973 — interaction signal

Stage 1 (borked vs works):
- `gh_fixed_ratio`: 5689 — топ-1
- `gh_severity_this_version`: 2017

**Вывод**: interaction features (game × proton_version) дают +0.0014 F1. Прирост на грани значимости, **не интегрировано**. `gh_severity_this_version` — самая важная фича в Stage 2, что подтверждает гипотезу об interaction signal, но покрытие 13.6% ограничивает эффект.

### Эксперимент 3: Deck Verified features

| Вариант | F1 macro | Δ vs baseline |
|---|---|---|
| B: deck_tests flags | 0.7320 | +0.0005 |
| C: deck_status + tests | 0.7316 | +0.0001 |
| A: deck_status only | 0.7315 | +0.0001 |
| Baseline | 0.7315 | — |

**Вывод**: Deck Verified данные (96% покрытие) не несут нового сигнала. Deck оценивает UX на Steam Deck (контроллер, текст, разрешение), а наша модель — работоспособность через Proton. Разные оси. **Не интегрировано**.
