# Phase 10: Качество данных и лейблов

## Контекст

После Phase 9: 119 фичей, F1=0.7253, cascade architecture (Stage 1 borked/works + Stage 2 tinkering/works_oob).

Phase 9.4–9.5 (14 экспериментов) показали, что модель достигла потолка для текущего уровня label noise (~15–20%). Добавление фичей, смена архитектуры (ordinal, focal loss, distillation, target encoding, cross-entity stats) — всё отрицательно или нейтрально.

**Главное ограничение — качество лейблов**, особенно граница tinkering ↔ works_oob (80% ошибок Stage 2). Текущий bottleneck: works_oob F1=0.508 (recall ~50%).

Два направления:
- (A) Улучшение существующих лейблов (relabeling, soft labels, confidence weighting)
- (B) Увеличение данных (re-scrape, minority oversampling)

Ожидаемый суммарный эффект: **+0.01–0.03 F1**.

---

## Phase 10.1 — Relabeling через extracted_data (1–2 дня, +0.005–0.015 F1)

### L1. Structured actions → relabeling

**Что:** Расширить relabeling (Phase 8) используя structured actions из LLM-извлечения (`extracted_data.actions_json`). Сейчас relabeling проверяет только `notes_customizations`, `notes_launch_flags` и effort-маркеры в тексте (~50 regex). Но у нас есть 13 типов structured actions с классификацией `effective`/`ineffective`/`unclear`.

**Правила:**
- Если `actions_json` пуст или содержит только `ineffective` действия → кандидат на relabel tinkering → works_oob
- Если есть `effective` действия типа `env_var`, `protontricks_verb`, `dll_override`, `registry_patch` → подтверждённый tinkering (не relabel)
- Если `actions_json` содержит только `runner_selection` (выбор GE-Proton) → граничный случай, обрабатывать отдельно

**Гипотеза:** Текущий regex-based relabeling ловит ~51% tinkering. Structured actions дают более точную классификацию: LLM уже разобрал текст и вычленил конкретные действия. Это должно уточнить boundary.

**Инференс:** ✅ Да — relabeling применяется только при обучении.
**Эффект:** Medium-high (+0.005–0.015 F1). Каждый % правильно relabeled = ~0.002 F1.
**Стоимость:** Низкая. Данные уже в БД. ~40 строк.

---

## Phase 10.2 — Confidence-weighted training (1–2 дня, +0.003–0.008 F1)

### L2. Per-game agreement weighting

**Что:** Использовать `sample_weight` в LightGBM на основе согласованности лейблов per game.

Для каждой игры: если 15 из 20 отчётов = tinkering и 5 = works_oob, то:
- Отчёты majority class (tinkering) получают weight=1.0
- Отчёты minority class (works_oob) получают weight=0.5–0.7
- Пропорционально: `weight = agreement_rate` (% отчётов с тем же лейблом в этой игре)

**Гипотеза:** Outlier-лейблы в пределах одной игры — скорее всего noise. Если 90% отчётов для Counter-Strike = works_oob, а один отчёт = tinkering, то этот один скорее ошибка. Уменьшение веса outlier-отчётов эквивалентно soft noise reduction без удаления данных.

**Инференс:** ✅ Да — weights только при обучении.
**Эффект:** Medium (+0.003–0.008 F1). Адресует ~15% noisy labels без потери данных.
**Стоимость:** Низкая. `sample_weight` в lgb.Dataset. ~25 строк.

### L3. ProtonDB confidence weighting

**Что:** Игры с `protondb_confidence='weak'` (mixed community verdicts) → пониженный sample_weight (0.7). `strong` → 1.0, `good` → 0.9.

**Гипотеза:** ProtonDB confidence = proxy для label noise: weak означает что community не согласна по поводу совместимости. Уменьшение веса таких отчётов снижает влияние noisy labels.

**Инференс:** ✅ Да — weight при обучении.
**Эффект:** Low (+0.001–0.003 F1). Покрытие только 20% игр.
**Стоимость:** Тривиально. ~10 строк.

---

## Phase 10.3 — Soft labels (2–3 дня, +0.003–0.010 F1)

### L4. Per-game soft labels

**Что:** Вместо hard labels (0/1/2) для Stage 2 — soft labels на основе per-game verdict distribution.

Для Stage 2 (tinkering vs works_oob):
- Игра с 80% tinkering, 20% works_oob → отчёт "works_oob" получает soft label `0.3 * tinkering + 0.7 * oob = 0.7` (вместо 1.0)
- Отчёт "tinkering" получает soft label `0.85 * tinkering + 0.15 * oob = 0.15` (вместо 0.0)
- Smoothing пропорционален доле противоположного класса в этой игре

**Отличие от Phase 9.1 label smoothing:** Phase 9.1 применяет global α=0.15 ко всем. Здесь — per-game adaptive smoothing, основанный на реальной agreement rate.

**Гипотеза:** Soft targets кодируют неопределённость per game. Для игр с strong agreement (95% tinkering) — labels почти hard. Для contested games (60/40) — сильный smoothing. Это адаптивно к шуму.

**Инференс:** ✅ Да — soft labels при обучении, cross_entropy objective уже поддерживает float targets.
**Эффект:** Medium (+0.003–0.010 F1). Объединяет noise robustness с per-game signal.
**Стоимость:** Medium. ~50 строк. Нужно аккуратно взаимодействовать с Phase 9.1 label smoothing (не double-smooth).

### L5. Verified games as anchor labels

**Что:** Для игр с `deck_status=3` (Verified) и `protondb_tier='platinum'` — повысить confidence лейблов works_oob. Для `deck_status=1` (Unsupported) — повысить confidence borked.

**Гипотеза:** Steam Deck Verified и ProtonDB Platinum — высококачественные внешние сигналы. Отчёты для таких игр, совпадающие с внешним verdict, заслуживают повышенного веса.

**Инференс:** ✅ Да.
**Эффект:** Low (+0.001–0.003 F1). Покрытие ~15% для Deck Verified.
**Стоимость:** Низкая. ~15 строк.

---

## Phase 10.4 — Увеличение данных (ongoing)

### L6. SMOTE на Cleanlab-filtered данных

**Что:** Oversampling works_oob (minority, 15%) через SMOTE или ADASYN, но только на данных после Cleanlab фильтрации (clean subset).

**Гипотеза:** works_oob — самый дефицитный класс. SMOTE на чистых данных генерирует реалистичные синтетические сэмплы. На noisy данных SMOTE усиливает шум — поэтому только post-Cleanlab.

**Инференс:** ✅ Да.
**Эффект:** Low-medium (+0.002–0.005 F1). Зависит от качества synthetic samples.
**Стоимость:** Низкая. `imblearn.over_sampling.SMOTE`. ~15 строк.

### L7. Периодический re-scrape

**Что:** Настроить периодический запуск worker для сбора новых отчётов с ProtonDB.

**Гипотеза:** ProtonDB получает ~5–10K новых отчётов/месяц. За 3 месяца = +15–30K отчётов (+5–8%). Особенно ценны отчёты для игр с <5 отчётов (37% игр имеют только 1 отчёт).

**Инференс:** ✅ Да.
**Эффект:** Indirect. +5–8% данных → +0.001–0.003 F1.
**Стоимость:** Medium. Worker уже реализован, нужна настройка cron/scheduling.

---

## Phase 10.5 — Tried_OOB верификация (1 день, +0.001–0.003 F1)

### L8. Противоречия tried_oob vs verdict

**Что:** 102K отчётов (29%) имеют `tried_oob`. Выявить противоречия:
- `tried_oob=no` + `verdict_oob=yes` → невозможно, но встречается? → noise
- `tried_oob=yes` + `verdict_oob=no` + текст "just works" → кандидат на relabel → oob
- `tried_oob=no` + `verdict=yes` → confirmed tinkering (не пробовал OOB, но работает с настройками)

**Гипотеза:** `tried_oob` — объективный сигнал (пользователь сам выбрал). Противоречия с verdict → noise.

**Инференс:** ✅ Да.
**Эффект:** Low (+0.001–0.003 F1). Только 29% данных, ~5% из них с противоречиями.
**Стоимость:** Низкая. ~20 строк.

---

## Порядок реализации

```
Phase 10.1 (1–2 дня):  L1 (extracted_data relabeling)         → +0.005–0.015 F1
Phase 10.2 (1–2 дня):  L2 + L3 (confidence weighting)          → +0.003–0.008 F1
Phase 10.3 (2–3 дня):  L4 + L5 (soft labels + anchor)          → +0.003–0.010 F1
Phase 10.4 (ongoing):  L6 + L7 (SMOTE + re-scrape)             → +0.002–0.005 F1
Phase 10.5 (1 день):   L8 (tried_oob verification)             → +0.001–0.003 F1
                                                         Итого:   +0.01–0.03 F1
```

## Метрики

| Метрика | Phase 9 финал | Цель Phase 10 |
|---|---|---|
| F1 macro | 0.7253 | **> 0.74** |
| works_oob F1 | 0.508 | **> 0.55** |
| borked F1 | 0.825 | **≥ 0.825** |
| ECE | 0.008 | **< 0.010** |
| Label noise (est.) | 15–20% | **< 12%** |
