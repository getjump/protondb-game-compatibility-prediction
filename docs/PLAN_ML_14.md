# Phase 14: Steam PICS features

## Контекст

Steam PICS (Package Info Cache System) даёт данные через CM протокол — bulk fetch 31K игр за 10 минут. Данные уже собраны на 100% (30968/30968).

PICS содержит поля, недоступные через Store API:
- `recommended_runtime` — Valve прямо указывает "proton-experimental" или "native"
- Granular Deck tests — per-test pass/fail
- `steamos_compatibility` — отдельная от `deck_category` категория
- `osarch` (32/64-bit), launch configs per OS, depot info

**Покрытие полей (30956 non-empty apps):**

| Поле | Покрытие | Новое vs имеющееся |
|---|---|---|
| `review_score` + `review_percentage` | 94% | **Новое** — не собирали |
| `recommended_runtime` | 58% (proton: 16263, native: 1749) | **Новое** — прямой сигнал |
| `deck_category` (granular) | 58% | Дополняет текущий `deck_status` |
| `steamos_compatibility` | 57% | **Новое** — отдельно от Deck |
| `deck_test_results` | 58% | **Новое** — per-test pass/fail |
| `osarch` (32/64-bit) | 34% | **Новое** |
| `has_linux_launch` | 16% | **Новое** — launch entry для Linux |
| `linux_depot_count` | 15% | **Новое** — Linux depot наличие |
| `oslist` | 96% | Подтверждает `has_linux_native` |
| `developer`, `publisher` | 96% | Уже есть из Steam Store |

---

## Phase 14.1 — Базовые PICS features (1 день, +0.005-0.015 F1)

### Новые фичи из PICS данных

**Категория A — Прямые сигналы совместимости:**

1. `recommended_runtime_is_native` (binary) — Valve рекомендует native runtime
2. `recommended_runtime_is_proton` (binary) — Valve рекомендует Proton
3. `recommended_runtime_proton_version` (ordinal) — версия Proton (experimental > 9.0 > 8.0 > ...)
   - Новый Proton = новые фиксы; старый = стабильный, проверенный
4. `steamos_compatibility` (ordinal 0-3) — SteamOS категория (отдельно от Deck)
5. `osarch_64bit` (binary) — 64-bit архитектура

**Категория B — Косвенные сигналы:**

6. `has_linux_launch` (binary) — есть launch entry для Linux
7. `has_linux_depot` (binary) — есть Linux depot
8. `linux_depot_ratio` = linux_depot_count / total_depot_count
9. `review_score` (1-9) — общее качество игры
10. `review_percentage` (0-100) — % положительных отзывов
11. `is_free` (binary) — F2P игры имеют другие паттерны

**Категория C — Deck test details (binary per-test):**

12. `deck_test_controller_ok` — контроллер работает
13. `deck_test_glyphs_match` — правильные иконки кнопок
14. `deck_test_text_legible` — текст читаем
15. `deck_test_performant` — производительность ОК
16. `deck_test_anticheat_fail` — блокирующий античит
17. `deck_test_count_pass` — кол-во пройденных тестов
18. `deck_test_count_fail` — кол-во проваленных
19. `deck_test_count_warn` — кол-во предупреждений

**Гипотеза:**
- `recommended_runtime` — самый сильный новый сигнал: Valve явно говорит какой Proton использовать. native = точно works, proton-experimental = скорее всего works, нет runtime = не тестировали.
- Deck granular tests дают больше информации чем single `deck_status`: `anticheat_fail` = прямой predictor borked.
- `review_score` коррелирует с quality/polish → косвенно с compatibility.

**Инференс:** Да — все поля per-game, доступны для любой игры.
**Эффект:** Medium-high (+0.005-0.015 F1). `recommended_runtime` и Deck tests — прямые сигналы.
**Стоимость:** ~40 строк в `_build_feature_matrix`. Данные уже в `enrichment_cache`.

---

## Phase 14.2 — Интеграция с merger (1 день)

### Добавление PICS данных в game_metadata

**Что:** Расширить `merge_metadata()` для обработки PICS cache. Добавить колонки в `game_metadata`:

```sql
ALTER TABLE game_metadata ADD COLUMN recommended_runtime TEXT;
ALTER TABLE game_metadata ADD COLUMN steamos_compatibility INTEGER;
ALTER TABLE game_metadata ADD COLUMN osarch TEXT;
ALTER TABLE game_metadata ADD COLUMN review_score INTEGER;
ALTER TABLE game_metadata ADD COLUMN review_percentage INTEGER;
ALTER TABLE game_metadata ADD COLUMN has_linux_launch INTEGER;
ALTER TABLE game_metadata ADD COLUMN linux_depot_count INTEGER;
ALTER TABLE game_metadata ADD COLUMN deck_tests_detailed_json TEXT;
```

**Альтернатива:** Читать напрямую из `enrichment_cache` в `_build_feature_matrix` (проще, не трогает merger). Для эксперимента — альтернатива.

---

## Phase 14.3 — pc_requirements парсинг (1 день, +0.003-0.010 F1)

### DirectX version extraction

**Что:** Парсить HTML поле `pc_requirements` из существующего Steam Store cache для извлечения DirectX версии.

**Паттерны:**
```python
dx_patterns = [
    r"DirectX.*?(\d+)",       # "DirectX: Version 11" → 11
    r"Direct3D\s*(\d+)",      # "Direct3D 12" → 12
    r"DX(\d+)",               # "DX11" → 11
    r"Vulkan",                # → dx_version = 0 (native Vulkan)
    r"OpenGL\s*([\d.]+)",     # "OpenGL 4.5" → encode separately
]
```

**Фичи:**
- `dx_version` (ordinal: 9, 10, 11, 12) — прямой impact на DXVK/VKD3D path
- `uses_vulkan` (binary) — native Vulkan = отличная совместимость
- `uses_opengl` (binary)

**Гипотеза:** DX version — один из самых прямых сигналов:
- DX9/10 → DXVK, очень стабильно → скорее works
- DX11 → DXVK, стабильно → скорее works
- DX12 → VKD3D, менее стабильно → больше шанс tinkering/borked
- Vulkan → native, без translation → works

**Данные:** `pc_requirements` уже в Steam Store cache (`enrichment_cache` source='steam'). Нужно только парсинг.

**Инференс:** Да.
**Эффект:** Low-medium (+0.003-0.010 F1).
**Стоимость:** ~30 строк (regex parsing).

---

## Порядок реализации

```
Phase 14.1 (1 день):   PICS features в feature matrix     → +0.005-0.015 F1
Phase 14.3 (1 день):   pc_requirements парсинг (DX ver)    → +0.003-0.010 F1
Phase 14.2 (1 день):   Интеграция с merger (optional)      → infrastructure
                                                     Итого: +0.008-0.025 F1
```

**Приоритет:** 14.1 → 14.3 → 14.2

14.1 — эксперимент с PICS features (данные уже есть, нужен только feature extraction).
14.3 — DX version из уже собранных Store данных.
14.2 — infrastructure (делать если 14.1 даёт результат и нужно интегрировать в pipeline).

---

## Зависимости

- `enrichment_cache` source='steam_pics' — ✅ 100% собрано (30968/30968)
- `enrichment_cache` source='steam' — уже собрано (Store API данные для pc_requirements)

---

## Результаты (2026-03-14)

| Experiment | F1 macro | ΔF1 | borked | tinkering | works_oob |
|---|---|---|---|---|---|
| baseline (IRT) | 0.7719 | — | 0.840 | 0.886 | 0.589 |
| 14.1a runtime only | 0.7712 | −0.001 | 0.839 | 0.886 | 0.589 |
| 14.1b all PICS | 0.7721 | +0.000 | 0.840 | 0.886 | 0.590 |

### Выводы

**Нулевой эффект.** PICS features не добавляют value поверх IRT + existing features.

Причины:
- `recommended_runtime` коррелирует с `has_linux_native` + `deck_status` (уже есть)
- `review_score` коррелирует с game aggregates
- Deck granular tests redundant при наличии `deck_status`
- `osarch` слишком sparse (34%)

PICS данные полезны для enrichment infrastructure (быстрее Store API), но как ML features — redundant. IRT + game aggregates доминируют.

**Статус: Phase 14 закрыт. Нет ML value.**

## Метрики

| Метрика | Phase 13 | Phase 14 |
|---|---|---|
| F1 macro | 0.7711 | 0.7721 (≈0) |
| works_oob F1 | 0.591 | 0.590 |
| borked F1 | 0.837 | 0.840 |
