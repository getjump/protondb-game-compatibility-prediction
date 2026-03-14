# ML Pipeline — план и анализ

## Архитектура

- **Модель**: LightGBM (multi-class classification)
- **Целевая переменная**: `borked` / `needs_tinkering` / `works_oob`
  - `borked` = verdict=no
  - `works_oob` = verdict_oob=yes
  - `needs_tinkering` = verdict=yes & oob != yes
- **Фичи**: 92 (hardware + game metadata + aggregated stats + SVD embeddings)
- **Артефакты**: `model.pkl` (LightGBM) + `embeddings.npz` (SVD) + `label_maps.json`
- **Нормализация**: heuristic (по умолчанию) или llm — переключается через `NORMALIZED_DATA_SOURCE` env / `--normalized-data` CLI-флаг

## Текущие результаты (2026-03-10)

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.6799 |
| F1 macro | 0.5553 |
| Samples | 348 536 |
| Features | 98 |
| Train/Test | 278 828 / 69 708 (time-based split) |

Изменения: убран target leakage, добавлены vendor-split driver versions, proton_version, class balancing.

### Per-class метрики

| Класс | Доля | Precision | Recall | F1 |
|-------|------|-----------|--------|-----|
| borked | 19.7% | 0.60 | 0.37 | 0.46 |
| needs_tinkering | 69.2% | 0.77 | 0.81 | 0.79 |
| works_oob | 11.1% | 0.40 | 0.44 | 0.42 |

### Confusion matrix

```
                  Predicted
              borked  tinkering  oob
Actual borked   3684     4862    1303
       tinker   2036    38179    7016
       oob       426     6673    5529
```

---

## Ревью фич

### Game metadata фичи: ожили ✓

`game_metadata` заполнена из enrichment_cache (30968 игр). Покрытие: engine 11K, graphics_apis 7.7K, drm 17.8K, developer 18.7K (Steam). Ключевые фичи в SHAP: `developer` (0.16, топ-5), `publisher` (0.08), `engine` (0.06).

### Target leakage: убран ✓

`pct_works_oob` и `avg_verdict_score` убраны из фич — были агрегатами целевой переменной. Удаление не повлияло на accuracy (+0.001).

### Живые фичи (топ по importance)

| Фича | Splits | SHAP | Группа |
|------|--------|------|--------|
| `kernel_major` | 134K | 0.20 | hardware |
| `driver_major` | 70K | 0.19 | hardware |
| `game_emb_0..15` | 15–60K | 0.02–0.10 | embedding |
| `gpu_family` | 59K | 0.10 | hardware |
| `total_reports` | 34K | 0.05 | aggregated |
| `pct_steam_launcher` | 30K | 0.06 | aggregated |
| `gpu_emb_0..15` | 8–18K | — | embedding |
| `os_family` | 17K | 0.03 | hardware |
| `cpu_emb_0..15` | 3–12K | — | embedding |
| `ram_gb` | 13K | 0.02 | hardware |
| `cpu_generation` | 10K | 0.02 | hardware |

### Неиспользуемые данные из reports

| Данные | Покрытие | Потенциал |
|--------|----------|-----------|
| `proton_version` + `custom_proton_version` | 7% + 12% | **Высокий** — версия Proton = ключевой фактор |
| `launch_options` | 14% | Средний |
| Per-report fault flags | 4–12% | Средний |
| `window_manager` | высокое | Низкий |
| Steam Deck fields | 13% | Средний (частично используется) |

---

## План улучшений (по приоритету)

### 1. Починить enrichment → game_metadata ✓

`game_metadata` заполнена из enrichment_cache: PCGamingWiki (30968) + Steam (19468). Фичи ожили: `developer` (SHAP 0.16), `publisher` (0.08), `engine` (0.06). F1 macro чуть просел (0.55 → 0.52) — вероятно developer/publisher добавляют шум (много категорий). Требуется тюнинг кардинальности или исключение этих фич.

### 2. Добавить `proton_version` как фичу ✓

Добавлены 3 фичи: `proton_major`, `is_ge_proton`, `has_proton_version`. Парсер покрывает 98.9% непустых значений (GE-Proton9-27, Proton-6.21-GE-2, 7.2-GE-2, 6.3-8 и т.д.).

Покрытие: 19% отчётов (81% NULL). Пока не попадают в топ SHAP — сигнал подавлен большим количеством NULL. Могут начать работать после class balancing.

### 3. Разделить driver version по вендору + полная версия ✓

Добавлены `nvidia_driver_version` (major + minor/1000) и `mesa_driver_version` (major + minor/10). `driver_major` оставлен как fallback для отчётов вне lookup.

**Результат**: `nvidia_driver_version` (SHAP 0.21) и `mesa_driver_version` (SHAP 0.18) вошли в топ-3. Суммарный SHAP по драйверам вырос в 2.5× (0.18 → 0.45). Accuracy/F1 на уровне baseline — эффект заблокирован class imbalance.

### 4. Class balancing ✓

Добавлен `class_weight={0: 2.0, 1: 1.0, 2: 3.0}`. Результаты:

| Веса | Accuracy | F1 macro | borked R | tinkering R | oob R |
|------|----------|----------|----------|-------------|-------|
| None | 0.7235 | 0.4897 | 0.30 | 0.97 | 0.13 |
| {0:1.5, 1:1, 2:2} | 0.6989 | 0.5454 | 0.38 | 0.87 | 0.31 |
| **{0:2, 1:1, 2:3}** | **0.6433** | **0.5526** | **0.43** | **0.72** | **0.53** |
| balanced | 0.4880 | 0.4830 | 0.48 | 0.39 | 0.85 |

F1 macro +13%, recall миноритарных классов вырос в 2-4×. `is_unbalance=True` не работает для multi-class.

### 5. Фичи форм-фактора: is_apu, is_igpu, is_mobile, is_steam_deck ✓

Добавлены 4 бинарные фичи из gpu_normalization_heuristic + runtime detection. Покрытие: is_steam_deck 14.1% (49K), is_igpu 1927, is_apu 1679, is_mobile 307 уникальных GPU-строк. Добавлен `is_mobile` парсинг в gpu_heuristic (Mobile/Laptop/Max-Q/Max-P + NVIDIA M-suffix chip codes).

Форм-фактор GPU значимо влияет на совместимость:

| Тип | Отчёты | Borked | OOB | Комментарий |
|-----|--------|--------|-----|-------------|
| Discrete | 274K | 0.202 | 0.115 | Базовый уровень |
| Steam Deck | 49K | 0.143 | 0.082 | Valve оптимизирует |
| APU (не Deck) | 4.4K | 0.153 | 0.171 | ROG Ally, Legion Go etc |
| iGPU Intel | 18K | 0.274 | 0.099 | На 7% хуже discrete |
| Laptop (Mobile/Max-Q) | 10.5K | 0.198–0.287 | — | Max-Q хуже desktop |
| Virtual | 2.7K | 0.294 | 0.111 | Ожидаемо плохо |

Для одной и той же GPU: RTX 3060 desktop borked=0.173 vs mobile=0.212 (+4%).

Добавить 4 бинарные фичи в `extract_hardware_features`:
```python
is_apu         # AMD APU (Deck, ROG Ally, встроенная Vega)
is_igpu        # Intel iGPU (HD/UHD/Iris) — без APU
is_mobile      # Laptop GPU (Mobile/Laptop/Max-Q в raw string)
is_steam_deck  # Отдельно от APU: gpu LIKE '%vangogh%' OR battery_performance IS NOT NULL
```

`is_apu` и `is_igpu` уже есть в `gpu_normalization_heuristic`. Семантика:
- `is_igpu=1` — **любая** встроенная графика: Intel HD/UHD/Iris, AMD Vega/RDNA iGPU в Ryzen APU, Steam Deck
- `is_apu=1` — AMD APU конкретно (Ryzen с Vega/RDNA, Steam Deck, ROG Ally)
- `is_apu` ⊂ `is_igpu` — каждый APU имеет встроенный GPU
- Текущая разметка (`is_apu=1 AND is_igpu=1` для AMD APU) корректна

**Требуется починка в `gpu_heuristic.py`**:
- Сейчас Intel iGPU помечен как `is_igpu=1`, AMD APU как `is_apu=1 AND is_igpu=1` — это правильно
- Но некоторые AMD APU (определённые через radeonsi chip name) не всегда получают `is_igpu=1` — нужно убедиться что все APU-чипы в `_APU_CHIPS` корректно проставляют оба флага
- Добавить `is_mobile` — парсится из raw string (Mobile/Laptop/Max-Q), сейчас такой колонки нет в таблице

### 6. Временные фичи ✓

Добавлен `report_age_days` (дни от самого нового отчёта). **SHAP 1.10 — топ-1 фича**. Accuracy +0.027, F1 macro +0.011. Разблокировал `proton_major` (0.08) и `is_ge_proton` (0.04) — ранее невидимые.

---

## SVD Embeddings

Матрицы ко-встречаемости hardware_family × game, усечённое SVD:
- **GPU**: 33 семейства × 30917 игр → 16 компонент (91.2% дисперсии)
- **CPU**: 17 семейств × 30936 игр → 16 компонент (100% дисперсии)
- **Game**: из GPU-game SVD, 30917 игр × 16 компонент

CPU embeddings дают 100% дисперсии на 16 компонентах = мало разнообразия (всего 17 семейств). Можно сократить до 8.

Game embeddings (`game_emb_0..15`) — 5-я по важности группа фич. Кодируют латентное "какая это игра с точки зрения железа".

## Нормализация данных

Два источника нормализованных данных (переключается через `NORMALIZED_DATA_SOURCE`):

| Источник | GPU | CPU | Driver | Скорость | Качество |
|----------|-----|-----|--------|----------|----------|
| `heuristic` | 35K строк, regex | 2.4K строк, regex | 4.9K строк, regex | <1 сек | 99.7% покрытие, без галлюцинаций |
| `llm` | 35K (LLM) | 2.4K (LLM) | — | ~9 часов | галлюцинирует числа |

По умолчанию используется `heuristic`. Таблицы: `*_normalization_heuristic` и `gpu_driver_normalization`.
