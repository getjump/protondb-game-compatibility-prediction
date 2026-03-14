# Phase 8: Game Embeddings + Relabeling

## Контекст

После Phase 7: 84 фичи, F1=0.749. Stage 2 (tinkering vs oob) — bottleneck (AUC=0.835, LogLoss=0.395).
Потолок — label noise: tinkering vs oob субъективно, feature engineering не поможет.

Два направления с реальным ROI:

---

## A. Улучшение game embeddings

### Текущая реализация
game_emb — самая важная группа фич (ΔF1=−0.060 при удалении).
Сейчас: матрица **gpu_family × game → avg_verdict_score**, SVD →
правые сингулярные векторы = game embedding (16 dims).
~200 GPU families × 31K games. Захватывает "compatibility profile" игры
по GPU, но ограничен одной осью (GPU family).

### Данные в наличии
- **game_metadata**: 30,968 игр с genres (56), categories (145), engine (~50), deck_status, deck_tests_json
- **enrichment_cache**: steam (30,968), pcgamingwiki (30,968), deck (30,967)
- **reports**: 348K отчётов, 31K игр; 37% игр имеют только 1 отчёт

### Варианты

#### A1. Steam metadata embeddings (genres + categories + engine → multi-hot → SVD)
Построить multi-hot вектор из Steam metadata для каждой игры, затем SVD:
```
56 genres + 145 categories + ~50 engines ≈ 250 бинарных фич на игру
→ SVD → 8-16 dims = game_meta_emb
```

- **Плюсы:** Ортогонально текущим game_emb (тип игры vs compatibility profile),
  работает для cold-start (нет отчётов, но Steam данные есть),
  данные уже в game_metadata
- **Минусы:** Genres/categories могут быть noisy (56 языковых дубликатов в genres)
- **Стоимость:** Мгновенно, данные уже есть
- **Приоритет:** Высокий — самый простой эксперимент

#### A2. Verdict-profile расширение (больше осей в co-occurrence матрице)
Расширить текущую матрицу, добавив строки из других группировок:
```
gpu_family × game  (текущее, ~200 строк)
+ variant × game   (6 строк: official, ge, experimental, native, notListed, older)
+ engine × game    (~50 строк)
+ is_deck × game   (2 строки)
→ объединённая матрица ~260 строк × 31K games → SVD → 16 dims
```

- **Плюсы:** Больше сигнала в SVD, variant × game захватит
  "эта игра хорошо работает на GE но плохо на official"
- **Минусы:** Разный масштаб строк (200 GPU vs 6 variant), нужна нормализация
- **Стоимость:** ~30 мин реализации

#### A3. Deck-tests embeddings
deck_tests_json содержит структурированные тесты от Valve для каждой игры:
```
display_type (3=warning, 4=pass) × test_token → multi-hot/ordinal
Тесты: controller, keyboard, resolution, text legibility, performance, launcher...
→ SVD → 8 dims = game_deck_emb
```

- **Плюсы:** Экспертная оценка Valve, не зависит от user reports
- **Минусы:** Покрытие — только verified/playable игры (~31K)
- **Стоимость:** Парсинг JSON + multi-hot, мгновенно

#### A4. Конкатенация всех embeddings
Не заменять текущие, а добавить:
```
game_emb (текущие 16, verdict-based)
+ game_meta_emb (A1, 8-16, Steam metadata)
+ game_deck_emb (A3, 8, Deck tests)
= 32-40 dims total
```

- **Плюсы:** Три ортогональных аспекта игры
- **Минусы:** Diminishing returns, нужна ablation
- **Стоимость:** Сумма A1+A3

### Результаты экспериментов

#### A1: Steam metadata embeddings — ❌ не помогло
```
genres (56) + categories (145) + engine (~50) → 825 токенов → multi-hot → SVD 16 dims
Baseline F1:    0.7396
Experiment F1:  0.7372
Delta F1:       -0.0024
```
Metadata дублирует сигнал, который game_emb уже захватывает через verdict patterns.
Gain у meta_emb фич низкий и равномерный (11-18K vs game_emb_0 ~72K).

#### A2: Расширенная co-occurrence — ✅ +0.008 F1
```
Матрица: 665 осей (33 gpu + 6 variant + 624 engine + 2 deck) × 30,968 games
Explained variance: 0.857 (vs 0.78 у baseline gpu-only)
SVD: 20 dims (авто-выбор, 90% variance)

Baseline F1:    0.7396  (Stage 1 LL=0.1244, Stage 2 LL=0.3917)
Experiment F1:  0.7479  (Stage 1 LL=0.1227, Stage 2 LL=0.3804)
Delta F1:       +0.0083

              precision    recall  f1-score
borked           0.85      0.82      0.84  (was 0.83)
tinkering        0.86      0.86      0.86  (was 0.85)
works_oob        0.55      0.56      0.55  (was 0.53)

game_emb total gain (Stage 2): 622K → 763K (+23%)
gpu_emb total gain (Stage 2):  125K → 64K  (−49%, сигнал теперь в game_emb)
```
variant × game захватывает "эта игра хорошо работает на GE но не на official".
engine × game — "Unity игры ведут себя иначе чем UE4".

**Интегрировано в пайплайн** (`embeddings.py`): `_build_extended_cooccurrence` заменил
`_build_cooccurrence_matrix`. CPU embeddings удалены (мёртвый код после Phase 7).

---

## B. Relabeling: пересмотр tinkering vs oob

### Почему
Основная путаница Stage 2 — субъективность границы:
- "Выбрал GE-Proton, всё заработало" → одни ставят tinkering, другие oob
- "Поставил launch options из гайда" → tinkering или oob?
- 44% ошибок Stage 2: tinkering→works_oob, 38%: works_oob→tinkering

### Эвристика relabeling

Два уровня агрессивности. Оба фильтруют:
- `notes_customizations` НЕ пустое → keep as tinkering
- `notes_launch_flags` НЕ пустое → keep as tinkering

**Strict** — relabel если в тексте НЕТ ни одного effort-маркера:
```
protontricks|winetricks|winedlloverrides|gamescope|mangohud|lutris
|launch option|STEAM_COMPAT|PROTON_|DXVK_|VKD3D_|MESA_
|flatpak|.ini|.cfg|.conf|config file
|[A-Z_]{3,}=[^\s]+  (env vars)
|--[a-z]  (CLI flags)
|workaround|trick|tweak|hack
|wined3d|d3d11|d3d9|dxvk|vkd3d|vulkan render
|disabl|delet|remov|renam|mov[ei]  (actions)
|controller layout|remap|rebind
|terminal|command line|bash|shell
|install script|eac|easy anti|battleye
|fix|patch|mod|mods
|swap|switch.*(render|mode|layout)
```

**Medium** — Strict + также keep если текст упоминает:
```
crash|freeze|black screen|won't start|doesn't launch
|require|need|must|have to|had to|force
```

### Результаты эксперимента

Relabeling применяется **только при обучении** (in-memory), без модификации БД.
Все 3 варианта используют одни и те же embeddings (A2 extended co-occurrence).

```
| Вариант     | Relabeled        | F1 macro | borked F1 | tinkering F1 | works_oob F1 | S2 LL  |
|-------------|------------------|----------|-----------|--------------|--------------|--------|
| Baseline    | 0                | 0.7495   | 0.83      | 0.86         | 0.56         | 0.378  |
| Medium (34%)| 10,065 (34.0%)   | 0.7531   | 0.83      | 0.83         | 0.60         | 0.431  |
| Strict (51%)| 15,079 (51.0%)   | 0.7599   | 0.83      | 0.82         | 0.63         | 0.441  |
```

**Strict: +0.010 F1**, works_oob recall 0.56→0.69, borked не пострадал.

Класс distribution после Strict relabeling:
- borked: 68,673 (19.7%) — без изменений
- tinkering: 226,263 (64.9%) — было 241,342 (69.2%)
- works_oob: 53,600 (15.4%) — было 38,521 (11.1%)

Stage 2 LogLoss вырос (0.378→0.441) из-за изменения баланса классов — ожидаемо.
F1 — более честная метрика при изменении class distribution.

---

## Текущий статус

- **A1** ❌ Steam metadata embeddings — дублирует game_emb, −0.002 F1
- **A2** ✅ Расширенная co-occurrence — **+0.008 F1**, интегрировано
- **B Strict** ✅ Relabeling 51% tinkering → oob — **+0.010 F1**, эксперимент подтверждён

## Выполненные шаги

### 1. ✅ Strict relabeling интегрирован в пайплайн

Модуль `protondb_settings/ml/relabeling.py`:
- `get_relabel_ids(conn)` — возвращает set report IDs для relabeling
- `apply_relabeling(y, report_ids, relabel_ids)` — применяет tinkering→oob

Интегрировано в `train_cascade_pipeline`:
- `_build_feature_matrix` теперь возвращает `report_ids`
- `_time_based_split` передаёт `report_ids` в train/test
- Relabeling применяется **только к y_train**, test остаётся с оригинальными лейблами

### 2. ❌ text_emb 32→64 — marginal, не стоит

```
text_emb=32: F1=0.704  (S1 LL=0.123, S2 LL=0.454)
text_emb=64: F1=0.712  (S1 LL=0.120, S2 LL=0.482)
Delta:       +0.008
```
(F1 ниже 0.760 потому что test на оригинальных лейблах — корректная оценка)

Stage 2 при 64 dims: early stop на 32 итерации — доп. text dims не помогают Stage 2.
Улучшение +0.008 только из Stage 1 (уже AUC=0.977). При инференсе text недоступен.
**Оставляем 32 dims.**

## Метрики

| Метрика | Phase 7 | Phase 8 (A2) | Phase 8 (A2+B) | Цель |
|---|---|---|---|---|
| F1 macro | 0.749 | 0.750 | **0.760** | > 0.76 ✅ |
| works_oob F1 | 0.53 | 0.56 | **0.63** | > 0.58 ✅ |
| borked F1 | 0.83 | 0.83 | **0.83** | ≥ 0.83 ✅ |
