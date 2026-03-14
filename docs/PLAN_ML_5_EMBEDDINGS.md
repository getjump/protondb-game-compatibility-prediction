# ML Pipeline — текстовые эмбеддинги (фаза 5, часть embeddings)

> Расширение [PLAN_ML_5.md](PLAN_ML_5.md). Текстовые фичи D+E дали +0.099 F1, но это грубые сигналы (длина, regex). Sentence embeddings могут захватить семантику.

## Контекст

**Текущее состояние** (после D+E):
- F1 macro: 0.692 (baseline 0.593)
- Stage 1 logloss: 0.178 (было 0.322) — хорошо оптимизирован
- Stage 2 logloss: 0.402 (было 0.418) — **bottleneck**, еле сдвинулся
- Покрытие: `all_text` 91%, `concluding_notes` 34%

**Гипотеза**: sentence embeddings захватят нюансы, которые regex не ловит:
- "works but needs tweaking for controller" vs "runs perfectly out of the box"
- "had to set PROTON_USE_WINED3D but after that perfect" (tinkering, не oob)
- "minor graphical glitches but playable" (степень серьёзности)

**Ожидание**: +0.005–0.015 F1 (основной буст уже взят через D+E)

---

## Выбор модели

| Модель | Dims | Размер | Скорость | Язык |
|--------|------|--------|----------|------|
| `all-MiniLM-L6-v2` | 384 | 80MB | ~14K sent/s (GPU) | EN |
| `all-mpnet-base-v2` | 768 | 420MB | ~2.5K sent/s (GPU) | EN |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | ~4K sent/s (GPU) | Multi |

**Рекомендация**: `all-MiniLM-L6-v2` — баланс качества и скорости. Тексты ProtonDB на 95%+ английские.

---

## Архитектура

### Precompute (offline)

```
reports.concluding_notes + all_text
        │
        ▼
  sentence-transformers (all-MiniLM-L6-v2)
        │
        ▼
  384-dim vectors × 348K = ~500MB raw
        │
        ▼
  SVD/PCA → 16-32 dims (как GPU/CPU embeddings)
        │
        ▼
  text_embeddings таблица в SQLite или .npz файл
```

### Варианты хранения

**A) NPZ файл** (аналог `embeddings.npz`):
- `text_embeddings.npz`: матрица (N × D) + report_id mapping
- Загружается при обучении, нет runtime зависимости от transformer
- Простой подход, уже отработан на GPU/CPU embeddings

**B) SQLite таблица**:
```sql
CREATE TABLE text_embeddings (
    report_id TEXT PRIMARY KEY,
    embedding_json TEXT  -- или BLOB
);
```
- Удобно для инкрементального обновления
- Но JSON/BLOB для 384 dims — неэффективно

**Рекомендация**: вариант A (NPZ), аналогично существующим embeddings.

### Inference

При inference нет sentence-transformers зависимости:
- Новый отчёт → precompute embedding → SVD transform → фичи
- Или: для inference использовать только D+E фичи (regex), embeddings только для training quality

---

## План экспериментов

### Эксперимент E1: Baseline embeddings

**Что делаем**:
1. Encode `concluding_notes` через `all-MiniLM-L6-v2` (34% покрытие)
2. SVD → 16 dims
3. NaN для отчётов без concluding_notes (как GPU embeddings)
4. Добавить к текущим D+E фичам, обучить cascade

**Ожидание**: +0.003–0.008 (низкое покрытие ограничивает)

### Эксперимент E2: All-text embeddings

**Что делаем**:
1. Encode `all_text` (конкатенация всех notes) — 91% покрытие
2. SVD → 16-32 dims
3. Обучить cascade

**Ожидание**: +0.005–0.012 (высокое покрытие)

### Эксперимент E3: Раздельные embeddings

**Что делаем**:
1. Отдельные embeddings для `concluding_notes` и `notes_verdict` (74% покрытие)
2. SVD каждого → 8-16 dims
3. Обучить cascade

**Цель**: понять, какое текстовое поле несёт больше сигнала

### Эксперимент E4: Размерность SVD

**Что делаем**:
1. Варьировать SVD dims: 8, 16, 32, 64
2. На лучшем варианте из E1-E3
3. Кривая quality vs dims

### Эксперимент E5: Комбинация с D+E

**Что делаем**:
1. Лучший вариант embeddings + D+E фичи
2. Feature importance: embeddings vs D+E — конкурируют или дополняют?
3. Ablation: embeddings без D+E vs D+E без embeddings

**Критерий успеха**: F1 > 0.70

---

## Реализация

### Шаг 1: Precompute embeddings (скрипт)

```python
# scripts/compute_text_embeddings.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
# Batch encode ~320K текстов
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
# ~25 секунд на GPU, ~10 минут на CPU
```

### Шаг 2: SVD + сохранение

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=16)
reduced = svd.fit_transform(embeddings)
np.savez("data/text_embeddings.npz",
    embeddings=reduced,
    report_ids=report_ids,
    svd_components=svd.components_,
    svd_mean=mean_vector,
)
```

### Шаг 3: Интеграция в feature matrix

В `_build_feature_matrix`: lookup report_id → embedding vector → `text_emb_0..15` фичи.
Аналогично существующим `gpu_emb_*`, `cpu_emb_*`, `game_emb_*`.

---

## Результаты экспериментов

| Эксперимент | F1 macro | Δ vs D+E baseline |
|---|---|---|
| E4: verdict SVD64 | 0.7335 | +0.038 |
| E5: concluding+verdict 16+16 | 0.7323 | +0.036 |
| **E4: verdict SVD32** | **0.7309** | **+0.035** |
| E3: verdict SVD16 | 0.7289 | +0.033 |
| E4: verdict SVD8 | 0.7238 | +0.028 |
| E2: all_text SVD16 | 0.7152 | +0.019 |
| E1: concluding SVD16 | 0.6997 | +0.004 |
| Baseline (D+E) | 0.6959 | — |

**Лучший источник**: `notes_verdict` (74% покрытие, 54 символа — концентрированный сигнал)
**Оптимальный SVD**: 32 dims (баланс качества/размерности, +0.002 от 64 dims не стоит)

### Интеграция в production

**Интегрировано** в `train_cascade_pipeline` (Step 2/9):
- `build_text_embeddings()` в `embeddings.py` — encode + SVD
- 32 text_emb_* фичи в feature matrix
- SVD components сохраняются в `embeddings.npz`

**Production результат**:
- **F1 macro: 0.729** (было 0.593 → 0.692 → 0.729)
- **borked: P=0.85, R=0.79** (было P=0.57, R=0.41)
- **ECE: 0.008** (было 0.018)
- **Confidence ≥ 0.7: 68%** данных, accuracy 90.8%
- Features: 104 → 149

---

## Риски

- **Stage 2 bottleneck — label noise**: если tinkering/oob неразличимы семантически, embeddings не помогут
- **Overfitting**: 384→16 SVD может потерять релевантный сигнал; 384 raw dims — overfitting
- **Runtime dependency**: если нужны embeddings при inference, требуется sentence-transformers в production
- **Покрытие**: 34% для concluding_notes, 91% для all_text — NaN handling критичен

## Приоритет

Средний. Ожидаемый gain скромный (+0.005–0.015), основной буст уже взят через D+E.
Стоит делать после исчерпания более дешёвых подходов (LLM features из PLAN_ML_5_LLM.md).
