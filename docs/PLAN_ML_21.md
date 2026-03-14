# Phase 21: Production aggregation и hierarchical prediction

## Контекст

Per-report F1=0.780. Per-(game, gpu_family) aggregated vote F1=**0.829**. Per-pair binary (borked vs works) F1=**0.902**. Модель значительно лучше при aggregation — individual report errors cancel out.

**Production use case:** пользователь спрашивает "будет ли Game X работать на моём GPU/Proton?" Ответ должен агрегировать все доступные отчёты, а не предсказывать один.

## Доступные размерности для агрегации

| Dimension | Values | Reports | Смысл |
|---|---|---|---|
| `app_id` (game) | 31K | 11.3/pair | Базовый — какая игра |
| `gpu_vendor` (nvidia/amd/intel) | 3 | — | Driver stack — NVIDIA vs Mesa |
| `variant` (official/ge/experimental/native) | 6 | — | Proton type — сильнейший Stage 2 feature |
| `is_steam_deck` (deck/desktop) | 2 | — | Form factor — Deck имеет свои issues |
| `kernel_major` (5.x/6.x) | 2 | — | Kernel generation |

## Группировки

| Группировка | Пар | Avg reports | Использование |
|---|---|---|---|
| `(game)` | 31K | 11.3 | Coarsest — "игра вообще работает?" |
| `(game, vendor)` | 52K | 6.7 | "Работает ли на NVIDIA / AMD?" |
| `(game, deck/desktop)` | 43K | 8.2 | "Работает ли на Steam Deck?" |
| `(game, variant)` | 62K | 5.6 | "Работает ли с GE-Proton / official?" |
| `(game, vendor, variant)` | ~80K | ~4.3 | Granular — "AMD + GE-Proton?" |
| `(game, vendor, deck)` | ~60K | ~5.8 | "NVIDIA desktop vs AMD Deck?" |

---

## Phase 21.1 — Multi-level aggregation evaluation (1 день)

### Сравнение группировок

**Что:** Eval per-pair F1 для каждой группировки. Найти оптимальный trade-off coverage vs accuracy.

**Метрики per группировку:**
- F1 macro / accuracy
- Coverage: % pairs с 3+ reports (reliable prediction)
- Confidence: avg max_proba в группе

---

## Phase 21.2 — Hierarchical fallback (1-2 дня)

### Каскадная агрегация от точной к грубой

**Алгоритм:**
```
Input: (game, gpu_vendor, variant, is_deck)

1. Ищем (game, vendor, variant, deck) — если 3+ reports → predict
2. Fallback: (game, vendor, deck) — если 3+ reports → predict
3. Fallback: (game, vendor) — если 3+ reports → predict
4. Fallback: (game) — всегда есть
5. Cold start (нет reports): predict из game metadata + IRT difficulty
```

**Каждый уровень:**
- Aggregated prediction (majority vote / mean proba)
- Confidence = f(n_reports, agreement)
- SHAP explanation (top-3 factors)

---

## Phase 21.3 — Temporal-aware aggregation (1 день)

### Свежие отчёты весят больше при агрегации

**Что:** При majority vote — weight by recency:
```python
weight = exp(-age_days * ln(2) / 365)  # half-life 1 year
weighted_vote = Σ weight_i * prediction_i / Σ weight_i
```

Старый "borked" от 2020 + свежий "works" от 2025 → aggregated = "works" (свежий важнее).

---

## Phase 21.4 — API response format (1 день)

### Структура ответа для production

```json
{
  "app_id": 1245620,
  "game": "ELDEN RING",
  "query": {"gpu_vendor": "nvidia", "variant": "official", "is_deck": false},

  "prediction": {
    "verdict": "works",
    "sub_verdict": "tinkering",
    "confidence": 0.87,
    "based_on": 23,
    "aggregation_level": "game_vendor"
  },

  "breakdown": {
    "borked_pct": 0.05,
    "tinkering_pct": 0.62,
    "works_oob_pct": 0.33
  },

  "factors": [
    {"feature": "variant", "impact": "+0.15", "detail": "official Proton well-supported"},
    {"feature": "irt_difficulty", "impact": "-0.08", "detail": "game has moderate difficulty score"},
    {"feature": "gpu_family", "impact": "+0.05", "detail": "NVIDIA well-tested"}
  ],

  "recommendation": {
    "proton_version": "Proton 9.0-3",
    "notes": "Most reports use official Proton. Consider GE-Proton for better compatibility."
  }
}
```

---

## Phase 21.5 — Aggregated model (2 дня, experimental)

### Модель обученная на aggregated pairs вместо individual reports

**Что:** Вместо 348K individual reports → 52K (game, vendor) pairs. Каждая pair:
- Features: game metadata + avg hardware features + IRT difficulty + report statistics
- Target: majority verdict

**Гипотеза:** Eliminates per-report noise entirely. Model learns game-level compatibility.

**Риск:** Меньше данных (52K vs 348K), потеря per-hardware granularity.

---

## Порядок

```
Phase 21.1 (1 день):   Multi-level aggregation eval        → evaluation
Phase 21.2 (1-2 дня):  Hierarchical fallback               → production algorithm
Phase 21.3 (1 день):   Temporal-aware aggregation           → better predictions
Phase 21.4 (1 день):   API response format                 → production readiness
Phase 21.5 (2 дня):    Aggregated model                    → experimental
```

## Ключевой вывод

Per-report F1=0.78 → per-pair F1=0.83 → per-pair binary F1=0.90.

Модель уже достаточно хороша для production. Вопрос не "как улучшить модель" а "как правильно агрегировать и отдать результат пользователю". Hierarchical aggregation + temporal weighting + confidence scoring = production-ready prediction API.
