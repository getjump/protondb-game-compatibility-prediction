# Phase 22: Text embeddings upgrade + Stage 1 optimization

## Контекст

Анализ показал:
- `text_emb_2` — **#1 feature в Stage 1** (gain 1.2M, 2x от следующего)
- Текущий text pipeline: `all-MiniLM-L6-v2` (384d) → SVD 32d, только `notes_verdict`
- 68% reports без текста, SVD теряет 92% variance
- Stage 1 borked recall=0.81, FN часто имеют подробный текст (fault_notes + mentions_perfect)

## Phase 22.1 — Text embeddings: all fields + no SVD (1 день, +0.005-0.015 F1)

### Расширение text input и увеличение dimensions

**Эксперименты:**

1. **All text fields** — concat `concluding_notes + notes_verdict + notes_extra + notes_customizations + all fault notes` вместо только `notes_verdict`
2. **No SVD** — использовать полные 384 dims MiniLM вместо SVD 32
3. **All fields + no SVD** — комбинация (384 dims на rich text)
4. **Larger model** — `all-mpnet-base-v2` (768d) или `nomic-embed-text-v1.5` (768d)

**Гипотеза:** text_emb_2 уже #1 feature при 32 dims и partial text. С 384 dims на full text — значительно больше signal для Stage 1 (borked detection from text descriptions).

**Стоимость:** ~20 строк (изменение `build_text_embeddings` params). Compute: ~5 min для 112K reports.

---

## Phase 22.2 — Stage 1 threshold + class weight tuning (0.5 дня, +0.003-0.010 F1)

### Улучшение borked recall

**Факт:** borked recall=0.81. 1827 FN (borked→works). 47% FN имеют P(borked)≥0.2.

**Эксперименты:**
1. **Stage 1 threshold** — снизить borked threshold с 0.5 до 0.4/0.45 в CascadeClassifier
2. **class_weight sweep** — {0:3}→{0:4},{0:5},{0:6}
3. **Combined** — threshold + weight

---

## Phase 22.3 — Contradictory report features (0.5 дня, +0.002-0.005 F1)

### Ловить "borked but fixed" reports

**Факт:** FN borked имеют `mentions_perfect=0.16` (vs TP=0.04) и `fault_notes=1.03` (vs TP=0.40). Пользователь описывает fix для borked game → модель думает "works".

**Фичи:**
- `contradictory_report` = mentions_perfect AND fault_notes_count > 0
- `fix_described` = mentions_fix AND verdict=borked → likely borked with workaround
- `text_sentiment_mismatch` = positive text + negative faults

---

## Порядок

```
Phase 22.1 (1 день):   Text embeddings upgrade             → +0.005-0.015 F1
Phase 22.2 (0.5 дня):  Stage 1 threshold/weight tuning     → +0.003-0.010 F1
Phase 22.3 (0.5 дня):  Contradictory report features       → +0.002-0.005 F1
```
