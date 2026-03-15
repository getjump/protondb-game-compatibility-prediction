# ProtonDB Game Compatibility Prediction

ML pipeline that predicts game compatibility with Proton/Wine on Linux using [ProtonDB](https://www.protondb.com/) community reports.

**Two-stage cascade LightGBM classifier** with IRT-based label denoising, trained on 350K+ user reports with 123 features.

### Results

| Evaluation | 3-class F1 | Binary F1 | Accuracy |
|---|---|---|---|
| Per-report | 0.780 | 0.906 | 0.828 |
| **Per-game (production)** | **0.871** | **0.943** | **0.934** |

Per-game predictions aggregate individual report predictions via majority vote — individual errors cancel out.

## Architecture

```
ProtonDB dump → Worker → SQLite DB → Preprocessing → ML Training → Prediction
                                          │
                          ┌────────────────┼────────────────┐
                          │                │                │
                    Enrichment      Normalization     LLM extraction
                  (Steam, PCGW,    (GPU/CPU/driver    (launch options,
                   ProtonDB API,    heuristic)         text analysis)
                   Steam PICS,
                   AWACY, GitHub)
```

**Stack:** Python 3.12, LightGBM, FastAPI, SQLite, SHAP, Click CLI.

### Key components

- **Worker** fetches ProtonDB data dumps and imports reports into SQLite
- **Preprocessing** enriches data from Steam Store, Steam PICS (bulk CM protocol), PCGamingWiki, AreWeAntiCheatYet, ProtonDB API (contributor data); normalizes GPU/CPU strings; extracts structured data from text via LLM
- **ML** trains a cascade classifier with IRT denoising, SVD embeddings, and per-game aggregation
- **IRT (Item Response Theory)** decomposes subjective tinkering/oob labels into per-annotator strictness and per-game difficulty — the key innovation (+0.030 F1)

### Key innovations

1. **IRT label denoising** — 1PL Item Response Theory separates annotator bias (θ) from game difficulty (d). Resolves the subjective tinkering↔works_oob boundary that causes 15-20% label noise. (+0.030 F1)
2. **Contributor-aware relabeling** — replaces Cleanlab + rule-based heuristics with IRT-informed relabeling based on annotator strictness. (+0.017 F1)
3. **Per-game majority vote** — aggregates per-report predictions at inference time. Errors cancel out, boosting F1 from 0.780 to 0.871. (+0.091 F1)

## Setup

```bash
git clone https://github.com/getjump/protondb-game-compatibility-prediction
cd protondb-game-compatibility-prediction

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env — configure LLM backend (local ollama, OpenRouter, or Claude CLI)
```

## Usage

### 1. Import data

```bash
# Download and import the latest ProtonDB dump
protondb-settings worker check
protondb-settings worker sync
```

### 2. Preprocess

```bash
# Cleaning + heuristic normalization + enrichment (no LLM required)
protondb-settings preprocess run

# Enrichment from additional sources
protondb-settings preprocess run --step enrichment --source protondb_reports
protondb-settings preprocess run --step enrichment --source steam_pics

# LLM-based preprocessing (requires local or cloud LLM)
protondb-settings preprocess llm normalize-gpu
protondb-settings preprocess llm normalize-cpu
protondb-settings preprocess llm parse-launch-options
protondb-settings preprocess llm extract

# LLM backends: --backend openai (default), openrouter, or claude-cli
protondb-settings preprocess llm --backend openrouter extract
```

### 3. Train

```bash
protondb-settings ml train-cascade
# Reuse Stage 1 for faster iteration on Stage 2:
protondb-settings ml train-cascade --reuse-stage1 data/model_stage1.pkl
```

### 4. Serve (WIP)

```bash
protondb-settings serve
```

## How it works

### Model

Two-stage cascade with IRT denoising:

1. **Stage 1:** borked (0) vs works (1) — catches broken games (F1=0.846)
2. **Stage 2:** tinkering (0) vs works_oob (1) — for non-broken games, distinguishes "needs tweaking" from "works out of the box" (F1=0.880 tinkering, 0.614 oob)
3. **IRT fitting:** decomposes annotator strictness (θ) and game difficulty (d) from contributor data
4. **Contributor-aware relabeling:** corrects labels from strict annotators
5. **Per-game aggregation:** majority vote at inference time

### Features (123 total)

| Group | Features | Description |
|-------|----------|-------------|
| Hardware | GPU family, driver versions, APU/iGPU flags | From report system info |
| Temporal | Report age | Days since report submission |
| Game metadata | Engine, genre, DRM, anticheat, Deck status | From Steam, PCGamingWiki, AWACY |
| SVD embeddings | GPU (16-20d), Game (16-20d) | From co-occurrence matrices |
| Text | Keywords, sentiment, note lengths | From user notes |
| Text embeddings | Sentence-transformer SVD (32d) | From verdict notes |
| Game aggregates | Customization rates, fault rates | Per-game community signals |
| **IRT features** | Game difficulty, contributor strictness | From Item Response Theory |
| **Error features** | Contributor consistency, game agreement | Per-annotator and per-game stats |
| Proton variant | official/ge/experimental/native | Runner type (top Stage 2 feature) |

### Label noise and IRT

The main challenge is subjective labeling on the tinkering/works_oob boundary (~15-20% noise). Different users have different standards — choosing GE-Proton is "tinkering" for strict users but "works_oob" for lenient ones.

**IRT solution:** Fit a 1PL Item Response Theory model on contributor×game interactions:
- **θ (theta):** per-contributor strictness — how likely they are to say "tinkering"
- **d (difficulty):** per-(game, GPU family) objective difficulty
- **P(tinkering) = σ(θ - d):** separates annotator bias from game reality

IRT features are the #2 and #4 most important features in Stage 2. Combined with contributor-aware relabeling, IRT provides +0.047 F1 improvement.

### Experiment history

50+ experiments across Phases 1-22. See [docs/](docs/) for detailed plans and results.

#### Foundation (Phases 1-8)

| Phase | Approach | Result |
|---|---|---|
| 1-3 | Python foundation, worker, preprocessing pipeline | Infrastructure |
| 4 | Single LightGBM model → cascade (borked/works → tinkering/oob) | F1=0.593 → cascade +0.009 |
| 5 | SVD embeddings (GPU×Game, CPU×Game co-occurrence) | +0.010 F1 |
| 6 | Game metadata enrichment (Steam, PCGamingWiki, AWACY, GitHub) | +0.024 F1 |
| 7 | Text features (keyword regex, sentiment, note lengths) | +0.008 F1 |
| 8 | Rule-based relabeling (tinkering→oob if no effort markers) | +0.010 F1 |

#### Noise reduction (Phases 9-10)

| Phase | Approach | ΔF1 | Outcome |
|---|---|---|---|
| 9.1 | Label smoothing α=0.15 (cross_entropy objective) | +0.008 | Noise-robust training |
| 9.2 | Per-game aggregate features (26 features) | +0.024 | Community signals |
| 9.3 | Cleanlab noise removal (3% of train) | +0.021 | Confident learning |
| 9.4 | Ordinal classification, distillation, focal loss | 0.000 | All negative (6 experiments) |
| 9.5 | Feature combinations, target encoding | −0.002 | All negative (8 experiments) |
| 10 | Text embeddings (sentence-transformers SVD 32d) | +0.005 | Partial text coverage (32%) |

#### IRT breakthrough (Phases 11-13) — main contribution

| Phase | Approach | ΔF1 | Outcome |
|---|---|---|---|
| 11.2 | Alternative models (CatBoost, XGBoost, HistGBM) | −0.002 | LightGBM optimal |
| **12.8** | **IRT features (game difficulty + annotator strictness)** | **+0.030** | **Key innovation** |
| 12.1-12.3 | Contributor features, sample weighting | −0.002 | IRT dominates |
| 13.1 | IRT-only relabeling (replaces Phase 8 + Cleanlab) | +0.013 | Simpler, better |
| **13.2** | **Contributor-aware relabeling by annotator θ** | **+0.017** | **Graduated relabeling** |
| 13.3 | Hybrid pipeline, confidence weighting | −0.001 | Weighting hurts |
| 13.4-13.5 | Iterative IRT, annotator SVD embeddings | +0.001 | Marginal |

#### Optimization attempts (Phases 14-19) — mostly negative

| Phase | Approach | ΔF1 | Outcome |
|---|---|---|---|
| 14 | Steam PICS features (runtime, deck tests, review) | 0.000 | Redundant with existing |
| 15 | Temporal features, Proton×Game SVD, Factorization Machines | −0.003..−0.017 | report_age_days sufficient |
| **16** | **Class weight 1.8x oob + error features** | **+0.006** | **Compensate temporal shift** |
| **17** | **HP tuning (reg=1.0) + ensemble** | **+0.004** | **Stronger regularization** |
| 18 | Threshold optimization, focal loss, adaptive soft labels | 0.000 | Already calibrated |
| 19 | LLM verdict inference (OpenRouter), data filtering | −0.001 | IRT already optimal |

#### Task reformulation (Phases 20-22) — production framing

| Phase | Approach | ΔF1 | Outcome |
|---|---|---|---|
| 20 | Per-(game, gpu) aggregated evaluation | — | F1 0.780→0.829 per-pair |
| **21** | **Per-game majority vote aggregation** | **+0.091** | **Production metric: F1=0.871** |
| 21.5 | Aggregated model (trained on pairs) | — | Leakage confirmed |
| 21.7 | Cold-start model (metadata only) | — | Binary F1=0.601 |
| 22 | Text embeddings upgrade, Stage 1 tuning | 0.000 | Pipeline saturated |

## Project structure

```
protondb_settings/
  api/          FastAPI server and routes
  db/           SQLite connection, migrations
  ml/
    irt.py            IRT fitting, features, contributor-aware relabeling
    aggregate.py      Per-game aggregated prediction
    models/           Cascade classifier (Stage 1 + Stage 2)
    features/         Feature engineering (embeddings, encoding, game aggregates)
    train.py          Training pipeline
    predict.py        Single-sample prediction
    noise.py          Cleanlab noise detection (legacy, replaced by IRT)
    relabeling.py     Rule-based relabeling (legacy, replaced by IRT)
  preprocessing/
    cleaning.py         Data cleaning
    normalize/          GPU/CPU/driver normalization (heuristic + LLM)
    enrichment/         External API data (Steam, Steam PICS, PCGamingWiki,
                        ProtonDB, ProtonDB Reports, AWACY, GitHub)
    llm/                LLM client (OpenAI, OpenRouter, Claude CLI backends)
    extract/            Structured text extraction + verdict inference
  worker/       ProtonDB dump fetcher and importer
  cli.py        Click CLI entry point
  config.py     Environment-based configuration
scripts/        Experiment scripts (50+ experiments, reproducibility)
docs/           Architecture plans, ML experiment logs, research notes
```

## License

[MIT](LICENSE)
