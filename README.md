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

50+ experiments across Phases 11-22. See [docs/](docs/) for detailed plans and results.

| Phase | Approach | ΔF1 | Outcome |
|---|---|---|---|
| 11.2 | Alternative models (CatBoost, XGBoost) | −0.002 | LightGBM optimal |
| **12.8** | **IRT features** | **+0.030** | **Decompose annotator bias** |
| **13.2** | **Contributor-aware relabeling** | **+0.017** | **Replace Cleanlab + Phase 8** |
| 14 | Steam PICS features | 0.000 | Redundant |
| 15 | Temporal features, FM | −0.003..−0.017 | report_age_days sufficient |
| **16** | **Class weight + error features** | **+0.006** | **Compensate class shift** |
| **17** | **HP tuning (reg=1.0, oob_w=1.8)** | **+0.004** | **Stronger regularization** |
| 18 | Threshold opt, focal loss, adaptive smooth | 0.000 | Already calibrated |
| 19 | LLM verdict inference, data filtering | −0.001 | IRT already optimal |
| **21** | **Per-game aggregation** | **+0.091** | **Majority vote at inference** |
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
