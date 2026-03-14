# ProtonDB Game Compatibility Prediction

ML pipeline that predicts game compatibility with Proton/Wine on Linux using [ProtonDB](https://www.protondb.com/) community reports.

**Two-stage cascade LightGBM classifier** (borked vs works, then tinkering vs works_oob) trained on 350K+ user reports with 119 features including SVD embeddings, game metadata from Steam/PCGamingWiki, and text features from user notes.

Current metrics: **F1 macro = 0.72**, accuracy = 0.77.

## Architecture

```
ProtonDB dump → Worker → SQLite DB → Preprocessing → ML Training → FastAPI
                                          │
                          ┌────────────────┼────────────────┐
                          │                │                │
                    Enrichment      Normalization     LLM extraction
                  (Steam, PCGW,    (GPU/CPU/driver    (launch options,
                   ProtonDB API,    heuristic)         text analysis)
                   AWACY, GitHub)
```

**Stack:** Python 3.12, LightGBM, FastAPI, SQLite, SHAP, Click CLI.

### Key components

- **Worker** fetches ProtonDB data dumps and imports reports into SQLite
- **Preprocessing** enriches data from Steam Store, PCGamingWiki, AreWeAntiCheatYet, ProtonDB API; normalizes GPU/CPU strings; extracts structured data from text via LLM
- **ML** trains a cascade classifier with SVD embeddings (GPU/CPU/game co-occurrence matrices) and SHAP explanations
- **API** (WIP) will serve predictions with top-3 SHAP factors per game

## Setup

```bash
git clone https://github.com/getjump/protondb-game-compatibility-prediction
cd protondb-game-compatibility-prediction

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env if using a cloud LLM provider (default: local ollama)
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

# LLM-based preprocessing (requires local or cloud LLM)
protondb-settings preprocess llm normalize-gpu
protondb-settings preprocess llm normalize-cpu
protondb-settings preprocess llm parse-launch-options
protondb-settings preprocess llm extract
```

### 3. Train

```bash
protondb-settings ml train
```

### 4. Serve (WIP)

```bash
protondb-settings serve
```

## How it works

### Model

Two-stage cascade:
1. **Stage 1:** borked (0) vs works (1) -- catches broken games
2. **Stage 2:** tinkering (0) vs works_oob (1) -- for non-broken games, distinguishes "needs tweaking" from "works out of the box"

### Features (119 total)

| Group | Features | Description |
|-------|----------|-------------|
| Hardware | GPU family, CPU family, driver version, RAM, OS | From report system info |
| Temporal | Report age | Days since report submission |
| Game metadata | Engine, genre, DRM, anticheat, Deck status | From Steam, PCGamingWiki, AWACY |
| SVD embeddings | GPU (16d), CPU (16d), Game (16d) | From co-occurrence matrices |
| Text | Keywords, sentiment, note lengths | From user notes and concluding text |
| Game aggregates | Customization rates, fault rates, verdict stats | Per-game community signals |
| Text embeddings | Sentence-transformer vectors (32d) | From concluding notes |

### Label noise

The main challenge is subjective labeling on the tinkering/works_oob boundary (~15-20% noise). Mitigations:
- Label smoothing (alpha=0.15) via LightGBM cross_entropy objective
- Cleanlab noise detection and removal
- Rule-based relabeling from extracted action data

See [docs/PLAN_ML_12.md](docs/PLAN_ML_12.md) for the current research direction using contributor data (IRT, Dawid-Skene).

## Project structure

```
protondb_settings/
  api/          FastAPI server and routes
  db/           SQLite connection, migrations
  ml/           Model training, features, evaluation
  preprocessing/
    cleaning.py         Data cleaning
    normalize/          GPU/CPU/driver normalization (heuristic + LLM)
    enrichment/         External API data (Steam, PCGamingWiki, ProtonDB, AWACY, GitHub)
    llm/                LLM-based extraction (launch options, text analysis)
    extract/            Structured text extraction
  worker/       ProtonDB dump fetcher and importer
  cli.py        Click CLI entry point
  config.py     Environment-based configuration
scripts/        Experiment scripts (reproducibility)
docs/           Architecture plans and experiment logs
```

## License

[MIT](LICENSE)
