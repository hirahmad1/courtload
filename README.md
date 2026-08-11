# CourtLoad — NBA Player Injury Risk Prediction

Front-office style injury risk model + interactive Streamlit dashboard.

## Problem statement

NBA front offices need to **spot injury risk early** — before a star breaks down on a back-to-back. CourtLoad predicts each player’s **14-day injury probability** from workload, schedule density, age curve, and injury history, then explains the score in plain language so coaches and analysts can act (rest, minutes cap, travel management).

## Headline result

| Metric | Value | Why it matters |
|---|---|---|
| **PR-AUC (primary)** | **0.603** | Best metric for **imbalanced** injury events |
| ROC-AUC | 0.720 | Ranking quality |
| Brier score | 0.205 | Probability calibration (after isotonic) |

Injuries are rare, so **PR-AUC is the honest headline** — not ROC-AUC alone.

## Architecture

1. **Model (brain)** — pandas + **LightGBM** + **isotonic calibration** + TreeSHAP  
2. **Web app (face)** — Streamlit `CourtLoad` dashboard (4 views)

## Quick start

```bash
cd hackathon
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_model
streamlit run app/streamlit_app.py
```

## Dashboard views (demo narrative)

Use this order for judges / video:

1. **League watchlist** — who’s at risk (sorted high → low), full Low→High spectrum  
2. **Player dossier** — plain-language why + SHAP + injury timeline + export brief  
3. **What-if scout** — change rest / workload → before/after risk delta (live demo wow)  
4. **Team roster** — color-coded roster tiles + ranking for one franchise  
5. **Compare** — side-by-side risk + SHAP for trade / load debates  

## Model features

Workload spikes, back-to-backs, rest, minutes trend, **age curve** (`age_sq`, `age_over_30`), position load, **schedule density** (`games_l30`, road games), usage, **team pace**, **efficiency-drop** fatigue proxy, prior injuries, **same-area recurrence**.

## Leakage & calibration

- Chronological train/test split by `as_of_date`
- History features are **as-of snapshot only** (no future injuries)
- Target `injured_next_14d` is never a feature
- Isotonic calibration fit on a late train slice (not the test set)

## Data

Realistic **synthetic** feature table for offline demos (hackathon-safe). Optional live enrichment:

```python
from src.data_collection import build_dataset
build_dataset(use_live_api=True)
```

Production path: join `nba_api` game logs + Kaggle / Basketball Reference injury logs on player + date.

## Project layout

```
config.py / requirements.txt
src/data_collection.py   # demo data (+ optional nba_api)
src/train_model.py       # LightGBM + calibration + SHAP
src/predict.py           # inference, summary, FO brief export
app/streamlit_app.py     # CourtLoad UI
data/processed/  models/
```
