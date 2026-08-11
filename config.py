"""Project-wide paths and model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Feature columns used by the model (must stay in sync across train / predict / app)
FEATURE_COLS = [
    "age",
    "age_sq",  # non-linear age curve
    "age_over_30",  # sharp risk lift after 30
    "is_big",  # C / PF — higher contact load
    "is_guard",
    "minutes_l7",
    "minutes_l14",
    "minutes_l30",
    "minutes_spike_7v30",  # recent load vs baseline
    "games_l7",
    "games_l14",
    "games_l30",  # schedule density
    "road_games_l14",  # travel / road-trip load
    "back_to_backs_l14",
    "avg_rest_days_l14",
    "minutes_trend_slope",  # rising minutes = fatigue risk
    "season_minutes_total",
    "games_played_season",
    "usage_proxy",  # touches / intensity per minute
    "team_pace",  # distance proxy per possession
    "efficiency_drop",  # rest-adjusted fatigue proxy
    "prior_injuries_365d",
    "days_since_last_injury",
    "injury_prone_flag",
    "recurrence_same_area",  # same body-part reinjury risk
]

TARGET_COL = "injured_next_14d"

# Rough team pace index (possessions / 100) for demo realism
TEAM_PACE = {
    "ATL": 102.5,
    "BOS": 99.0,
    "CHA": 100.5,
    "CHI": 98.5,
    "CLE": 97.5,
    "DAL": 99.5,
    "DEN": 98.0,
    "DET": 100.0,
    "GSW": 103.0,
    "HOU": 99.0,
    "IND": 104.0,
    "LAC": 98.5,
    "LAL": 100.0,
    "MEM": 101.5,
    "MIA": 97.0,
    "MIL": 101.0,
    "MIN": 99.5,
    "NOP": 100.5,
    "NYK": 97.5,
    "OKC": 101.0,
    "ORL": 98.0,
    "PHI": 97.5,
    "PHX": 99.0,
    "POR": 98.5,
    "SAC": 102.0,
    "SAS": 100.0,
    "TOR": 99.5,
    "UTA": 100.5,
    "WAS": 101.5,
}

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 300,
    "verbosity": -1,
    "random_state": 42,
}

RISK_BANDS = [
    (0.0, 0.25, "Low", "#2ecc71"),
    (0.25, 0.45, "Moderate", "#f1c40f"),
    (0.45, 0.65, "Elevated", "#e67e22"),
    (0.65, 1.01, "High", "#e74c3c"),
]
