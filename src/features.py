"""Feature helpers shared by training and the Streamlit app."""
from __future__ import annotations

import pandas as pd

from config import FEATURE_COLS, TARGET_COL


def load_feature_table(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def xy_split(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def risk_band(prob: float) -> tuple[str, str]:
    from config import RISK_BANDS

    for lo, hi, label, color in RISK_BANDS:
        if lo <= prob < hi:
            return label, color
    return "High", "#e74c3c"


def humanize_feature(name: str) -> str:
    mapping = {
        "age": "Age",
        "age_sq": "Age curve (squared)",
        "age_over_30": "Years over age 30",
        "is_big": "Big-man position (C/PF)",
        "is_guard": "Guard position",
        "minutes_l7": "Minutes (last 7 days)",
        "minutes_l14": "Minutes (last 14 days)",
        "minutes_l30": "Minutes (last 30 days)",
        "minutes_spike_7v30": "Minutes spike (7d vs 30d)",
        "games_l7": "Games (last 7 days)",
        "games_l14": "Games (last 14 days)",
        "games_l30": "Games (last 30 days)",
        "road_games_l14": "Road games (last 14 days)",
        "back_to_backs_l14": "Back-to-backs (last 14 days)",
        "avg_rest_days_l14": "Avg rest days (last 14 days)",
        "minutes_trend_slope": "Minutes trend (rising load)",
        "season_minutes_total": "Season minutes total",
        "games_played_season": "Games played (season)",
        "usage_proxy": "Usage intensity",
        "team_pace": "Team pace",
        "efficiency_drop": "Recent efficiency drop (fatigue)",
        "prior_injuries_365d": "Prior injuries (12 months)",
        "days_since_last_injury": "Days since last injury",
        "injury_prone_flag": "Injury-prone history flag",
        "recurrence_same_area": "Same-area injury recurrence",
    }
    return mapping.get(name, name.replace("_", " ").title())
