"""Inference + per-player SHAP explanations + plain-language summaries."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLS, MODELS_DIR, TEAM_PACE
from src.features import humanize_feature, risk_band


def load_model():
    path = MODELS_DIR / "injury_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train_model")
    return joblib.load(path)


def load_calibrator():
    path = MODELS_DIR / "probability_calibrator.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_explainer():
    path = MODELS_DIR / "shap_explainer.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _calibrate(raw: np.ndarray | float, calibrator=None) -> np.ndarray | float:
    calibrator = calibrator if calibrator is not None else load_calibrator()
    if calibrator is None:
        return raw
    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    out = np.asarray(calibrator.transform(arr), dtype=float)
    return float(out[0]) if np.ndim(raw) == 0 else out


def enrich_derived_fields(row: pd.Series) -> pd.Series:
    """Fill age-curve / defaults so older snapshots still score after feature upgrades."""
    out = row.copy()
    age = int(out.get("age", 28))
    out["age_sq"] = float(out["age_sq"]) if "age_sq" in out.index and pd.notna(out.get("age_sq")) else (age ** 2) / 100.0
    out["age_over_30"] = float(out["age_over_30"]) if "age_over_30" in out.index and pd.notna(out.get("age_over_30")) else float(max(age - 30, 0))
    if "games_l30" not in out.index or pd.isna(out.get("games_l30")):
        out["games_l30"] = int(out.get("games_l14", 6)) + 5
    if "road_games_l14" not in out.index or pd.isna(out.get("road_games_l14")):
        out["road_games_l14"] = int(max(int(out.get("games_l14", 6)) // 2, 0))
    if "team_pace" not in out.index or pd.isna(out.get("team_pace")):
        out["team_pace"] = float(TEAM_PACE.get(str(out.get("team", "")), 99.0))
    if "efficiency_drop" not in out.index or pd.isna(out.get("efficiency_drop")):
        slope = float(out.get("minutes_trend_slope", 0) or 0)
        out["efficiency_drop"] = float(np.clip(0.04 + 0.08 * max(slope, 0), 0, 0.4))
    if "recurrence_same_area" not in out.index or pd.isna(out.get("recurrence_same_area")):
        out["recurrence_same_area"] = int(int(out.get("prior_injuries_365d", 0)) >= 2)
    return out


def build_manual_snapshot(
    *,
    player_name: str,
    age: int,
    position: str,
    minutes_l7: float,
    minutes_l14: float,
    back_to_backs_l14: int,
    avg_rest_days_l14: float,
    prior_injuries_365d: int,
    days_since_last_injury: int,
    minutes_l30: float | None = None,
    games_l7: int = 3,
    games_l14: int = 6,
    games_l30: int | None = None,
    road_games_l14: int | None = None,
    season_minutes_total: float | None = None,
    games_played_season: int = 50,
    usage_proxy: float = 0.75,
    team: str = "WHAT-IF",
    team_pace: float | None = None,
    efficiency_drop: float = 0.05,
    recurrence_same_area: int | None = None,
) -> pd.Series:
    """Build a model-ready feature row from the what-if form."""
    pos = position.upper().strip()
    is_big = int(pos in {"C", "PF"})
    is_guard = int(pos in {"PG", "SG"})

    minutes_l30 = float(minutes_l30) if minutes_l30 is not None else float(minutes_l14) * 0.95
    minutes_l30 = max(minutes_l30, 1.0)
    minutes_spike_7v30 = (float(minutes_l7) / minutes_l30) - 1.0
    minutes_trend_slope = (float(minutes_l7) - float(minutes_l14)) / 7.0
    if season_minutes_total is None:
        season_minutes_total = float(games_played_season) * float(minutes_l14)
    injury_prone_flag = int(prior_injuries_365d >= 3)
    if games_l30 is None:
        games_l30 = int(games_l14) + 5
    if road_games_l14 is None:
        road_games_l14 = int(games_l14) // 2
    if team_pace is None:
        team_pace = float(TEAM_PACE.get(team, 99.0))
    if recurrence_same_area is None:
        recurrence_same_area = int(prior_injuries_365d >= 2)

    return pd.Series(
        {
            "player_name": player_name.strip() or "Custom player",
            "team": team,
            "position": pos,
            "age": int(age),
            "age_sq": (int(age) ** 2) / 100.0,
            "age_over_30": float(max(int(age) - 30, 0)),
            "is_big": is_big,
            "is_guard": is_guard,
            "minutes_l7": float(minutes_l7),
            "minutes_l14": float(minutes_l14),
            "minutes_l30": float(minutes_l30),
            "minutes_spike_7v30": float(minutes_spike_7v30),
            "games_l7": int(games_l7),
            "games_l14": int(games_l14),
            "games_l30": int(games_l30),
            "road_games_l14": int(road_games_l14),
            "back_to_backs_l14": int(back_to_backs_l14),
            "avg_rest_days_l14": float(avg_rest_days_l14),
            "minutes_trend_slope": float(minutes_trend_slope),
            "season_minutes_total": float(season_minutes_total),
            "games_played_season": int(games_played_season),
            "usage_proxy": float(usage_proxy),
            "team_pace": float(team_pace),
            "efficiency_drop": float(efficiency_drop),
            "prior_injuries_365d": int(prior_injuries_365d),
            "days_since_last_injury": int(days_since_last_injury),
            "injury_prone_flag": injury_prone_flag,
            "recurrence_same_area": int(recurrence_same_area),
        }
    )


def predict_risk(row: pd.Series | dict, model=None, calibrator=None) -> float:
    model = model or load_model()
    if isinstance(row, dict):
        row = pd.Series(row)
    row = enrich_derived_fields(row)
    X = pd.DataFrame([row[FEATURE_COLS].astype(float)])
    raw = float(model.predict_proba(X)[0, 1])
    return float(_calibrate(raw, calibrator))


def explain_player(row: pd.Series | dict, model=None, explainer=None, top_k: int = 8) -> pd.DataFrame:
    model = model or load_model()
    explainer = explainer or load_explainer()
    if isinstance(row, dict):
        row = pd.Series(row)
    row = enrich_derived_fields(row)
    X = pd.DataFrame([row[FEATURE_COLS].astype(float)])

    if explainer is not None:
        try:
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            values = np.asarray(sv).reshape(-1)
        except Exception:
            values = _fallback_contrib(model, X)
    else:
        values = _fallback_contrib(model, X)

    out = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "feature_label": [humanize_feature(c) for c in FEATURE_COLS],
            "value": X.iloc[0].values,
            "shap": values,
        }
    )
    out["abs_shap"] = out["shap"].abs()
    out["direction"] = np.where(out["shap"] >= 0, "Increases risk", "Decreases risk")
    return out.sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)


def _reason_phrase(feature: str, value: float) -> str:
    if feature == "back_to_backs_l14":
        n = int(value)
        return f"{n} back-to-back{'s' if n != 1 else ''} in 14 days"
    if feature == "minutes_spike_7v30":
        return f"a {value:+.0%} workload spike (7d vs 30d baseline)"
    if feature == "minutes_l7":
        return f"heavy recent minutes ({value:.0f} in the last 7 days)"
    if feature == "minutes_l14":
        return f"elevated 14-day minutes ({value:.0f})"
    if feature == "minutes_trend_slope":
        return "a rising minutes trend"
    if feature == "prior_injuries_365d":
        n = int(value)
        return f"{n} prior injur{'ies' if n != 1 else 'y'} in the past year"
    if feature == "days_since_last_injury":
        return f"only {int(value)} days since the last injury"
    if feature == "injury_prone_flag":
        return "an injury-prone history flag"
    if feature == "age":
        return f"age-related load risk ({int(value)})"
    if feature == "age_over_30":
        return f"{value:.0f} years past age-30 load curve"
    if feature == "age_sq":
        return "a steep age-curve effect"
    if feature == "is_big":
        return "big-man contact load (C/PF)"
    if feature == "avg_rest_days_l14":
        return f"limited rest ({value:.1f} avg days between games)"
    if feature == "usage_proxy":
        return "high usage intensity"
    if feature == "season_minutes_total":
        return f"high season minutes ({value:.0f})"
    if feature == "games_l30":
        return f"dense schedule ({int(value)} games in 30 days)"
    if feature == "road_games_l14":
        return f"{int(value)} road games in 14 days"
    if feature == "team_pace":
        return f"high team pace ({value:.1f})"
    if feature == "efficiency_drop":
        return "a recent efficiency drop (fatigue signal)"
    if feature == "recurrence_same_area":
        return "same-area injury recurrence risk"
    return humanize_feature(feature).lower()


def plain_language_summary(
    player_name: str,
    risk: float,
    band: str,
    shap_df: pd.DataFrame,
) -> str:
    ups = shap_df[shap_df["shap"] > 0].sort_values("shap", ascending=False).head(2)
    if ups.empty:
        return (
            f"{player_name} is at {band.upper()} risk ({risk:.0%} chance of injury "
            f"in the next 14 days), with no single dominant load driver."
        )
    phrases = [_reason_phrase(r.feature, float(r.value)) for r in ups.itertuples()]
    due = phrases[0] if len(phrases) == 1 else f"{phrases[0]} and {phrases[1]}"
    return f"{player_name} is at {band.upper()} risk primarily due to {due}."


def format_front_office_report(
    row: pd.Series,
    risk: float,
    band: str,
    shap_df: pd.DataFrame,
) -> str:
    briefing = plain_language_summary(str(row.get("player_name", "Player")), risk, band, shap_df)
    lines = [
        "COURTLOAD — Front Office Injury Brief",
        "=" * 42,
        briefing,
        "",
        f"Player: {row.get('player_name')}",
        f"Team / Pos / Age: {row.get('team')} / {row.get('position')} / {row.get('age')}",
        f"Calibrated 14-day risk: {risk:.1%} ({band})",
        "",
        "Workload snapshot",
        f"- Minutes L7 / L14 / L30: {row.get('minutes_l7'):.1f} / {row.get('minutes_l14'):.1f} / {row.get('minutes_l30'):.1f}",
        f"- Spike 7d vs 30d: {float(row.get('minutes_spike_7v30', 0)):+.0%}",
        f"- Back-to-backs L14: {int(row.get('back_to_backs_l14', 0))}",
        f"- Avg rest L14: {float(row.get('avg_rest_days_l14', 0)):.1f} days",
        f"- Road games L14: {int(row.get('road_games_l14', 0))} | Games L30: {int(row.get('games_l30', 0))}",
        "",
        "History",
        f"- Prior injuries (12mo): {int(row.get('prior_injuries_365d', 0))}",
        f"- Days since last injury: {int(row.get('days_since_last_injury', 0))}",
        f"- Same-area recurrence: {int(row.get('recurrence_same_area', 0))}",
        "",
        "Top SHAP drivers",
    ]
    for r in shap_df.head(5).itertuples():
        direction = "↑ risk" if r.shap >= 0 else "↓ risk"
        lines.append(f"- {r.feature_label}: {r.shap:+.3f} ({direction}), value={r.value:.2f}")
    lines.extend(["", "Generated by CourtLoad · for load-management discussion only."])
    return "\n".join(lines)


def _fallback_contrib(model, X: pd.DataFrame) -> np.ndarray:
    imp = getattr(model, "feature_importances_", np.ones(len(FEATURE_COLS)))
    z = (X.iloc[0].values - X.iloc[0].values.mean()) / (X.iloc[0].values.std() + 1e-6)
    return imp * z


def score_frame(df: pd.DataFrame, model=None, calibrator=None) -> pd.DataFrame:
    model = model or load_model()
    calibrator = calibrator if calibrator is not None else load_calibrator()
    out = df.copy()
    enriched = out.apply(enrich_derived_fields, axis=1)
    raw = model.predict_proba(enriched[FEATURE_COLS].astype(float))[:, 1]
    out["injury_risk"] = _calibrate(raw, calibrator)
    out["risk_band"] = out["injury_risk"].apply(lambda p: risk_band(p)[0])
    out["risk_color"] = out["injury_risk"].apply(lambda p: risk_band(p)[1])
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = enriched[col]
        else:
            out[col] = enriched[col]
    return out.sort_values("injury_risk", ascending=False)
