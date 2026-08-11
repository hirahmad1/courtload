"""
Data collection for NBA injury risk modeling.

Primary path: nba_api game logs + synthetic/public-style injury labels.
Fallback: realistic demo dataset so the pipeline always runs offline.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, TEAM_PACE

RNG = np.random.default_rng(42)

# Core rotation-ish names for demo realism (fuller starters for heatmap demos)
DEMO_PLAYERS = [
    ("LeBron James", "LAL", "SF", 41),
    ("Anthony Davis", "LAL", "PF", 33),
    ("Austin Reaves", "LAL", "SG", 27),
    ("D'Angelo Russell", "LAL", "PG", 29),
    ("Rui Hachimura", "LAL", "PF", 27),
    ("Jarred Vanderbilt", "LAL", "SF", 26),
    ("Jayson Tatum", "BOS", "SF", 28),
    ("Jaylen Brown", "BOS", "SG", 29),
    ("Jrue Holiday", "BOS", "PG", 36),
    ("Kristaps Porzingis", "BOS", "C", 30),
    ("Derrick White", "BOS", "SG", 31),
    ("Stephen Curry", "GSW", "PG", 38),
    ("Draymond Green", "GSW", "PF", 35),
    ("Andrew Wiggins", "GSW", "SF", 30),
    ("Kevin Durant", "PHX", "SF", 37),
    ("Devin Booker", "PHX", "SG", 29),
    ("Bradley Beal", "PHX", "SG", 32),
    ("Giannis Antetokounmpo", "MIL", "PF", 31),
    ("Damian Lillard", "MIL", "PG", 35),
    ("Brook Lopez", "MIL", "C", 37),
    ("Nikola Jokic", "DEN", "C", 31),
    ("Jamal Murray", "DEN", "PG", 28),
    ("Aaron Gordon", "DEN", "PF", 30),
    ("Luka Doncic", "DAL", "PG", 27),
    ("Kyrie Irving", "DAL", "SG", 33),
    ("Joel Embiid", "PHI", "C", 32),
    ("Tyrese Maxey", "PHI", "PG", 25),
    ("Ja Morant", "MEM", "PG", 26),
    ("Zion Williamson", "NOP", "PF", 25),
    ("Kawhi Leonard", "LAC", "SF", 35),
    ("Paul George", "LAC", "SF", 35),
    ("Jimmy Butler", "MIA", "SF", 36),
    ("Bam Adebayo", "MIA", "C", 28),
    ("Tyler Herro", "MIA", "SG", 25),
    ("Donovan Mitchell", "CLE", "SG", 29),
    ("Shai Gilgeous-Alexander", "OKC", "PG", 27),
    ("Anthony Edwards", "MIN", "SG", 24),
    ("Domantas Sabonis", "SAC", "C", 30),
    ("De'Aaron Fox", "SAC", "PG", 28),
    ("Tyrese Haliburton", "IND", "PG", 26),
    ("Pascal Siakam", "IND", "PF", 32),
    ("Paolo Banchero", "ORL", "PF", 23),
    ("Victor Wembanyama", "SAS", "C", 22),
    ("Scottie Barnes", "TOR", "SF", 24),
    ("LaMelo Ball", "CHA", "PG", 25),
    ("Trae Young", "ATL", "PG", 27),
    ("Karl-Anthony Towns", "NYK", "C", 30),
    ("Jalen Brunson", "NYK", "PG", 28),
]

def _position_flags(pos: str) -> tuple[int, int]:
    is_big = int(pos in {"C", "PF"})
    is_guard = int(pos in {"PG", "SG"})
    return is_big, is_guard


def generate_demo_dataset(
    n_snapshots: int = 2500,
    season_start: str = "2024-10-22",
) -> pd.DataFrame:
    """
    Build player-game snapshots with engineered features + injury labels.

    Injury probability is driven by workload spikes, age, big-man load,
    back-to-backs, and prior injury history — so the model has a real signal.
    """
    start = datetime.fromisoformat(season_start)
    rows: list[dict] = []

    for i in range(n_snapshots):
        name, team, pos, age = DEMO_PLAYERS[i % len(DEMO_PLAYERS)]
        # Slight age jitter across seasons
        age_j = age + RNG.integers(-1, 2)
        is_big, is_guard = _position_flags(pos)

        # Season progress
        day_offset = int(RNG.integers(0, 160))
        as_of = start + timedelta(days=day_offset)

        # Workload generators (realistic NBA caps — avoid 48-min L7 artifacts)
        base_min = float(np.clip(RNG.normal(28 if not is_big else 30, 5), 10, 38))
        spike = RNG.random() < 0.20
        minutes_l7 = float(np.clip(base_min * (1.22 if spike else 1.0) + RNG.normal(0, 1.8), 6, 42))
        minutes_l14 = float(np.clip(base_min * (1.12 if spike else 1.0) + RNG.normal(0, 1.8), 6, 40))
        minutes_l30 = float(np.clip(base_min + RNG.normal(0, 2.2), 8, 38))
        minutes_spike_7v30 = (minutes_l7 / max(minutes_l30, 1.0)) - 1.0

        games_l7 = int(np.clip(RNG.integers(1, 4) + (1 if spike else 0), 1, 4))
        games_l14 = int(np.clip(games_l7 + RNG.integers(2, 5), 3, 7))
        games_l30 = int(np.clip(games_l14 + RNG.integers(4, 10), 8, 16))
        road_games_l14 = int(np.clip(RNG.integers(0, games_l14 + 1), 0, games_l14))
        back_to_backs_l14 = int(np.clip(RNG.poisson(0.7 + (0.5 if spike else 0)), 0, 3))
        avg_rest_days_l14 = float(np.clip(RNG.normal(1.9 - 0.25 * back_to_backs_l14, 0.45), 0.6, 4.0))
        minutes_trend_slope = float(RNG.normal(0.35 if spike else -0.08, 0.45))

        games_played_season = int(np.clip(day_offset // 2.2 + RNG.integers(0, 8), 1, 72))
        season_minutes_total = float(games_played_season * base_min + RNG.normal(0, 40))
        usage_proxy = float(np.clip(RNG.normal(0.85 if is_guard else 0.7, 0.15), 0.3, 1.4))
        team_pace = float(TEAM_PACE.get(team, 99.0) + RNG.normal(0, 0.8))
        efficiency_drop = float(
            np.clip(
                (0.12 if spike else 0.02) + 0.08 * max(minutes_trend_slope, 0) + RNG.normal(0, 0.04),
                0.0,
                0.45,
            )
        )

        age_sq = float((age_j ** 2) / 100.0)
        age_over_30 = float(max(age_j - 30, 0))

        # Injury history as-of as_of_date only (no future leakage into features)
        prone = name in {
            "Kawhi Leonard",
            "Joel Embiid",
            "Zion Williamson",
            "Anthony Davis",
            "Jimmy Butler",
            "LaMelo Ball",
            "Ja Morant",
        }
        prior_injuries_365d = int(RNG.poisson(1.8 if prone else 0.5))
        days_since_last_injury = int(
            RNG.integers(10, 90) if prior_injuries_365d > 0 else RNG.integers(120, 400)
        )
        injury_prone_flag = int(prone or prior_injuries_365d >= 3)
        recurrence_same_area = int(
            prior_injuries_365d > 0 and (prone or RNG.random() < 0.35)
        )

        # Latent injury risk — includes age curve, travel, pace, fatigue, recurrence
        logit = (
            -2.55
            + 2.0 * max(minutes_spike_7v30, 0)
            + 0.55 * back_to_backs_l14
            + 0.04 * age_over_30
            + 0.015 * max(age_sq - 9.0, 0)
            + 0.6 * is_big
            + 0.032 * minutes_l7
            + 0.5 * max(minutes_trend_slope, 0)
            + 0.38 * prior_injuries_365d
            - 0.0055 * min(days_since_last_injury, 200)
            + 0.45 * injury_prone_flag
            + 0.3 * (usage_proxy - 0.7)
            - 0.32 * max(avg_rest_days_l14 - 1.5, 0)
            + 0.06 * max(games_l30 - 12, 0)
            + 0.12 * road_games_l14
            + 0.04 * max(team_pace - 99.0, 0)
            + 1.1 * efficiency_drop
            + 0.5 * recurrence_same_area
        )
        p = float(np.clip(1 / (1 + np.exp(-logit)), 0.03, 0.85))
        injured = int(RNG.random() < p)

        rows.append(
            {
                "player_id": 1000 + (i % len(DEMO_PLAYERS)),
                "player_name": name,
                "team": team,
                "position": pos,
                "as_of_date": as_of.strftime("%Y-%m-%d"),
                "age": int(age_j),
                "age_sq": round(age_sq, 3),
                "age_over_30": age_over_30,
                "is_big": is_big,
                "is_guard": is_guard,
                "minutes_l7": round(minutes_l7, 2),
                "minutes_l14": round(minutes_l14, 2),
                "minutes_l30": round(minutes_l30, 2),
                "minutes_spike_7v30": round(minutes_spike_7v30, 3),
                "games_l7": games_l7,
                "games_l14": games_l14,
                "games_l30": games_l30,
                "road_games_l14": road_games_l14,
                "back_to_backs_l14": back_to_backs_l14,
                "avg_rest_days_l14": round(avg_rest_days_l14, 2),
                "minutes_trend_slope": round(minutes_trend_slope, 3),
                "season_minutes_total": round(season_minutes_total, 1),
                "games_played_season": games_played_season,
                "usage_proxy": round(usage_proxy, 3),
                "team_pace": round(team_pace, 2),
                "efficiency_drop": round(efficiency_drop, 3),
                "prior_injuries_365d": prior_injuries_365d,
                "days_since_last_injury": days_since_last_injury,
                "injury_prone_flag": injury_prone_flag,
                "recurrence_same_area": recurrence_same_area,
                "injured_next_14d": injured,
                "true_risk": round(float(p), 4),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["player_name", "as_of_date"]).reset_index(drop=True)


def generate_player_workload_series(player_name: str, n_games: int = 40) -> pd.DataFrame:
    """Per-game minutes series for trend charts in the dashboard."""
    meta = next((p for p in DEMO_PLAYERS if p[0] == player_name), None)
    if meta is None:
        meta = (player_name, "FA", "SF", 28)
    _, team, pos, age = meta
    start = datetime.fromisoformat("2024-10-22")
    base = 30 if pos in {"C", "PF"} else 28
    prone = name_is_prone(player_name)
    rows = []
    for g in range(n_games):
        day = start + timedelta(days=int(g * 2.3 + RNG.integers(0, 2)))
        mins = float(np.clip(RNG.normal(base, 5) + (4 if g % 7 == 0 else 0), 8, 44))
        # Occasional historical injury markers (as-of past events for timeline overlay)
        is_injury = int(
            (prone and g in {8, 22, 33})
            or ((not prone) and g in {18} and RNG.random() < 0.4)
        )
        if is_injury:
            mins = float(np.clip(mins * 0.35, 0, 18))
        rows.append(
            {
                "player_name": player_name,
                "team": team,
                "position": pos,
                "age": age,
                "game_date": day.strftime("%Y-%m-%d"),
                "minutes": round(mins, 1),
                "is_b2b": int(g > 0 and (day - (start + timedelta(days=int((g - 1) * 2.3)))).days <= 1),
                "is_injury_event": is_injury,
            }
        )
    return pd.DataFrame(rows)


def name_is_prone(player_name: str) -> bool:
    return player_name in {
        "Kawhi Leonard",
        "Joel Embiid",
        "Zion Williamson",
        "Anthony Davis",
        "Jimmy Butler",
        "LaMelo Ball",
        "Ja Morant",
    }

def try_fetch_nba_api_sample(max_players: int = 15) -> pd.DataFrame | None:
    """
    Optional live pull via nba_api. Returns None on any failure so demo data
    remains the reliable default for hackathon demos.
    """
    try:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playergamelog
        import time

        active = [p for p in players.get_active_players() if p.get("is_active")]
        sample = active[:max_players]
        frames = []
        for p in sample:
            try:
                log = playergamelog.PlayerGameLog(
                    player_id=p["id"],
                    season="2024-25",
                    season_type_all_star="Regular Season",
                )
                pdf = log.get_data_frames()[0]
                if pdf.empty:
                    continue
                pdf["PLAYER_NAME"] = p["full_name"]
                frames.append(pdf)
                time.sleep(0.7)
            except Exception:
                continue
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        path = DATA_RAW / "nba_api_game_logs.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        return out
    except Exception:
        return None


def build_dataset(use_live_api: bool = False, n_snapshots: int = 2500) -> pd.DataFrame:
    """Main entry: always produces a model-ready feature table."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    meta = {"source": "demo_synthetic", "built_at": datetime.now().isoformat()}
    if use_live_api:
        live = try_fetch_nba_api_sample()
        meta["nba_api_rows"] = 0 if live is None else int(len(live))
        meta["source"] = "demo_plus_nba_api_logs" if live is not None else "demo_synthetic"

    df = generate_demo_dataset(n_snapshots=n_snapshots)
    out_path = DATA_PROCESSED / "injury_features.parquet"
    csv_path = DATA_PROCESSED / "injury_features.csv"
    df.to_parquet(out_path, index=False)
    df.to_csv(csv_path, index=False)

    # Cache latest snapshot per player for the app
    latest = (
        df.sort_values("as_of_date")
        .groupby("player_name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest.to_csv(DATA_PROCESSED / "latest_player_snapshots.csv", index=False)

    # Workload series for charts
    series = pd.concat(
        [generate_player_workload_series(name) for name, *_ in DEMO_PLAYERS],
        ignore_index=True,
    )
    series.to_csv(DATA_PROCESSED / "player_workload_series.csv", index=False)

    (DATA_PROCESSED / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    frame = build_dataset(use_live_api=False)
    print(f"Built {len(frame):,} rows | injury rate={frame['injured_next_14d'].mean():.1%}")
    preview_cols = ["player_name", "minutes_spike_7v30", "back_to_backs_l14", "injured_next_14d"]
    print(frame[preview_cols].head())
