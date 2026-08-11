"""
CourtLoad — NBA Injury Risk Command Center

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PROCESSED, MODELS_DIR
from src.features import humanize_feature, risk_band
from src.predict import (
    build_manual_snapshot,
    explain_player,
    format_front_office_report,
    load_calibrator,
    load_explainer,
    load_model,
    plain_language_summary,
    predict_risk,
    score_frame,
)

st.set_page_config(
    page_title="CourtLoad — Injury Risk",
    page_icon="⬤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette constants (day defaults; night via get_theme_colors)
PLOT_INK = "#10231a"
PLOT_GRID = "rgba(16,35,26,0.10)"
RISK_UP = "#c62828"
RISK_DOWN = "#1b5e40"
B2B = "#b8892a"

_BRAND_CSS_PATH = Path(__file__).resolve().parent / "brand.css"


def get_theme() -> str:
    return st.session_state.get("theme", "Day")


def get_theme_colors() -> dict:
    if get_theme() == "Night":
        return {
            "ink": "#e8f0ea",
            "ink_soft": "rgba(232,240,234,0.68)",
            "grid": "rgba(232,240,234,0.12)",
            "plot_bg": "rgba(12,28,22,0.55)",
            "pine": "#7dba95",
            "risk_up": "#ff6b6b",
            "risk_down": "#7dba95",
            "b2b": "#d4a84b",
            "fill": "rgba(125,186,149,0.14)",
        }
    return {
        "ink": "#10231a",
        "ink_soft": "rgba(16,35,26,0.65)",
        "grid": "rgba(16,35,26,0.10)",
        "plot_bg": "rgba(255,255,255,0.28)",
        "pine": "#1b5e40",
        "risk_up": "#c62828",
        "risk_down": "#1b5e40",
        "b2b": "#b8892a",
        "fill": "rgba(27,94,64,0.10)",
    }


def apply_theme(theme: str) -> None:
    """Inject brand CSS + set data-theme on Streamlit root for day/night."""
    st.session_state["theme"] = theme
    mode = "night" if theme == "Night" else "day"
    css = _BRAND_CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    import streamlit.components.v1 as components

    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const app = doc.querySelector('.stApp');
        if (app) {{
          app.setAttribute('data-theme', '{mode}');
        }}
        </script>
        """,
        height=0,
        width=0,
    )


def _plot_layout(**extra):
    c = get_theme_colors()
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=c["plot_bg"],
        font=dict(family="Manrope", color=c["ink"], size=12),
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], showline=False),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], showline=False),
    )
    base.update(extra)
    return base


def risk_dial_svg(prob: float, color: str) -> str:
    """Semi-circle gauge — single-line SVG so Streamlit markdown HTML stays intact."""
    c = get_theme_colors()
    p = float(np.clip(prob, 0, 1))
    angle = 180 - p * 180
    rad = np.deg2rad(angle)
    cx, cy, r = 90, 88, 70
    x = cx + r * np.cos(rad)
    y = cy - r * np.sin(rad)
    large = 1 if p > 0.5 else 0
    start_x, start_y = cx - r, cy
    arc = f"M {start_x:.1f} {start_y:.1f} A {r} {r} 0 {large} 1 {x:.1f} {y:.1f}"
    track = "rgba(232,240,234,0.18)" if get_theme() == "Night" else "rgba(16,35,26,0.12)"
    return (
        f'<svg width="160" height="100" viewBox="0 0 180 110" aria-hidden="true">'
        f'<path d="M 20 88 A 70 70 0 0 1 160 88" fill="none" stroke="{track}" '
        f'stroke-width="12"/>'
        f'<path d="{arc}" fill="none" stroke="{color}" stroke-width="12"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}"/>'
        f'<circle cx="90" cy="88" r="3.5" fill="{c["ink"]}"/>'
        f"</svg>"
    )


def render_risk_dial(risk: float, color: str, label: str, note: str = "", *, height: int = 128) -> None:
    """Render dial + % via components.html (theme-aware)."""
    import streamlit.components.v1 as components

    c = get_theme_colors()
    note_html = (
        f'<div style="margin-top:6px;font-size:14px;color:{c["ink_soft"]};'
        f'font-family:Manrope,sans-serif;line-height:1.35;">{note}</div>'
        if note
        else ""
    )
    html = f"""
    <div style="display:flex;align-items:center;gap:14px;font-family:Manrope,sans-serif;color:{c["ink"]};">
      {risk_dial_svg(risk, color)}
      <div>
        <div style="font-family:JetBrains Mono,monospace;font-size:2.35rem;font-weight:600;line-height:1;color:{c["ink"]};">{risk:.0%}</div>
        <div style="display:inline-block;margin-top:8px;font-family:JetBrains Mono,monospace;font-size:0.72rem;
             letter-spacing:0.1em;text-transform:uppercase;padding:4px 8px;background:{color};color:#fff;font-weight:600;">
          {label}
        </div>
        {note_html}
      </div>
    </div>
    """
    components.html(html, height=height, scrolling=False)


@st.cache_resource
def get_model_bundle():
    return load_model(), load_explainer(), load_calibrator()


@st.cache_data
def load_scored() -> pd.DataFrame:
    path = DATA_PROCESSED / "latest_scored.csv"
    if not path.exists():
        snap = DATA_PROCESSED / "latest_player_snapshots.csv"
        if not snap.exists():
            st.error("No data found. Run `python -m src.train_model` first.")
            st.stop()
        model, _, calibrator = get_model_bundle()
        return score_frame(pd.read_csv(snap), model, calibrator)
    return pd.read_csv(path).sort_values("injury_risk", ascending=False).reset_index(drop=True)


@st.cache_data
def load_workload() -> pd.DataFrame:
    path = DATA_PROCESSED / "player_workload_series.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_importance() -> pd.DataFrame:
    shap_path = MODELS_DIR / "shap_importance.csv"
    fi_path = MODELS_DIR / "feature_importance.csv"
    if shap_path.exists():
        df = pd.read_csv(shap_path)
    elif fi_path.exists():
        df = pd.read_csv(fi_path).rename(columns={"importance": "mean_abs_shap"})
    else:
        return pd.DataFrame()
    df["label"] = df["feature"].map(humanize_feature)
    return df


def render_hero():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">NBA load intelligence</div>
          <h1 class="hero-brand">COURT<em>LOAD</em></h1>
          <div class="hero-rule"></div>
          <p class="hero-sub">
            Calibrated 14-day injury risk from minutes spikes, schedule density,
            and recovery history — for load management decisions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workload_figure(pw: pd.DataFrame, title_note: str = "Gold = back-to-back · red = prior injury"):
    tc = get_theme_colors()
    pw = pw.copy()
    pw["game_date"] = pd.to_datetime(pw["game_date"])
    if "is_injury_event" not in pw.columns:
        pw["is_injury_event"] = 0
    marker_colors = np.where(
        pw["is_injury_event"] == 1,
        tc["risk_up"],
        np.where(pw["is_b2b"] == 1, tc["b2b"], tc["risk_down"]),
    )
    marker_sizes = np.where(pw["is_injury_event"] == 1, 12, 7)
    fig = go.Figure(
        go.Scatter(
            x=pw["game_date"],
            y=pw["minutes"],
            mode="lines+markers",
            line=dict(color=tc["risk_down"], width=2.4),
            marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=0)),
            fill="tozeroy",
            fillcolor=tc["fill"],
            customdata=np.stack([pw["is_b2b"], pw["is_injury_event"]], axis=1),
            hovertemplate="%{x|%b %d}<br>%{y:.1f} min<br>B2B=%{customdata[0]} · injury=%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        **_plot_layout(
            height=250,
            yaxis_title="Minutes",
            showlegend=False,
            title=dict(text=title_note, font=dict(size=12, color=tc["ink_soft"])),
        )
    )
    return fig


def render_export_button(row, risk, label, shap_df, key: str):
    report = format_front_office_report(row, risk, label, shap_df)
    st.download_button(
        "Download brief (.txt)",
        data=report,
        file_name=f"courtload_{str(row.get('player_name', 'player')).replace(' ', '_').lower()}_brief.txt",
        mime="text/plain",
        key=key,
        use_container_width=True,
    )


def render_whatif_scout(model, explainer, calibrator):
    st.markdown(
        """
        <div class="section-head">
          <h2>What-if scout</h2>
          <p>Build a player snapshot, score risk, then add rest to preview load-management impact.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("whatif_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Player name", value="Custom Player")
            age = st.number_input("Age", min_value=18, max_value=45, value=28, step=1)
            position = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"], index=3)
            minutes_l7 = st.slider("Minutes (last 7 days)", 0.0, 42.0, 34.0, 0.5)
            minutes_l14 = st.slider("Minutes (last 14 days)", 0.0, 40.0, 30.0, 0.5)
        with c2:
            back_to_backs = st.number_input("Back-to-backs (last 14 days)", 0, 4, 2, 1)
            avg_rest = st.slider("Avg rest days (last 14 days)", 0.5, 4.0, 1.4, 0.1)
            prior_injuries = st.number_input("Prior injuries (last 12 months)", 0, 10, 1, 1)
            days_since = st.number_input("Days since last injury", 1, 500, 25, 1)
            minutes_l30 = st.slider("Minutes baseline (last 30 days)", 5.0, 40.0, 28.0, 0.5)

        with st.expander("Schedule / advanced"):
            a1, a2, a3, a4 = st.columns(4)
            games_l7 = a1.number_input("Games L7", 1, 4, 3)
            games_l14 = a2.number_input("Games L14", 2, 8, 6)
            games_l30 = a3.number_input("Games L30", 6, 16, 12)
            road_games = a4.number_input("Road games L14", 0, 8, 3)
            b1, b2, b3 = st.columns(3)
            usage = b1.slider("Usage proxy", 0.3, 1.4, 0.75, 0.05)
            pace = b2.slider("Team pace", 95.0, 106.0, 99.0, 0.5)
            eff_drop = b3.slider("Efficiency drop", 0.0, 0.4, 0.08, 0.01)
            recurrence = st.checkbox("Same-area injury recurrence", value=False)
            games_season = st.slider("Games played (season)", 1, 82, 50)

        submitted = st.form_submit_button("Score risk", type="primary", use_container_width=True)

    if not submitted and "whatif_row" not in st.session_state:
        st.markdown(
            '<div class="empty-hint">Complete the form and select <strong>Score risk</strong> to open the rest simulator and SHAP drivers.</div>',
            unsafe_allow_html=True,
        )
        return

    if submitted:
        row = build_manual_snapshot(
            player_name=player_name,
            age=int(age),
            position=position,
            minutes_l7=float(minutes_l7),
            minutes_l14=float(minutes_l14),
            minutes_l30=float(minutes_l30),
            back_to_backs_l14=int(back_to_backs),
            avg_rest_days_l14=float(avg_rest),
            prior_injuries_365d=int(prior_injuries),
            days_since_last_injury=int(days_since),
            games_l7=int(games_l7),
            games_l14=int(games_l14),
            games_l30=int(games_l30),
            road_games_l14=int(road_games),
            games_played_season=int(games_season),
            usage_proxy=float(usage),
            team_pace=float(pace),
            efficiency_drop=float(eff_drop),
            recurrence_same_area=int(recurrence),
        )
        st.session_state["whatif_row"] = row
        st.session_state["whatif_base_rest"] = float(avg_rest)

    base = st.session_state["whatif_row"].copy()
    st.markdown('<div class="divider-label">Rest simulator</div>', unsafe_allow_html=True)
    extra_rest = st.slider(
        "Additional rest days",
        0.0,
        3.0,
        0.0,
        0.1,
        help="Simulates sitting a player or spacing the schedule. Risk updates live.",
    )
    sim = base.copy()
    sim["avg_rest_days_l14"] = float(base["avg_rest_days_l14"]) + float(extra_rest)
    # Extra rest also softens B2B / road pressure in the scenario
    if extra_rest >= 1.0:
        sim["back_to_backs_l14"] = max(int(base["back_to_backs_l14"]) - 1, 0)
    if extra_rest >= 2.0:
        sim["road_games_l14"] = max(int(base.get("road_games_l14", 0)) - 1, 0)
        sim["efficiency_drop"] = max(float(base.get("efficiency_drop", 0)) - 0.05, 0)

    base_risk = predict_risk(base, model=model, calibrator=calibrator)
    risk = predict_risk(sim, model=model, calibrator=calibrator)
    label, color = risk_band(risk)
    base_label, base_color = risk_band(base_risk)
    shap_df = explain_player(sim, model=model, explainer=explainer, top_k=8)
    briefing = plain_language_summary(str(sim["player_name"]), risk, label, shap_df)
    delta = risk - base_risk
    tc = get_theme_colors()
    delta_color = tc["risk_down"] if delta <= 0 else tc["risk_up"]

    st.markdown(
        f"""
        <div class="delta-strip">
          <div class="cell">
            <div class="k">Current</div>
            <div class="v" style="color:{base_color};">{base_risk:.0%}</div>
            <div class="s" style="color:{base_color};">{base_label}</div>
          </div>
          <div class="arrow">→</div>
          <div class="cell">
            <div class="k">With +{extra_rest:.1f}d rest</div>
            <div class="v" style="color:{color};">{risk:.0%}</div>
            <div class="s" style="color:{color};">{label}</div>
          </div>
          <div class="cell">
            <div class="k">Delta</div>
            <div class="v" style="color:{delta_color};">{delta:+.0%} pts</div>
            <div class="s" style="color:{tc["ink_soft"]};font-weight:500;">Change from added rest</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.95, 1.15], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="dossier">
              <h3 class="dossier-name">{sim["player_name"]}</h3>
              <div class="dossier-meta">WHAT-IF · {sim["position"]} · age {int(sim["age"])}</div>
              <div class="briefing">{briefing}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_risk_dial(
            risk,
            color,
            label,
            note=f"Baseline {base_risk:.0%} → simulated {risk:.0%} ({delta:+.0%} pts)",
        )
        st.markdown(
            f"""
            <div class="micro-grid">
              <div class="micro"><div class="k">Minutes L7</div><div class="v">{sim["minutes_l7"]:.0f}</div></div>
              <div class="micro"><div class="k">Spike vs 30d</div><div class="v">{sim["minutes_spike_7v30"]:+.0%}</div></div>
              <div class="micro"><div class="k">Rest (simulated)</div><div class="v">{sim["avg_rest_days_l14"]:.1f}d</div></div>
              <div class="micro"><div class="k">B2Bs L14</div><div class="v">{int(sim["back_to_backs_l14"])}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_export_button(sim, risk, label, shap_df, key="export_whatif")

    with right:
        st.markdown('<div class="divider-label">Score drivers · SHAP</div>', unsafe_allow_html=True)
        tc = get_theme_colors()
        colors = [tc["risk_up"] if v > 0 else tc["risk_down"] for v in shap_df["shap"]]
        fig_shap = go.Figure(
            go.Bar(
                x=shap_df["shap"],
                y=shap_df["feature_label"],
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="%{y}<br>SHAP %{x:.3f}<br>Value %{customdata:.2f}<extra></extra>",
                customdata=shap_df["value"],
            )
        )
        fig_shap.update_layout(
            **_plot_layout(
                height=360,
                xaxis_title="Impact on injury probability",
                yaxis=dict(autorange="reversed", gridcolor=tc["grid"]),
            )
        )
        st.plotly_chart(fig_shap, width="stretch")


def render_team_heatmap(scored: pd.DataFrame):
    st.markdown(
        """
        <div class="section-head">
          <h2>Team roster</h2>
          <p>Glance view of monitored players for one club, ranked by 14-day risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Prefer teams with the largest monitored roster for a useful default
    counts = scored.groupby("team")["player_name"].count().sort_values(ascending=False)
    teams = counts.index.tolist()
    default_idx = teams.index("LAL") if "LAL" in teams else 0
    team = st.selectbox("Team", teams, index=default_idx, key="heatmap_team")
    roster = scored[scored["team"] == team].sort_values("injury_risk", ascending=False).copy()
    if roster.empty:
        st.markdown('<div class="empty-hint">No players monitored for this team.</div>', unsafe_allow_html=True)
        return

    st.caption(f"{len(roster)} players · high → low risk")

    n_cols = 3
    for start in range(0, len(roster), n_cols):
        chunk = roster.iloc[start : start + n_cols]
        cols = st.columns(n_cols)
        for col, r in zip(cols, chunk.itertuples()):
            band, color = risk_band(float(r.injury_risk))
            with col:
                st.markdown(
                    f"""
                    <div class="roster-tile" style="--tile-accent:{color};">
                      <div class="name">{r.player_name}</div>
                      <div class="meta">{r.position} · age {int(r.age)}</div>
                      <div class="risk">{float(r.injury_risk):.0%}</div>
                      <div class="band">{band}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="divider-label">Roster ranking</div>', unsafe_allow_html=True)
    tc = get_theme_colors()
    fig = go.Figure(
        go.Bar(
            x=roster["injury_risk"],
            y=roster["player_name"],
            orientation="h",
            marker=dict(
                color=roster["injury_risk"],
                colorscale=[[0, tc["risk_down"]], [0.45, tc["b2b"]], [1, tc["risk_up"]]],
                cmin=0,
                cmax=1,
            ),
            hovertemplate="%{y}<br>Risk %{x:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_plot_layout(
            height=max(280, 32 * len(roster)),
            xaxis_title="Injury risk",
            yaxis=dict(autorange="reversed"),
            title=dict(text=f"{team} · high → low", font=dict(size=14)),
        )
    )
    st.plotly_chart(fig, width="stretch")


def render_compare(scored: pd.DataFrame, model, explainer, calibrator):
    st.markdown(
        """
        <div class="section-head">
          <h2>Compare</h2>
          <p>Side-by-side risk profiles for roster and load debates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    names = scored.sort_values("injury_risk", ascending=False)["player_name"].tolist()
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Player A", names, index=0, key="cmp_a")
    with c2:
        default_b = 1 if len(names) > 1 else 0
        p2 = st.selectbox("Player B", names, index=default_b, key="cmp_b")

    rows = [scored[scored["player_name"] == p1].iloc[0], scored[scored["player_name"] == p2].iloc[0]]
    cols = st.columns(2)
    for col, row in zip(cols, rows):
        risk = float(row["injury_risk"])
        label, color = risk_band(risk)
        shap_df = explain_player(row, model=model, explainer=explainer, top_k=5)
        briefing = plain_language_summary(str(row["player_name"]), risk, label, shap_df)
        with col:
            st.markdown(
                f"""
                <div class="dossier">
                  <h3 class="dossier-name">{row["player_name"]}</h3>
                  <div class="dossier-meta">{row["team"]} · {row["position"]} · age {int(row["age"])}</div>
                  <div class="briefing">{briefing}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_risk_dial(risk, color, label)
            st.markdown(
                f"""
                <div class="micro-grid">
                  <div class="micro"><div class="k">Min L7</div><div class="v">{row["minutes_l7"]:.0f}</div></div>
                  <div class="micro"><div class="k">Spike</div><div class="v">{row["minutes_spike_7v30"]:+.0%}</div></div>
                  <div class="micro"><div class="k">B2B</div><div class="v">{int(row["back_to_backs_l14"])}</div></div>
                  <div class="micro"><div class="k">Games L30</div><div class="v">{int(row.get("games_l30", 0))}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tc = get_theme_colors()
            colors = [tc["risk_up"] if v > 0 else tc["risk_down"] for v in shap_df["shap"]]
            fig = go.Figure(
                go.Bar(
                    x=shap_df["shap"],
                    y=shap_df["feature_label"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                )
            )
            fig.update_layout(
                **_plot_layout(height=260, yaxis=dict(autorange="reversed"), xaxis_title="SHAP")
            )
            st.plotly_chart(fig, width="stretch")


def render_theme_toggle():
    """Small day/night icon at the top of the main page (not sidebar)."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Day"
    is_night = st.session_state["theme"] == "Night"
    icon = "☀" if is_night else "☾"
    tip = "Switch to Day" if is_night else "Switch to Night"
    _, tip_col = st.columns([0.92, 0.08])
    with tip_col:
        if st.button(icon, key="theme_icon", help=tip, use_container_width=True):
            st.session_state["theme"] = "Day" if is_night else "Night"
            st.rerun()


def main():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Day"

    apply_theme(st.session_state["theme"])
    render_theme_toggle()
    render_hero()
    model, explainer, calibrator = get_model_bundle()
    scored = load_scored()
    workload = load_workload()
    importance = load_importance()

    if "risk_band" not in scored.columns or any(c not in scored.columns for c in ["games_l30", "age_sq"]):
        scored = score_frame(scored, model, calibrator)

    tab_watch, tab_whatif, tab_heat, tab_cmp = st.tabs(
        ["Watchlist", "What-if", "Roster", "Compare"]
    )

    with tab_whatif:
        render_whatif_scout(model, explainer, calibrator)

    with tab_heat:
        render_team_heatmap(scored)

    with tab_cmp:
        render_compare(scored, model, explainer, calibrator)

    with tab_watch:
        view = scored.sort_values("injury_risk", ascending=False).copy()

        high_n = int((scored["injury_risk"] >= 0.65).sum())
        elev_n = int(((scored["injury_risk"] >= 0.45) & (scored["injury_risk"] < 0.65)).sum())
        avg_risk = float(scored["injury_risk"].mean())

        st.markdown(
            f"""
            <div class="stat-strip">
              <div class="stat-cell">
                <div class="label">Monitored</div>
                <div class="value">{len(scored)}</div>
                <div class="hint">Active snapshots</div>
              </div>
              <div class="stat-cell alert">
                <div class="label">High risk</div>
                <div class="value">{high_n}</div>
                <div class="hint">Prob ≥ 65%</div>
              </div>
              <div class="stat-cell">
                <div class="label">Elevated</div>
                <div class="value">{elev_n}</div>
                <div class="hint">45–65% watch band</div>
              </div>
              <div class="stat-cell">
                <div class="label">League avg</div>
                <div class="value">{avg_risk:.0%}</div>
                <div class="hint">14-day outlook</div>
              </div>
            </div>
            <p class="spectrum-note">
              Bands run Low → Moderate → Elevated → High.
              <strong>Elevated</strong> is the actionable middle — tighten load before risk becomes High.
            </p>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.0, 1.25], gap="large")

        with left:
            st.markdown(
                """
                <div class="section-head">
                  <h2>Watchlist</h2>
                  <p>Ranked by 14-day injury probability. Select a row or search.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            table = view[
                ["player_name", "team", "position", "age", "injury_risk", "risk_band", "minutes_l7", "back_to_backs_l14"]
            ].copy()
            table["injury_risk"] = table["injury_risk"].map(lambda x: round(float(x), 3))
            table = table.rename(
                columns={
                    "player_name": "Player",
                    "team": "Team",
                    "position": "Pos",
                    "age": "Age",
                    "injury_risk": "Risk",
                    "risk_band": "Band",
                    "minutes_l7": "Min L7",
                    "back_to_backs_l14": "B2B",
                }
            )
            event = st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                height=520,
            )

            selected_name = table.iloc[0]["Player"] if len(table) else None
            if event and event.selection and event.selection.rows:
                selected_name = table.iloc[event.selection.rows[0]]["Player"]

            player_names = scored["player_name"].tolist()
            pick = st.selectbox(
                "Search player",
                player_names,
                index=player_names.index(selected_name) if selected_name in player_names else 0,
            )
            selected_name = pick

        row = scored[scored["player_name"] == selected_name].iloc[0]
        risk = float(row["injury_risk"])
        label, color = risk_band(risk)
        shap_df = explain_player(row, model=model, explainer=explainer, top_k=8)
        briefing = plain_language_summary(selected_name, risk, label, shap_df)

        with right:
            st.markdown(
                f"""
                <div class="dossier">
                  <div class="section-head">
                    <h2>Dossier</h2>
                    <p>Calibrated score, drivers, and season load.</p>
                  </div>
                  <h3 class="dossier-name">{selected_name}</h3>
                  <div class="dossier-meta">{row["team"]} · {row["position"]} · age {int(row["age"])}</div>
                  <div class="briefing">{briefing}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_risk_dial(risk, color, label, note="Calibrated probability · next 14 days")
            st.markdown(
                f"""
                <div class="micro-grid">
                  <div class="micro"><div class="k">Minutes L7</div><div class="v">{row["minutes_l7"]:.0f}</div></div>
                  <div class="micro"><div class="k">Spike vs 30d</div><div class="v">{row["minutes_spike_7v30"]:+.0%}</div></div>
                  <div class="micro"><div class="k">Back-to-backs L14</div><div class="v">{int(row["back_to_backs_l14"])}</div></div>
                  <div class="micro"><div class="k">Games L30</div><div class="v">{int(row.get("games_l30", 0))}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_export_button(row, risk, label, shap_df, key="export_watch")

            st.markdown('<div class="divider-label">Score drivers · SHAP</div>', unsafe_allow_html=True)
            tc = get_theme_colors()
            colors = [tc["risk_up"] if v > 0 else tc["risk_down"] for v in shap_df["shap"]]
            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["shap"],
                    y=shap_df["feature_label"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    hovertemplate="%{y}<br>SHAP %{x:.3f}<br>Value %{customdata:.2f}<extra></extra>",
                    customdata=shap_df["value"],
                )
            )
            fig_shap.update_layout(
                **_plot_layout(
                    height=300,
                    xaxis_title="Impact on injury probability",
                    yaxis=dict(autorange="reversed", gridcolor=tc["grid"]),
                )
            )
            st.plotly_chart(fig_shap, width="stretch")

            pw = workload[workload["player_name"] == selected_name].copy() if not workload.empty else pd.DataFrame()
            if not pw.empty:
                st.markdown(
                    '<div class="divider-label">Season workload</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(workload_figure(pw), width="stretch")

        st.markdown(
            """
            <div class="section-head" style="margin-top:1.75rem">
              <h2>Model insight</h2>
              <p>League-wide feature influence on injury probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not importance.empty:
            top = importance.head(12).iloc[::-1]
            fig_imp = px.bar(
                top,
                x="mean_abs_shap",
                y="label",
                orientation="h",
                labels={"mean_abs_shap": "Mean |SHAP|", "label": ""},
            )
            fig_imp.update_traces(marker_color=get_theme_colors()["pine"], marker_line_width=0)
            fig_imp.update_layout(**_plot_layout(height=380, showlegend=False))
            st.plotly_chart(fig_imp, width="stretch")

        with st.expander("Methodology"):
            st.markdown(
                """
                **Target:** injured / misses time within 14 days of a snapshot.

                **Features (as-of date only):** age curve (`age_sq`, `age_over_30`), position load,
                rolling minutes, schedule density (`games_l30`, road games), back-to-backs, rest,
                usage, team pace, efficiency drop (fatigue), prior injuries, same-area recurrence.
                The label `injured_next_14d` is never a feature.

                **Model:** LightGBM + isotonic calibration. Chronological holdout.
                **Headline metric: PR-AUC**.

                **Dashboard:** plain-language briefing, what-if rest simulator, team heatmap,
                player compare, injury timeline overlay, front-office brief export.
                """
            )


if __name__ == "__main__":
    main()
