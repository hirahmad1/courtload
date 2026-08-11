"""
Train a gradient-boosted injury risk model + export SHAP artifacts.

Tries LightGBM first, then XGBoost, then sklearn GradientBoosting.
Applies isotonic calibration so watchlist probabilities stay credible.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED, FEATURE_COLS, LGBM_PARAMS, MODELS_DIR, TARGET_COL
from src.data_collection import build_dataset
from src.features import load_feature_table, xy_split


def _make_model():
    try:
        import lightgbm as lgb

        return "lightgbm", lgb.LGBMClassifier(**LGBM_PARAMS)
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier

        return "xgboost", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
        )
    except Exception:
        pass
    from sklearn.ensemble import GradientBoostingClassifier

    return "sklearn_gbm", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def _time_aware_split(df: pd.DataFrame, test_size: float = 0.2):
    """Prefer chronological split when dates exist; else stratified random."""
    if "as_of_date" in df.columns:
        df = df.sort_values("as_of_date")
        cut = int(len(df) * (1 - test_size))
        train_df, test_df = df.iloc[:cut], df.iloc[cut:]
        if test_df[TARGET_COL].nunique() < 2 or train_df[TARGET_COL].nunique() < 2:
            return train_test_split(df, test_size=test_size, stratify=df[TARGET_COL], random_state=42)
        return train_df, test_df
    return train_test_split(df, test_size=test_size, stratify=df[TARGET_COL], random_state=42)


def _fit_calibrator(raw_proba: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Map raw GBM scores → calibrated probabilities (clip extremes)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.92)
    iso.fit(raw_proba, y)
    return iso


def _apply_calibrator(calibrator: IsotonicRegression | None, raw: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw
    return np.asarray(calibrator.transform(raw), dtype=float)


def compute_shap_values(model, X: pd.DataFrame, max_rows: int = 400):
    """Return (explainer, shap_values_array, X_sample) — binary positive class."""
    import shap

    sample = X.sample(n=min(max_rows, len(X)), random_state=42)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        return explainer, np.asarray(sv), sample
    except Exception:
        explainer = shap.Explainer(model.predict_proba, sample)
        explanation = explainer(sample)
        vals = explanation.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return explainer, np.asarray(vals), sample


def train(rebuild_data: bool = True, n_snapshots: int = 2500) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if rebuild_data or not (DATA_PROCESSED / "injury_features.csv").exists():
        build_dataset(use_live_api=False, n_snapshots=n_snapshots)

    df = load_feature_table(DATA_PROCESSED / "injury_features.csv")
    train_df, test_df = _time_aware_split(df)

    # Chronological fit / calibrate split inside the train window (no test leakage)
    train_df = train_df.sort_values("as_of_date") if "as_of_date" in train_df.columns else train_df
    cal_cut = int(len(train_df) * 0.85)
    fit_df, cal_df = train_df.iloc[:cal_cut], train_df.iloc[cal_cut:]
    if cal_df[TARGET_COL].nunique() < 2:
        fit_df, cal_df = train_df, train_df.sample(frac=0.2, random_state=42)

    X_fit, y_fit = xy_split(fit_df)
    X_cal, y_cal = xy_split(cal_df)
    X_test, y_test = xy_split(test_df)
    X_train_full, y_train_full = xy_split(train_df)

    backend, model = _make_model()
    model.fit(X_fit, y_fit)

    calibrator = _fit_calibrator(model.predict_proba(X_cal)[:, 1], y_cal.to_numpy())

    # Refit on full train for final booster; keep calibrator from held-out cal slice
    model.fit(X_train_full, y_train_full)

    raw_test = model.predict_proba(X_test)[:, 1]
    proba = _apply_calibrator(calibrator, raw_test)
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "backend": backend,
        "headline_metric": "pr_auc",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "injury_rate_train": float(y_train_full.mean()),
        "injury_rate_test": float(y_test.mean()),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "brier_uncalibrated": float(brier_score_loss(y_test, raw_test)),
        "report": classification_report(y_test, preds, output_dict=True),
        "feature_cols": FEATURE_COLS,
        "leakage_controls": [
            "Chronological train/test split by as_of_date",
            "prior_injuries_365d / days_since_last_injury are as-of snapshot only",
            "Target injured_next_14d is never used as a feature",
            "Isotonic calibration fit on a late train slice, not the test set",
        ],
        "calibration": "isotonic",
    }

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": 0.0})
    imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    try:
        explainer, shap_vals, _sample = compute_shap_values(model, X_train_full)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_imp = pd.DataFrame(
            {"feature": FEATURE_COLS, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_imp.to_csv(MODELS_DIR / "shap_importance.csv", index=False)
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
        metrics["shap_top5"] = shap_imp.head(5).to_dict(orient="records")
    except Exception as exc:
        metrics["shap_error"] = str(exc)

    joblib.dump(model, MODELS_DIR / "injury_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / "probability_calibrator.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latest = pd.read_csv(DATA_PROCESSED / "latest_player_snapshots.csv")
    raw_latest = model.predict_proba(latest[FEATURE_COLS])[:, 1]
    latest["injury_risk"] = _apply_calibrator(calibrator, raw_latest)
    latest = latest.sort_values("injury_risk", ascending=False)
    latest.to_csv(DATA_PROCESSED / "latest_scored.csv", index=False)

    print(f"Backend: {backend} | calibration: isotonic")
    print(
        f"PR-AUC (headline): {metrics['pr_auc']:.3f} | "
        f"ROC-AUC: {metrics['roc_auc']:.3f} | Brier: {metrics['brier']:.3f}"
    )
    print(f"Watchlist risk range: {latest['injury_risk'].min():.1%} – {latest['injury_risk'].max():.1%}")
    print("Top features:")
    print(imp.head(8).to_string(index=False))
    return metrics


if __name__ == "__main__":
    train(rebuild_data=True)
