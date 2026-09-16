"""Helpers shared across phases: hand-off files, metrics, encoding."""
import json
import os
import pickle

import numpy as np
import pandas as pd

from config import (INTERIM_DIR, RESULTS_DIR, TEST_START_YEAR, TEST_END_YEAR, TRAIN_END_YEAR,
                    PANEL_MODE, LEGACY, XGB_PARAMS)


# ---------------------------------------------------------------------------
# Hand-off between phases
# ---------------------------------------------------------------------------
def save_interim(obj, name):
    path = os.path.join(INTERIM_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {path}")


def load_interim(name):
    path = os.path.join(INTERIM_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the earlier phase that creates it first "
            f"(PANEL_MODE={PANEL_MODE})."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(**kwargs):
    path = os.path.join(INTERIM_DIR, "meta.json")
    meta = load_meta() if os.path.exists(path) else {}
    meta.update(kwargs)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=int)


def load_meta():
    with open(os.path.join(INTERIM_DIR, "meta.json")) as f:
        return json.load(f)


def results_path(name):
    # Use forward slashes: matplotlib on Windows can trip on paths built by
    # os.path.join when combined with bbox_inches savefig.
    return f"{RESULTS_DIR}/{name}"


def banner(text):
    print("\n" + "=" * 62)
    print(f"{text}   [PANEL_MODE={PANEL_MODE}]")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def feature_matrix(frame, feats):
    return frame[feats].apply(pd.to_numeric, errors="coerce").fillna(0)


def split_masks(df):
    train_mask = df["year"] <= TRAIN_END_YEAR
    test_mask = (df["year"] >= TEST_START_YEAR) & (df["year"] <= TEST_END_YEAR)
    return train_mask, test_mask


def safe_encode(le, s):
    """Encode with an already-fitted LabelEncoder; unseen categories -> -1."""
    known = {c: i for i, c in enumerate(le.classes_)}
    return s.astype(str).map(known).fillna(-1).astype(int)


def encode_frame(frame, encoders):
    """Apply the phase-3 encoders to a new frame (e.g. the full network)."""
    frame["MATERIAL_enc"] = safe_encode(encoders["MATERIAL"], frame["MATERIAL"])
    frame["TYPE_enc"] = safe_encode(encoders["TYPE"], frame["TYPE"])
    frame["ZONE_enc"] = safe_encode(encoders["ZONE"], frame["ZONE"])
    frame["SOIL_enc"] = safe_encode(encoders["SOIL_TYPE"], frame["SOIL_TYPE"])
    return frame


# ---------------------------------------------------------------------------
# XGBoost with proper categorical handling
# ---------------------------------------------------------------------------
# Material, pressure zone and soil are label-encoded integers. CatBoost is told
# they are categorical via cat_features. For XGBoost (used in the ablation,
# spatial CV and SHAP) they are passed as pandas categoricals with
# enable_categorical=True, so the integer codes are never treated as ordered.
# Legacy mode keeps the original numeric handling so it reproduces old results.
CAT_ENCODER_KEYS = {"MATERIAL_enc": "MATERIAL", "ZONE_enc": "ZONE", "SOIL_enc": "SOIL_TYPE"}


def xgb_matrix(frame, feats, encoders):
    X = feature_matrix(frame, feats)
    if LEGACY:
        return X
    for f in feats:
        if f in CAT_ENCODER_KEYS:
            n = len(encoders[CAT_ENCODER_KEYS[f]].classes_)
            codes = X[f].astype(int).where(X[f] >= 0).astype("Int64")   # unseen -> missing
            X[f] = pd.Categorical(codes, categories=range(n))
    return X


def make_xgb(scale_pos_weight):
    import xgboost as xgb
    return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                             enable_categorical=not LEGACY, **XGB_PARAMS)


def xgb_shap_values(model, X):
    """SHAP values for an XGBoost model. Fixed mode uses XGBoost's built-in
    TreeSHAP, which supports categorical splits."""
    if LEGACY:
        import shap
        return shap.TreeExplainer(model).shap_values(X)
    import xgboost as xgb
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True),
                                           pred_contribs=True)
    return contribs[:, :-1]          # last column is the bias term


def display_matrix(X, names):
    """Numeric copy for SHAP plot colouring (categorical codes; colour arbitrary)."""
    D = X.copy()
    for c in D.columns:
        if isinstance(D[c].dtype, pd.CategoricalDtype):
            D[c] = D[c].cat.codes.replace(-1, np.nan).astype(float)
    D.columns = names
    return D


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_lorenz(y_true, y_prob, weights=None):
    """Lorenz curve values (Forero-Ortiz et al. 2026 primary metric)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights)
    order = np.argsort(y_prob)[::-1]
    cum_net = np.cumsum(w[order]) / w.sum()
    cum_fail = np.cumsum(y_true[order]) / max(y_true.sum(), 1)
    vals = {}
    for r in [0.01, 0.05, 0.10, 0.25]:
        idx = min(np.searchsorted(cum_net, r), len(cum_fail) - 1)
        vals[f"{int(r*100)}pct"] = cum_fail[idx]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    vals["auc"] = trap(cum_fail, cum_net)
    return vals, cum_net, cum_fail


def annual_lorenz(frame, prob, years, thresholds=(0.01, 0.05, 0.10), min_pos=5):
    """Annual-renewal Lorenz: rank the in-service network each year, renew the
    top X% by length, record the share of that year's failures captured.

    Returns (summary {k: (mean, sd)}, years used, per-year {k: [values]}).
    """
    prob = np.asarray(prob)
    per_year = {f"{int(t*100)}pct": [] for t in thresholds}
    used = []
    for yr in years:
        m = frame["year"].values == yr
        yt = frame["target"].values[m]
        if yt.sum() < min_pos:
            continue
        pr = prob[m]
        w = frame["LENGTH"].values[m]
        order = np.argsort(pr)[::-1]
        cum_len = np.cumsum(w[order]) / w.sum()
        cum_fail = np.cumsum(yt[order]) / yt.sum()
        for t in thresholds:
            idx = min(np.searchsorted(cum_len, t), len(cum_fail) - 1)
            per_year[f"{int(t*100)}pct"].append(cum_fail[idx])
        used.append(int(yr))
    summ = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_year.items() if v}
    return summ, used, per_year
