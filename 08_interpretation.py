"""
Phase 8 — Interpretation and operational outputs.

  - SHAP summary and bar plots (XGBoost; CatBoost's explainer is very slow)
  - material-stratified SHAP (cast iron vs PVC)
  - material-stratified Lorenz (full network, first test year)
  - top-100 highest-risk pipes, confirmed against 2019-2025 breaks

Outputs (results/): shap_summary.png, shap_bar.png, shap_ci_vs_pvc.png,
                    material_lorenz.csv, top100_risky_pipes.csv
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import LEGACY, FEATURES, FEATURE_NAMES, TEST_START_YEAR, TEST_END_YEAR, RANDOM_STATE
from common import (banner, load_interim, xgb_matrix, xgb_shap_values, display_matrix,
                    split_masks, compute_lorenz, results_path)

warnings.filterwarnings("ignore")
banner("PHASE 8 — INTERPRETATION")

df = load_interim("df")
encoders = load_interim("encoders")
df_breaks = load_interim("df_breaks")
models = load_interim("models")
full = load_interim("full_network")
fn, prob_fn, fn_years = full["fn"], full["prob_fn"], full["fn_years"]

_, test_mask = split_masks(df)
X = xgb_matrix(df, FEATURES, encoders)
X_te = X[test_mask]

# ---------------------------------------------------------------------------
# SHAP — global
# ---------------------------------------------------------------------------
print("  SHAP analysis (XGBoost)...")
xgb_model = models["XGBoost"]
X_shap = X_te.sample(min(1000, len(X_te)), random_state=RANDOM_STATE)
shap_values = xgb_shap_values(xgb_model, X_shap)
X_disp = display_matrix(X_shap, FEATURE_NAMES)   # colour for categorical features is arbitrary

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_disp, show=False, max_display=len(FEATURE_NAMES))
plt.title("SHAP — XGBoost, 1-Year Window (Temporal Split)")
plt.tight_layout(); plt.savefig(results_path("shap_summary.png"), dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_disp, plot_type="bar", show=False, max_display=len(FEATURE_NAMES))
plt.tight_layout(); plt.savefig(results_path("shap_bar.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------
# SHAP — cast iron vs PVC
# ---------------------------------------------------------------------------
print("  Material-stratified SHAP...")
ci_idx = df.index[test_mask & (df["MATERIAL"] == "CI")]
pvc_idx = df.index[test_mask & (df["MATERIAL"] == "PVC")]
ci_s = X.loc[ci_idx].sample(min(500, len(ci_idx)), random_state=RANDOM_STATE)
pvc_s = X.loc[pvc_idx].sample(min(500, len(pvc_idx)), random_state=RANDOM_STATE)
shap_ci, shap_pvc = xgb_shap_values(xgb_model, ci_s), xgb_shap_values(xgb_model, pvc_s)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, sv, sample, title in [(axes[0], shap_ci, ci_s, "Cast Iron (CI)"),
                              (axes[1], shap_pvc, pvc_s, "PVC")]:
    disp = display_matrix(sample, FEATURE_NAMES)
    plt.sca(ax)
    shap.summary_plot(sv, disp, plot_type="bar", show=False, max_display=len(FEATURE_NAMES))
    ax.set_title(title)
    ax.set_xlabel("mean |SHAP value|")          # shorter label: avoids overlap
plt.suptitle("Material-Stratified Failure Drivers", fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(results_path("shap_ci_vs_pvc.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------
# Material-stratified Lorenz — full network, first test year
# ---------------------------------------------------------------------------
yr0 = fn_years[0]
snap = fn[fn["year"] == yr0].copy()
snap["prob"] = prob_fn[fn["year"].values == yr0]
print(f"\n  Material-stratified Lorenz (full network, {yr0}):")
mat_rows = []
for mat in snap["MATERIAL"].value_counts().head(10).index:
    m = snap["MATERIAL"] == mat
    if m.sum() < 100 or snap.loc[m, "target"].sum() < 5:
        continue
    lz, _, _ = compute_lorenz(snap.loc[m, "target"].values, snap.loc[m, "prob"].values,
                              weights=snap.loc[m, "LENGTH"].values)
    mat_rows.append({"Material": mat, "Pipes": int(m.sum()),
                     "Length_share": round(snap.loc[m, "LENGTH"].sum() / snap["LENGTH"].sum(), 4),
                     "Failures": int(snap.loc[m, "target"].sum()),
                     "Lorenz_1%": round(lz["1pct"], 4), "Lorenz_5%": round(lz["5pct"], 4),
                     "Lorenz_10%": round(lz["10pct"], 4)})
mat_df = pd.DataFrame(mat_rows).sort_values("Lorenz_5%", ascending=False)
print(mat_df.to_string(index=False))
mat_df.to_csv(results_path("material_lorenz.csv"), index=False)

# ---------------------------------------------------------------------------
# Top-100 pipes — operational illustration only
# ---------------------------------------------------------------------------
# Fixed mode: the list is ranked on the 2019 one-year risk score and checked
# against 2019 breaks only, so prediction and outcome use the same 1-year window.
# The 2019-2025 column is kept as descriptive context, not as a performance claim.
print(f"\n  Top-100 risky pipes (ranked {yr0})...")
broke_in_window = set(df_breaks.loc[df_breaks["BREAK_DATE"].between(
    TEST_START_YEAR, TEST_END_YEAR), "PIPE_ID"])
snap["risk_score"] = snap["prob"]
snap["broke_same_year"] = snap["target"].astype(int)
snap["broke_2019_2025"] = snap["PIPE_ID"].isin(broke_in_window).astype(int)
snap["confirmed"] = snap["broke_2019_2025"] if LEGACY else snap["broke_same_year"]

cols = ["PIPE_ID", "MATERIAL", "INSTALLED_YEAR", "age", "DIAM", "LENGTH", "ZONE",
        "SOIL_TYPE", "BREAKS_prev", "TSLF_prev", "risk_score",
        "broke_same_year", "broke_2019_2025", "confirmed"]
if "YEAR_MISSING" in snap.columns:
    cols.insert(3, "YEAR_MISSING")
top100 = snap.nlargest(100, "risk_score")[cols].round(4)

same_rate = snap["broke_same_year"].mean()
hits = int(top100["broke_same_year"].sum())
print(f"    Broke in {yr0}: {hits}/100 (network rate {same_rate*100:.3f}%, "
      f"lift {(hits/100)/same_rate:.1f}x)" if same_rate > 0 else f"    Broke in {yr0}: {hits}/100")
print(f"    Broke any time {TEST_START_YEAR}-{TEST_END_YEAR} (descriptive): "
      f"{int(top100['broke_2019_2025'].sum())}/100")
print(top100.head(10).to_string(index=False))
top100.to_csv(results_path("top100_risky_pipes.csv"), index=False)
print("Phase 8 done.")
