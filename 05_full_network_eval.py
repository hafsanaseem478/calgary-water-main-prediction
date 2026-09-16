"""
Phase 5 — Full-network annual-renewal evaluation (headline results).

Every in-service pipe is scored in each test year with the model frozen at
2018, then ranked within each year.

Outputs (interim/): full_network
Outputs (results/): full_network_scores.csv, full_network_summary.csv,
                    per_year_metrics.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from config import LEGACY, FEATURES, TEST_START_YEAR, TEST_END_YEAR, LORENZ_THRESHOLDS
from common import (banner, load_interim, load_meta, save_interim, feature_matrix,
                    xgb_matrix, encode_frame, annual_lorenz, results_path)
from panels import build_history_panel, legacy_full_network

warnings.filterwarnings("ignore")
banner("PHASE 5 — FULL-NETWORK EVALUATION")

all_pipes = load_interim("all_pipes")
df_breaks = load_interim("df_breaks")
annual = load_interim("annual")
lookups = load_interim("spatial_lookups")
encoders = load_interim("encoders")
models = load_interim("models")
meta = load_meta()
min_year, max_year = meta["min_year"], meta["max_year"]

test_years = np.arange(TEST_START_YEAR, TEST_END_YEAR + 1)
all_years = np.arange(min_year, max_year + 1)

print("  Building full-network panel (all pipes, test years)...")
if LEGACY:
    fn = legacy_full_network(all_pipes, df_breaks, annual, min_year, max_year,
                             meta["mode_diam"], lookups["soil_map"], lookups["traffic"],
                             test_years, all_years)
else:
    fn = build_history_panel(all_pipes, df_breaks, years=test_years, all_years=all_years,
                             annual=annual, soil_map=lookups["soil_map"],
                             traffic=lookups["traffic"])

fn = encode_frame(fn, encoders)
y_fn = fn["target"].values
print(f"  Rows: {len(fn):,}   Pipes: {fn['PIPE_ID'].nunique():,}")
print(f"  Positives: {y_fn.sum():,}   Base rate: {y_fn.mean()*100:.3f}%")

best_name = models["best_name"]
X_fn = (xgb_matrix(fn, FEATURES, encoders) if best_name == "XGBoost"
        else feature_matrix(fn, FEATURES))
prob_fn = models[best_name].predict_proba(X_fn)[:, 1]

# ---- pooled ----
auc_pooled = roc_auc_score(y_fn, prob_fn)
pr_pooled = average_precision_score(y_fn, prob_fn)
print(f"\n  Model: {best_name}")
print(f"  Pooled AUC-ROC: {auc_pooled:.4f}")
print(f"  Pooled PR-AUC:  {pr_pooled:.4f}   (lift vs base rate {y_fn.mean():.5f}: "
      f"{pr_pooled / y_fn.mean():.1f}x)")

# ---- annual-renewal Lorenz ----
fn_summ, fn_years, per_year_lz = annual_lorenz(fn, prob_fn, test_years, LORENZ_THRESHOLDS)
print(f"\n  Annual-renewal Lorenz (mean +/- sd over {fn_years[0]}-{fn_years[-1]}):")
for k, (mu, sd) in fn_summ.items():
    print(f"    Renew {k:>4} of network length -> capture {mu*100:5.1f}% +/- {sd*100:4.1f}%")

# ---- per year ----
rows = []
for i, yr in enumerate(fn_years):
    m = fn["year"].values == yr
    base_rate = y_fn[m].mean()
    ap = average_precision_score(y_fn[m], prob_fn[m])
    row = {"year": yr, "pipes": int(m.sum()), "failures": int(y_fn[m].sum()),
           "AUC-ROC": round(roc_auc_score(y_fn[m], prob_fn[m]), 4),
           "PR-AUC": round(ap, 4), "PR-AUC lift": round(ap / base_rate, 1)}
    for k in per_year_lz:
        row[f"Lorenz@{k.replace('pct', '%')}"] = round(per_year_lz[k][i] * 100, 1)
    rows.append(row)
per_year = pd.DataFrame(rows)
print("\n  Per-year (model frozen at 2018):")
print(per_year.to_string(index=False))

summary = pd.DataFrame([
    {"Metric": "Model", "Value": best_name},
    {"Metric": "Pipes", "Value": fn["PIPE_ID"].nunique()},
    {"Metric": "Pipe-years", "Value": len(fn)},
    {"Metric": "Failures", "Value": int(y_fn.sum())},
    {"Metric": "Base rate", "Value": round(y_fn.mean(), 6)},
    {"Metric": "Pooled AUC-ROC", "Value": round(auc_pooled, 4)},
    {"Metric": "Pooled PR-AUC", "Value": round(pr_pooled, 4)},
    {"Metric": "Pooled PR-AUC lift", "Value": round(pr_pooled / y_fn.mean(), 1)},
] + [
    {"Metric": f"Annual Lorenz@{k.replace('pct', '%')} mean", "Value": round(mu * 100, 1)}
    for k, (mu, _) in fn_summ.items()
] + [
    {"Metric": f"Annual Lorenz@{k.replace('pct', '%')} sd", "Value": round(sd * 100, 1)}
    for k, (_, sd) in fn_summ.items()
])

summary.to_csv(results_path("full_network_summary.csv"), index=False)
per_year.to_csv(results_path("per_year_metrics.csv"), index=False)
fn[["PIPE_ID", "year", "target"]].assign(prob=prob_fn).to_csv(
    results_path("full_network_scores.csv"), index=False)
save_interim({"fn": fn, "prob_fn": prob_fn, "fn_years": fn_years}, "full_network")
print("Phase 5 done.")
