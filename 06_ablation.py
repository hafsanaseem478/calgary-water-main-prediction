"""
Phase 6 — Ablation on break-history features.

Three XGBoost variants (same model family, same settings): full feature set,
no break history, break history + age only. Each is evaluated on the
case-control test sample AND on the full network, so the columns are labelled.

Outputs (results/): ablation_study.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from config import FEATURES, HISTORY_FEATURES, TEST_START_YEAR, TEST_END_YEAR
from common import (banner, load_interim, xgb_matrix, make_xgb, split_masks,
                    annual_lorenz, results_path)

warnings.filterwarnings("ignore")
banner("PHASE 6 — ABLATION")

df = load_interim("df")
encoders = load_interim("encoders")
full = load_interim("full_network")
fn = full["fn"]
test_years = np.arange(TEST_START_YEAR, TEST_END_YEAR + 1)

train_mask, test_mask = split_masks(df)
y_tr, y_te = df.loc[train_mask, "target"], df.loc[test_mask, "target"]
y_fn = fn["target"].values
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

features_no_hist = [f for f in FEATURES if f not in HISTORY_FEATURES]
features_hist_age = HISTORY_FEATURES + ["age"]

rows = []
for label, feats in [
    (f"Full model ({len(FEATURES)} features)", FEATURES),
    (f"No break history ({len(features_no_hist)} features)", features_no_hist),
    (f"History + age only ({len(features_hist_age)} features)", features_hist_age),
]:
    print(f"\n  Training: {label}...")
    m = make_xgb(spw)
    m.fit(xgb_matrix(df.loc[train_mask], feats, encoders), y_tr)

    p_te = m.predict_proba(xgb_matrix(df.loc[test_mask], feats, encoders))[:, 1]
    p_fn = m.predict_proba(xgb_matrix(fn, feats, encoders))[:, 1]
    summ, _, _ = annual_lorenz(fn, p_fn, test_years)

    row = {
        "Feature set": label,
        "AUC-ROC (case-control test)": round(roc_auc_score(y_te, p_te), 4),
        "PR-AUC (case-control test)": round(average_precision_score(y_te, p_te), 4),
        "AUC-ROC (full network)": round(roc_auc_score(y_fn, p_fn), 4),
        "PR-AUC (full network)": round(average_precision_score(y_fn, p_fn), 4),
        "AnnLorenz@1% (fn)": round(summ["1pct"][0] * 100, 1),
        "AnnLorenz@5% (fn)": round(summ["5pct"][0] * 100, 1),
        "AnnLorenz@10% (fn)": round(summ["10pct"][0] * 100, 1),
    }
    rows.append(row)
    print(f"    full-network AUC={row['AUC-ROC (full network)']:.4f}  "
          f"AnnLorenz@5%={row['AnnLorenz@5% (fn)']:.1f}%")

abl = pd.DataFrame(rows)
print("\nABLATION SUMMARY:")
print(abl.to_string(index=False))
abl.to_csv(results_path("ablation_study.csv"), index=False)
print("Phase 6 done.")
