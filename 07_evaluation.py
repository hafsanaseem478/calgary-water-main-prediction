"""
Phase 7 — Evaluation figures and robustness checks.

  - Lorenz curve for the first test year (full network)
  - bootstrap confidence intervals (full network, first test year)
  - spatial block cross-validation (pressure zones as blocks, training sample)
  - ROC / PR curves and MCC thresholds (case-control test sample)

Outputs (results/): lorenz_curve.png, bootstrap_ci.csv, spatial_cv.csv,
                    roc_pr_curves.png, mcc_thresholds.csv
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

from config import (LEGACY, FEATURES, TRAIN_END_YEAR, N_BOOTSTRAP,
                    N_SPATIAL_FOLDS, RANDOM_STATE)
from common import banner, load_interim, xgb_matrix, make_xgb, split_masks, results_path

warnings.filterwarnings("ignore")
banner("PHASE 7 — EVALUATION")

df = load_interim("df")
encoders = load_interim("encoders")
models = load_interim("models")
preds = load_interim("test_predictions")
full = load_interim("full_network")
fn, prob_fn, fn_years = full["fn"], full["prob_fn"], full["fn_years"]
best_name = models["best_name"]

# ---------------------------------------------------------------------------
# Lorenz curve — full network, first test year
# ---------------------------------------------------------------------------
yr0 = fn_years[0]
m0 = fn["year"].values == yr0
p0, yt0, w0 = prob_fn[m0], fn["target"].values[m0], fn["LENGTH"].values[m0]
order0 = np.argsort(p0)[::-1]
cn0 = np.cumsum(w0[order0]) / w0.sum()
cf0 = np.cumsum(yt0[order0]) / max(yt0.sum(), 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(cn0 * 100, cf0 * 100, linewidth=2, color="#2196F3", label=f"{best_name} ({yr0} network)")
ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Random")
for r in [1, 5, 10]:
    idx = min(np.searchsorted(cn0, r / 100), len(cf0) - 1)
    ax.scatter([r], [cf0[idx] * 100], s=80, zorder=5)
    ax.annotate(f"{cf0[idx]*100:.1f}%", (r, cf0[idx] * 100),
                textcoords="offset points", xytext=(10, 0), fontsize=10)
ax.set_xlabel("% of network renewed (by length)")
ax.set_ylabel("% of failures captured")
ax.set_title(f"Annual-Renewal Lorenz — full network, {yr0} (model frozen at <= {TRAIN_END_YEAR})")
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0, 50]); ax.set_ylim([0, 100])
plt.tight_layout(); plt.savefig(results_path("lorenz_curve.png"), dpi=150); plt.close()
print(f"  Lorenz curve saved ({yr0})")

# ---------------------------------------------------------------------------
# Bootstrap CI — full network, first test year
# ---------------------------------------------------------------------------
print(f"\n  Bootstrap CI ({N_BOOTSTRAP} resamples, full network {yr0})...")
rng = np.random.default_rng(RANDOM_STATE)
boot = {"AUC-ROC": [], "PR-AUC": [], "P@100": [], "Lorenz@5%": []}
n_snap = len(yt0)
for _ in range(N_BOOTSTRAP):
    idx = rng.choice(n_snap, n_snap, replace=True)
    yb, pb, wb = yt0[idx], p0[idx], w0[idx]
    if yb.sum() < 2:
        continue
    boot["AUC-ROC"].append(roc_auc_score(yb, pb))
    boot["PR-AUC"].append(average_precision_score(yb, pb))
    o = np.argsort(pb)[::-1]
    boot["P@100"].append(yb[o[:100]].sum() / 100)
    cl = np.cumsum(wb[o]) / wb.sum()
    cfl = np.cumsum(yb[o]) / yb.sum()
    boot["Lorenz@5%"].append(cfl[min(np.searchsorted(cl, 0.05), len(cfl) - 1)])
ci = pd.DataFrame([{"Metric": k, "Mean": round(np.mean(v), 4),
                    "2.5%": round(np.percentile(v, 2.5), 4),
                    "97.5%": round(np.percentile(v, 97.5), 4)} for k, v in boot.items()])
print(ci.to_string(index=False))
ci.to_csv(results_path("bootstrap_ci.csv"), index=False)

# ---------------------------------------------------------------------------
# Spatial block CV — pressure zones as blocks, training sample
# ---------------------------------------------------------------------------
# Fixed mode: pressure zone is removed from the features for this test (a
# held-out zone is by construction unseen, so its code carries no meaning),
# and class weights are computed from each fold's training rows only.
cv_features = FEATURES if LEGACY else [f for f in FEATURES if f != "ZONE_enc"]
print(f"\n  Spatial block CV ({N_SPATIAL_FOLDS}-fold, blocks = pressure zones, "
      f"{len(cv_features)} features)...")
train_mask, _ = split_masks(df)
X_tr = xgb_matrix(df.loc[train_mask], cv_features, encoders).reset_index(drop=True)
y_tr = df.loc[train_mask, "target"].values
groups = df.loc[train_mask, "ZONE_enc"].values

cv_rows = []
for fold, (tr_i, val_i) in enumerate(GroupKFold(n_splits=N_SPATIAL_FOLDS).split(X_tr, y_tr, groups)):
    if y_tr[val_i].sum() < 5:
        continue
    if LEGACY:
        spw = models["scale_pos_weight"]           # original: whole training set
    else:
        spw = (y_tr[tr_i] == 0).sum() / max((y_tr[tr_i] == 1).sum(), 1)
    m = make_xgb(spw)
    if LEGACY:
        m.fit(X_tr.values[tr_i], y_tr[tr_i])
        pf = m.predict_proba(X_tr.values[val_i])[:, 1]
    else:
        m.fit(X_tr.iloc[tr_i], y_tr[tr_i])
        pf = m.predict_proba(X_tr.iloc[val_i])[:, 1]
    cv_rows.append({"fold": fold + 1, "zones_held_out": len(np.unique(groups[val_i])),
                    "val_rows": len(val_i), "val_failures": int(y_tr[val_i].sum()),
                    "AUC-ROC": round(roc_auc_score(y_tr[val_i], pf), 4),
                    "PR-AUC": round(average_precision_score(y_tr[val_i], pf), 4)})
    print(f"    Fold {fold+1}: AUC={cv_rows[-1]['AUC-ROC']:.4f}")
cv = pd.DataFrame(cv_rows)
if len(cv):
    mean_row = {"fold": "mean", "AUC-ROC": round(cv["AUC-ROC"].mean(), 4),
                "PR-AUC": round(cv["PR-AUC"].mean(), 4)}
    sd_row = {"fold": "sd", "AUC-ROC": round(cv["AUC-ROC"].std(ddof=0), 4),
              "PR-AUC": round(cv["PR-AUC"].std(ddof=0), 4)}
    print(f"  Spatial CV AUC: {mean_row['AUC-ROC']:.4f} +/- {sd_row['AUC-ROC']:.4f}")
    cv = pd.concat([cv, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
cv.to_csv(results_path("spatial_cv.csv"), index=False)

# ---------------------------------------------------------------------------
# ROC / PR and MCC — case-control test sample (enriched in failures)
# ---------------------------------------------------------------------------
y_te = preds["y_te"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, p, color in [("XGBoost", preds["xgb_prob"], "#2196F3"),
                       ("CatBoost", preds["cat_prob"], "#FF5722")]:
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})", color=color, linewidth=2)
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.3f})", color=color, linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC — case-control test sample")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall — case-control test sample")
for a in axes:
    a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(results_path("roc_pr_curves.png"), dpi=150); plt.close()

prob = preds["best_prob"]
mcc_rows = []
for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    yp = (prob >= t).astype(int)
    if yp.sum() == 0:
        continue
    mcc_rows.append({"Threshold": t, "MCC": round(matthews_corrcoef(y_te, yp), 4),
                     "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
                     "Recall": round(recall_score(y_te, yp), 4), "Flagged": int(yp.sum())})
mcc = pd.DataFrame(mcc_rows)
print("\n  MCC across thresholds (case-control test sample):")
print(mcc.to_string(index=False))
mcc.to_csv(results_path("mcc_thresholds.csv"), index=False)
print("Phase 7 done.")
