"""
Phase 4 — Model selection and training.

Fixed mode (default):
  1. Fit XGBoost and CatBoost on years <= 2016.
  2. Compare them on the FULL NETWORK in the validation years 2017-2018
     (PR-AUC; annual-renewal Lorenz reported alongside).
  3. Retrain both on years <= 2018. The model chosen in step 2 is used for
     the 2019-2025 headline evaluation, so test years never influence the choice.

Legacy mode reproduces the original behaviour (selection on the 2019-2025
case-control test sample).

Outputs (interim/): models, test_predictions
Outputs (results/): model_comparison.csv, model_selection_validation.csv (fixed mode)
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (LEGACY, FEATURES, CAT_PARAMS, TRAIN_END_YEAR, TEST_START_YEAR,
                    TEST_END_YEAR, VALIDATION_START_YEAR)
from common import (banner, load_interim, load_meta, save_interim, save_meta,
                    feature_matrix, xgb_matrix, make_xgb, encode_frame,
                    split_masks, compute_lorenz, annual_lorenz, results_path)

warnings.filterwarnings("ignore")
banner("PHASE 4 — MODEL SELECTION AND TRAINING")

df = load_interim("df")
encoders = load_interim("encoders")
y = df["target"]
cat_features = [i for i, f in enumerate(FEATURES) if f.endswith("_enc")]


def fit_both(mask):
    """Train XGBoost and CatBoost on the rows in `mask`."""
    y_fit = y[mask]
    spw = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    xgb_m = make_xgb(spw)
    xgb_m.fit(xgb_matrix(df.loc[mask], FEATURES, encoders), y_fit)
    cat_m = CatBoostClassifier(cat_features=cat_features, **CAT_PARAMS)
    cat_m.fit(feature_matrix(df.loc[mask], FEATURES), y_fit)
    return xgb_m, cat_m, spw


def predict(name, model, frame):
    X = (xgb_matrix(frame, FEATURES, encoders) if name == "XGBoost"
         else feature_matrix(frame, FEATURES))
    return model.predict_proba(X)[:, 1]


train_mask, test_mask = split_masks(df)
y_tr, y_te = y[train_mask], y[test_mask]
print(f"  Train: <= {TRAIN_END_YEAR}, {train_mask.sum():,} rows, {y_tr.sum():,} pos ({y_tr.mean()*100:.2f}%)")
print(f"  Test:  {TEST_START_YEAR}-{TEST_END_YEAR}, {test_mask.sum():,} rows, {y_te.sum():,} pos ({y_te.mean()*100:.2f}%)")

# ---------------------------------------------------------------------------
# Step 1-2: selection on a validation period inside the training era
# ---------------------------------------------------------------------------
if not LEGACY:
    from panels import build_history_panel

    inner_mask = df["year"] < VALIDATION_START_YEAR
    val_years = np.arange(VALIDATION_START_YEAR, TRAIN_END_YEAR + 1)
    print(f"\n  Selection: fit on <= {VALIDATION_START_YEAR - 1}, "
          f"validate on full network {val_years[0]}-{val_years[-1]}")
    xgb_v, cat_v, _ = fit_both(inner_mask)

    meta = load_meta()
    lookups = load_interim("spatial_lookups")
    fn_val = build_history_panel(
        load_interim("all_pipes"), load_interim("df_breaks"), years=val_years,
        all_years=np.arange(meta["min_year"], meta["max_year"] + 1),
        annual=load_interim("annual"), soil_map=lookups["soil_map"],
        traffic=lookups["traffic"])
    fn_val = encode_frame(fn_val, encoders)
    y_val = fn_val["target"].values
    print(f"  Validation network: {len(fn_val):,} pipe-years, {y_val.sum():,} failures")

    sel_rows = []
    for name, model in [("XGBoost", xgb_v), ("CatBoost", cat_v)]:
        p = predict(name, model, fn_val)
        lz, _, _ = annual_lorenz(fn_val, p, val_years, min_pos=1)
        sel_rows.append({
            "Model": name,
            "PR-AUC (full network, validation)": round(average_precision_score(y_val, p), 4),
            "AUC-ROC (full network, validation)": round(roc_auc_score(y_val, p), 4),
            "AnnLorenz@1% (validation)": round(lz["1pct"][0] * 100, 1),
            "AnnLorenz@5% (validation)": round(lz["5pct"][0] * 100, 1),
            "AnnLorenz@10% (validation)": round(lz["10pct"][0] * 100, 1),
            "_pr": average_precision_score(y_val, p),
        })
    sel = pd.DataFrame(sel_rows)
    best_name = sel.loc[sel["_pr"].idxmax(), "Model"]
    sel = sel.drop(columns="_pr")
    sel["Selected"] = (sel["Model"] == best_name).astype(int)
    print("\n" + sel.to_string(index=False))
    print(f"\n  Selected on validation PR-AUC: {best_name}")
    sel.to_csv(results_path("model_selection_validation.csv"), index=False)

# ---------------------------------------------------------------------------
# Step 3: retrain on everything up to 2018
# ---------------------------------------------------------------------------
print(f"\n  Retraining both models on <= {TRAIN_END_YEAR}...")
xgb_model, cat_model, spw = fit_both(train_mask)
xgb_prob = predict("XGBoost", xgb_model, df.loc[test_mask])
cat_prob = predict("CatBoost", cat_model, df.loc[test_mask])

rows = []
for name, p in [("XGBoost", xgb_prob), ("CatBoost", cat_prob)]:
    lz, _, _ = compute_lorenz(y_te.values, p)
    rows.append({"Model": name,
                 "AUC-ROC (case-control test)": round(roc_auc_score(y_te, p), 4),
                 "PR-AUC (case-control test)": round(average_precision_score(y_te, p), 4),
                 "Lorenz@5% (count, case-control)": round(lz["5pct"] * 100, 1)})
comp = pd.DataFrame(rows)
print("\n  Case-control test sample (for reference, NOT used for selection):"
      if not LEGACY else "")
print(comp.to_string(index=False))

if LEGACY:
    # Original behaviour: select on the test sample.
    xgb_pr, cat_pr = average_precision_score(y_te, xgb_prob), average_precision_score(y_te, cat_prob)
    best_name = "CatBoost" if cat_pr > xgb_pr else "XGBoost"
    print(f"\nBest model (legacy: selected on test PR-AUC): {best_name}")

comp.to_csv(results_path("model_comparison.csv"), index=False)
save_interim({"XGBoost": xgb_model, "CatBoost": cat_model, "best_name": best_name,
              "scale_pos_weight": spw}, "models")
save_interim({"y_te": y_te.values, "xgb_prob": xgb_prob, "cat_prob": cat_prob,
              "best_prob": cat_prob if best_name == "CatBoost" else xgb_prob}, "test_predictions")
save_meta(best_model=best_name)
print("Phase 4 done.")
