"""
Phase 3 — Feature engineering and categorical encoding.

Outputs (interim/): df (panel with features), encoders
"""
import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import LEGACY, FEATURES, TRAIN_END_YEAR
from common import banner, load_interim, load_meta, save_interim, feature_matrix

warnings.filterwarnings("ignore")
banner("PHASE 3 — FEATURES")

panel = load_interim("panel")
sampling = load_interim("sampling")
meta = load_meta()

df = panel.copy().sort_values(["PIPE_ID", "year"], kind="mergesort").reset_index(drop=True)

if LEGACY:
    # Original: target and lagged features derived by shifting the panel.
    df["target"] = (df["BREAKED"] > 0).astype(int)
    df["BREAKS_prev"] = df.groupby("PIPE_ID")["BREAKS"].shift(1).fillna(0)
    df["TSLF_prev"] = df.groupby("PIPE_ID")["TSLF"].shift(1).fillna(0)
    df["broke_last_year"] = df.groupby("PIPE_ID")["target"].shift(1).fillna(0)
    df["has_broken_ever"] = (df.groupby("PIPE_ID")["BREAKED"].cumsum().shift(1).fillna(0) > 0).astype(int)
# Fixed mode: target and lagged history features were already built in phase 2
# by build_history_panel(), the same function used for full-network scoring.

encoders = {}
for col, enc_col in [("MATERIAL", "MATERIAL_enc"), ("TYPE", "TYPE_enc"),
                     ("ZONE", "ZONE_enc"), ("SOIL_TYPE", "SOIL_enc")]:
    le = LabelEncoder()
    df[enc_col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = feature_matrix(df, FEATURES)
print(f"  Features: {len(FEATURES)}")
print(f"  Positive rate: {df['target'].mean()*100:.3f}%")

# ---- Proxy-leakage check: install-year handling must not differ by group ----
is_case = df.groupby("PIPE_ID")["PIPE_ID"].first().isin(sampling["broken_by_train_end"])
if LEGACY:
    first_iy = df.groupby("PIPE_ID")["INSTALLED_YEAR"].first()
    print("  Install-year check (share defaulting to min_year):")
    print(f"    cases    (broke <= {TRAIN_END_YEAR}): {first_iy[is_case].eq(meta['min_year']).mean()*100:5.1f}%")
    print(f"    controls                : {first_iy[~is_case].eq(meta['min_year']).mean()*100:5.1f}%")
else:
    miss = df.groupby("PIPE_ID")["YEAR_MISSING"].first()
    iy = df.groupby("PIPE_ID")["INSTALLED_YEAR"].first()
    print("  Install-year check (same rule applied to all pipes):")
    print(f"    missing install year  cases: {miss[is_case].mean()*100:5.1f}%   "
          f"controls: {miss[~is_case].mean()*100:5.1f}%")
    print(f"    median install year   cases: {iy[is_case].median():.0f}      "
          f"controls: {iy[~is_case].median():.0f}")
    print("    -> differences here reflect real pipe populations, not construction")

save_interim(df, "df")
save_interim(encoders, "encoders")
print("Phase 3 done.")
