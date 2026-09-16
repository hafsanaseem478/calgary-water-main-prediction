"""
Phase 2 — Sampling frame and pipe-year training panel.

Cases:    pipes with a break recorded on or before the training cutoff (2018).
Controls: random sample (3 per case) from every other pipe, using no
          test-period information. Sampled controls that fail in 2019-2025
          keep their real target=1 in those years.

Outputs (interim/): panel, sampling
"""
import numpy as np
import warnings

from config import (LEGACY, TRAIN_END_YEAR, TEST_START_YEAR, TEST_END_YEAR,
                    NEGATIVE_SAMPLING_RATIO, RANDOM_STATE)
from common import banner, load_interim, load_meta, save_interim, results_path
from panels import build_history_panel, legacy_case_control_panel

warnings.filterwarnings("ignore")
banner("PHASE 2 — BUILD TRAINING PANEL")

all_pipes = load_interim("all_pipes")
df_breaks = load_interim("df_breaks")
annual = load_interim("annual")
lookups = load_interim("spatial_lookups")
meta = load_meta()
min_year, max_year, mode_diam = meta["min_year"], meta["max_year"], meta["mode_diam"]

if LEGACY:
    panel, broken_by_train_end, sampled_ids = legacy_case_control_panel(
        all_pipes, df_breaks, annual, min_year, max_year, mode_diam,
        lookups["soil_map"], lookups["traffic"],
        TRAIN_END_YEAR, TEST_START_YEAR, TEST_END_YEAR,
        NEGATIVE_SAMPLING_RATIO, RANDOM_STATE)
else:
    broken_by_train_end = set(
        df_breaks.loc[df_breaks["BREAK_DATE"] <= TRAIN_END_YEAR, "PIPE_ID"].unique())
    candidate_neg_ids = np.array(sorted(set(all_pipes["PIPE_ID"].values) - broken_by_train_end))
    n_sample = min(len(broken_by_train_end) * NEGATIVE_SAMPLING_RATIO, len(candidate_neg_ids))
    np.random.seed(RANDOM_STATE)
    sampled_ids = np.random.choice(candidate_neg_ids, size=n_sample, replace=False)

    test_breakers = set(df_breaks.loc[df_breaks["BREAK_DATE"].between(
        TEST_START_YEAR, TEST_END_YEAR), "PIPE_ID"])
    print(f"  Cases (broke <= {TRAIN_END_YEAR}):     {len(broken_by_train_end):,}")
    print(f"  Controls sampled:               {len(sampled_ids):,}")
    print(f"    -> fail in {TEST_START_YEAR}-{TEST_END_YEAR}:       "
          f"{len(set(sampled_ids) & test_breakers):,}")

    keep_ids = broken_by_train_end | set(sampled_ids)
    pipes = all_pipes[all_pipes["PIPE_ID"].isin(keep_ids)]
    panel = build_history_panel(
        pipes, df_breaks,
        years=np.arange(min_year, max_year + 1),
        all_years=np.arange(min_year, max_year + 1),
        annual=annual, soil_map=lookups["soil_map"], traffic=lookups["traffic"])

print(f"  Panel: {len(panel):,} rows, {panel['PIPE_ID'].nunique():,} pipes")
print(f"  ZONE=UNKNOWN: {(panel['ZONE'] == 'UNKNOWN').mean()*100:.2f}%")
print(f"  Break rate: {(panel['BREAKED'] > 0).mean()*100:.3f}%")

save_interim(panel, "panel")
save_interim({"broken_by_train_end": broken_by_train_end, "sampled_ids": sampled_ids}, "sampling")
panel.to_csv(results_path("calgary_panel_final.csv"), index=False)
print("Phase 2 done.")
