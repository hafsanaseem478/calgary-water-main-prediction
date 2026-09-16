"""
Pipe-year panel builders.

build_history_panel()   — used in PANEL_MODE='fixed' for BOTH the training
                          panel and full-network scoring, so every feature
                          means the same thing at training and prediction time.

legacy_*()              — the original pipeline.py logic, kept verbatim so
                          PANEL_MODE='legacy' reproduces the earlier results.
"""
import numpy as np
import pandas as pd

PIPE_COLS = ["PIPE_ID", "MATERIAL", "INSTALLED_YEAR", "DIAM", "TYPE",
             "LENGTH", "X", "Y", "ZONE"]
BREAK_TYPES = ["A", "B", "C", "D", "E", "F", "G", "S"]


def _clean_length(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce").fillna(0)


def _wide_to_long(wide, name):
    long = wide.reset_index().melt(id_vars="PIPE_ID", var_name="year", value_name=name)
    long["year"] = long["year"].astype(int)
    return long


def _attach_context(frame, annual, soil_map, traffic):
    """Weather (not a model feature), soil, traffic (not a model feature)."""
    if annual is not None:
        frame = frame.merge(annual, left_on="year", right_index=True, how="left")
    frame["SOIL_TYPE"] = (frame["PIPE_ID"].map(soil_map).fillna("Unknown")
                          if soil_map is not None else "Unknown")
    if traffic is not None:
        frame["TRAFFIC_VOLUME"] = frame["PIPE_ID"].map(traffic["volume"]).fillna(0)
        frame["DIST_TO_ROAD"] = frame["PIPE_ID"].map(traffic["dist"]).fillna(999)
    else:
        frame["TRAFFIC_VOLUME"] = 0
        frame["DIST_TO_ROAD"] = 999
    return frame


# ===========================================================================
# FIXED MODE
# ===========================================================================
def build_history_panel(pipes, df_breaks, years, all_years,
                        annual=None, soil_map=None, traffic=None):
    """One row per pipe per in-service year, with the 1-year target and lagged
    break-history features.

    Conventions (identical for every pipe):
      - a pipe enters the risk set the year AFTER installation: at the start of
        year t, only pipes in service by the end of t-1 are known to a utility
        (the mains layer is a later snapshot, so pipes installed in year t would
        otherwise be scored before they exist)
      - rows start at max(installation year + 1, first year in `years`)
      - age            = year - INSTALLED_YEAR
      - BREAKS_prev    = breaks recorded strictly before `year`
      - TSLF_prev      = years since the most recent break before `year`,
                         measured at year-1; if no earlier break, age - 1
      - target         = 1 if at least one break recorded in `year`
    """
    years = np.asarray(years, dtype=int)
    all_years = np.asarray(all_years, dtype=int)

    base = pd.DataFrame(pipes[PIPE_COLS + ["YEAR_MISSING"]]).copy()
    base["LENGTH"] = _clean_length(base["LENGTH"])

    out = base.loc[base.index.repeat(len(years))].reset_index(drop=True)
    out["year"] = np.tile(years, len(base))
    out = out[out["year"] > out["INSTALLED_YEAR"]].reset_index(drop=True)
    out["age"] = out["year"] - out["INSTALLED_YEAR"]

    brk = df_breaks[df_breaks["PIPE_ID"].isin(base["PIPE_ID"])]
    if len(brk):
        bk = (brk.groupby(["PIPE_ID", "BREAK_DATE"]).size()
              .unstack(fill_value=0)
              .reindex(columns=all_years, fill_value=0))
        bk.index.name = "PIPE_ID"
        cum_prev = bk.cumsum(axis=1).shift(1, axis=1).fillna(0)
        brk_prev1 = bk.shift(1, axis=1).fillna(0)
        last = pd.DataFrame(np.where(bk.values > 0, all_years[None, :], np.nan),
                            index=bk.index, columns=all_years).ffill(axis=1)
        last_prev = last.shift(1, axis=1)
        for wide, name in [(bk, "BREAKED"), (cum_prev, "BREAKS_prev"),
                           (brk_prev1, "_bl"), (last_prev, "_last_prev")]:
            out = out.merge(_wide_to_long(wide, name), on=["PIPE_ID", "year"], how="left")
    else:
        out["BREAKED"], out["BREAKS_prev"], out["_bl"], out["_last_prev"] = 0, 0, 0, np.nan

    out["BREAKED"] = out["BREAKED"].fillna(0)
    out["BREAKS_prev"] = out["BREAKS_prev"].fillna(0)
    out["target"] = (out["BREAKED"] > 0).astype(int)
    out["broke_last_year"] = (out["_bl"].fillna(0) > 0).astype(int)
    out["has_broken_ever"] = (out["BREAKS_prev"] > 0).astype(int)
    out["TSLF_prev"] = np.where(out["_last_prev"].notna(),
                                (out["year"] - 1) - out["_last_prev"],
                                out["age"] - 1)
    out["TSLF_prev"] = out["TSLF_prev"].clip(lower=0)
    out = out.drop(columns=["_bl", "_last_prev"])

    return _attach_context(out, annual, soil_map, traffic)


# ===========================================================================
# LEGACY MODE — original pipeline.py logic
# ===========================================================================
def legacy_case_control_panel(all_pipes, df_breaks, annual, min_year, max_year,
                              mode_diam, soil_map, traffic,
                              train_end_year, test_start_year, test_end_year,
                              negative_sampling_ratio, random_state):
    """Original Stages 5-7: per-pipe loop for broken pipes, then controls."""
    # ---- Stage 5: broken pipes ----
    print("  Building panel for broken pipes (~20 min)...")
    panel_broken_list = []
    n_pipes = df_breaks["PIPE_ID"].nunique()
    for i, (pid, pipe_df) in enumerate(df_breaks.groupby("PIPE_ID")):
        seq = pipe_df.copy()
        brk_dates, brk_counts = np.unique(seq["BREAK_DATE"], return_counts=True)
        brk_map = dict(zip(brk_dates, brk_counts))
        seq["BREAKED"] = [brk_map[d] for d in seq["BREAK_DATE"]]
        seq = seq.drop_duplicates(subset="BREAK_DATE", keep="first").set_index("BREAK_DATE")

        template = seq.iloc[0].copy()
        template[BREAK_TYPES + ["BREAKED"]] = 0

        inst_year = int(seq["INSTALLED_YEAR"].iloc[0])
        seq = seq[seq.index >= inst_year]
        obs_years = seq.index

        seq = seq.reindex(range(min_year, max_year + 1))
        seq.loc[~seq.index.isin(obs_years)] = template.values
        seq["BREAKS"] = seq["BREAKED"].cumsum()
        seq["TSLF"] = seq.groupby(np.cumsum(seq["BREAKED"] > 0)).cumcount()
        seq.loc[seq["BREAKED"] > 0, "TSLF"] = 0

        seq.index.name = "year"
        seq = seq.reset_index().merge(annual, left_on="year", right_index=True, how="inner")
        panel_broken_list.append(seq)
        if (i + 1) % 3000 == 0:
            print(f"    {i+1}/{n_pipes}")

    panel_broken = pd.concat(panel_broken_list, axis=0, ignore_index=True)
    panel_broken["age"] = (panel_broken["year"] - panel_broken["INSTALLED_YEAR"]).clip(lower=0)
    panel_broken["LENGTH"] = _clean_length(panel_broken["LENGTH"])
    print(f"  Broken panel: {len(panel_broken):,} rows, {panel_broken['PIPE_ID'].nunique():,} pipes")

    # ---- Stage 6: negative sampling ----
    broken_by_train_end = set(
        df_breaks.loc[df_breaks["BREAK_DATE"] <= train_end_year, "PIPE_ID"].unique())
    candidate_neg_ids = np.array(sorted(set(all_pipes["PIPE_ID"].values) - broken_by_train_end))
    n_sample = min(len(broken_by_train_end) * negative_sampling_ratio, len(candidate_neg_ids))
    np.random.seed(random_state)
    sampled_ids = np.random.choice(candidate_neg_ids, size=n_sample, replace=False)

    broken_ids = set(panel_broken["PIPE_ID"].unique())
    future_breakers = sorted(set(sampled_ids) & broken_ids)
    true_never = np.array(sorted(set(sampled_ids) - broken_ids))

    keep_ids = broken_by_train_end | set(sampled_ids)
    panel_broken = panel_broken[panel_broken["PIPE_ID"].isin(keep_ids)].copy()

    print(f"  Cases (broke <= {train_end_year}):        {len(broken_by_train_end):,}")
    print(f"  Controls sampled:                  {len(sampled_ids):,}")
    print(f"    -> fail in {test_start_year}-{test_end_year}:          {len(future_breakers):,}")
    print(f"    -> never fail:                   {len(true_never):,}")

    nb = all_pipes.loc[true_never].copy()
    nb["INSTALLED_YEAR"] = pd.to_numeric(nb["YEAR"], errors="coerce").fillna(min_year).astype(int)
    nb["INSTALLED_YEAR"] = nb["INSTALLED_YEAR"].clip(lower=min_year, upper=max_year)

    years = np.arange(min_year, max_year + 1)
    nb_expanded = nb.loc[nb.index.repeat(len(years))].reset_index(drop=True)
    nb_expanded["year"] = np.tile(years, len(nb))
    nb_expanded = nb_expanded[nb_expanded["year"] >= nb_expanded["INSTALLED_YEAR"]].copy()
    nb_expanded["age"] = (nb_expanded["year"] - nb_expanded["INSTALLED_YEAR"]).clip(lower=0)
    nb_expanded["TSLF"] = nb_expanded["age"]
    nb_expanded["BREAKED"] = 0
    nb_expanded["BREAKS"] = 0
    for c in BREAK_TYPES:
        nb_expanded[c] = 0
    nb_expanded["LENGTH"] = _clean_length(nb_expanded["LENGTH"])
    nb_expanded["DIAM"] = pd.to_numeric(nb_expanded["DIAM"], errors="coerce").fillna(mode_diam)
    nb_expanded = nb_expanded.merge(annual, left_on="year", right_index=True, how="left")

    common = [c for c in panel_broken.columns if c in nb_expanded.columns]
    panel = pd.concat([panel_broken[common], nb_expanded[common]], axis=0, ignore_index=True)

    # ---- Stage 7: soil & traffic ----
    panel = _attach_context(panel, None, soil_map, traffic)
    return panel, broken_by_train_end, sampled_ids


def legacy_full_network(all_pipes, df_breaks, annual, min_year, max_year, mode_diam,
                        soil_map, traffic, test_years, all_years):
    """Original full-network test panel construction."""
    base = all_pipes[["PIPE_ID", "MATERIAL", "YEAR", "DIAM", "TYPE",
                      "LENGTH", "X", "Y", "ZONE"]].copy()
    base["INSTALLED_YEAR"] = (pd.to_numeric(base["YEAR"], errors="coerce")
                              .fillna(min_year).astype(int)
                              .clip(lower=min_year, upper=max_year))
    base["LENGTH"] = _clean_length(base["LENGTH"])
    base["DIAM"] = pd.to_numeric(base["DIAM"], errors="coerce").fillna(mode_diam)

    fn = base.loc[base.index.repeat(len(test_years))].reset_index(drop=True)
    fn["year"] = np.tile(test_years, len(base))
    fn = fn[fn["year"] >= fn["INSTALLED_YEAR"]].reset_index(drop=True)
    fn["age"] = (fn["year"] - fn["INSTALLED_YEAR"]).clip(lower=0)

    bk = (df_breaks.groupby(["PIPE_ID", "BREAK_DATE"]).size()
          .unstack(fill_value=0)
          .reindex(columns=all_years, fill_value=0))
    bk.index.name = "PIPE_ID"
    cum_prev = bk.cumsum(axis=1).shift(1, axis=1).fillna(0)
    brk_prev1 = bk.shift(1, axis=1).fillna(0)
    _lastyr = pd.DataFrame(np.where(bk.values > 0, all_years[None, :], np.nan),
                           index=bk.index, columns=all_years).ffill(axis=1)
    last_prev = _lastyr.shift(1, axis=1)

    for wide, name in [(bk, "BREAKED"), (cum_prev, "BREAKS_prev"),
                       (brk_prev1, "_bl"), (last_prev, "_last_prev")]:
        fn = fn.merge(_wide_to_long(wide, name), on=["PIPE_ID", "year"], how="left")

    fn["BREAKED"] = fn["BREAKED"].fillna(0)
    fn["BREAKS_prev"] = fn["BREAKS_prev"].fillna(0)
    fn["target"] = (fn["BREAKED"] > 0).astype(int)
    fn["broke_last_year"] = (fn["_bl"].fillna(0) > 0).astype(int)
    fn["has_broken_ever"] = (fn["BREAKS_prev"] > 0).astype(int)
    fn["TSLF_prev"] = np.where(fn["_last_prev"].notna(),
                               (fn["year"] - 1) - fn["_last_prev"],
                               fn["age"] - 1)
    fn["TSLF_prev"] = fn["TSLF_prev"].clip(lower=0)
    for c in BREAK_TYPES:
        fn[c] = 0
    return _attach_context(fn, annual, soil_map, traffic)
