"""
Phase 1 — Load and link raw data.

  - water mains -> straight 2-point segments (one PIPE_ID each)
  - pressure zone for every segment
  - breaks matched to the nearest segment within 20 m
  - installation-year rule
  - annual weather table (loaded for completeness; not a model feature)
  - soil type and traffic lookups per segment

Outputs (interim/): all_pipes, df_breaks, annual, spatial_lookups, meta.json
"""
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (PATH_MAIN, PATH_BREAKS, PATH_ZONES, PATH_WEATHER, PATH_SOIL,
                    PATH_TRAFFIC, BREAK_MATCH_MAX_DIST_M, METRIC_CRS, LEGACY)
from common import banner, save_interim, save_meta
from geo import extract_line_segments, simplify_soil

warnings.filterwarnings("ignore")
banner("PHASE 1 — LOAD DATA")

# ---------------------------------------------------------------------------
# 1. Water mains
# ---------------------------------------------------------------------------
print("[1] Loading main pipes...")
mains = gpd.read_file(PATH_MAIN, GEOM_POSSIBLE_NAMES="MULTILINESTRING")
mains = mains.drop_duplicates()
mains["DIAM"] = mains["DIAM"].astype(str).str.replace(",", "").astype(float).astype(int)
mode_diam = int(mains["DIAM"].mode().iloc[0])
mains.loc[mains["DIAM"] == 0, "DIAM"] = mode_diam
mains.crs = "EPSG:4326"
mains = mains.to_crs(epsg=METRIC_CRS)

print("  Extracting segments...")
main_segments = extract_line_segments(mains)
main_segments["TYPE"] = "MAIN"
main_segments["YEAR"] = pd.to_datetime(main_segments["YEAR"], errors="coerce").dt.year
if "P_ZONE" in main_segments.columns:
    main_segments = main_segments.drop(columns=["P_ZONE"])

cols = ["MATERIAL", "YEAR", "DIAM", "TYPE", "LENGTH", "geometry"]
all_pipes = main_segments[cols].copy().reset_index(drop=True)
all_pipes["PIPE_ID"] = all_pipes.index
all_pipes = gpd.GeoDataFrame(all_pipes, geometry="geometry", crs=f"EPSG:{METRIC_CRS}")
all_pipes["X"] = all_pipes.geometry.centroid.x
all_pipes["Y"] = all_pipes.geometry.centroid.y
print(f"  {len(all_pipes):,} pipe segments")

# ---------------------------------------------------------------------------
# 2. Pressure zones
# ---------------------------------------------------------------------------
print("[2] Assigning pressure zones...")
zones = gpd.read_file(PATH_ZONES, GEOM_POSSIBLE_NAMES="MULTIPOLYGON")
zones.crs = "EPSG:4326"
zones = zones.to_crs(epsg=METRIC_CRS)
zoned = gpd.sjoin_nearest(all_pipes, zones[["ZONE", "geometry"]],
                          how="left", distance_col="zone_dist")
zoned = zoned.sort_values("zone_dist").drop_duplicates(subset="PIPE_ID", keep="first")
all_pipes["ZONE"] = all_pipes["PIPE_ID"].map(
    dict(zip(zoned["PIPE_ID"], zoned["ZONE"]))).fillna("UNKNOWN")
print(f"  Real zones: {(all_pipes['ZONE'] != 'UNKNOWN').sum():,}/{len(all_pipes):,}")

# ---------------------------------------------------------------------------
# 3. Breaks -> nearest pipe
# ---------------------------------------------------------------------------
print("[3] Matching breaks to pipes...")
breaks = gpd.read_file(PATH_BREAKS, GEOM_POSSIBLE_NAMES="point")
breaks = breaks.drop_duplicates()
for bt in ["A", "B", "C", "D", "E", "F", "G", "S"]:
    breaks[bt] = breaks["BREAK_TYPE"].str.contains(bt, na=False).astype(int)
breaks.crs = "EPSG:4326"
breaks = breaks.to_crs(epsg=METRIC_CRS)

matched = gpd.sjoin_nearest(
    breaks,
    all_pipes[["PIPE_ID", "MATERIAL", "YEAR", "DIAM", "TYPE", "LENGTH",
               "X", "Y", "ZONE", "geometry"]],
    distance_col="distance", how="inner", max_distance=BREAK_MATCH_MAX_DIST_M,
    lsuffix="brk", rsuffix="pipe",
)
print(f"  {len(matched):,} break-pipe pairs, median distance {matched['distance'].median():.2f} m")
matched = matched.drop(columns=["distance", "index_pipe"], errors="ignore")

df_breaks = pd.DataFrame({
    "PIPE_ID": matched["PIPE_ID"].astype(int),
    "BREAK_DATE": pd.to_datetime(matched["BREAK_DATE"], errors="coerce").dt.year,
    "A": matched["A"], "B": matched["B"], "C": matched["C"], "D": matched["D"],
    "E": matched["E"], "F": matched["F"], "G": matched["G"], "S": matched["S"],
    "MATERIAL": matched["MATERIAL"], "INSTALLED_YEAR": matched["YEAR"],
    "DIAM": matched["DIAM"], "TYPE": matched["TYPE"], "LENGTH": matched["LENGTH"],
    "ZONE": matched["ZONE"], "X": matched["X"], "Y": matched["Y"],
})

if LEGACY:
    # Original behaviour: breaks on pipes with a missing install year are dropped.
    df_breaks = df_breaks.dropna(subset=["BREAK_DATE", "INSTALLED_YEAR"])
    df_breaks[["BREAK_DATE", "INSTALLED_YEAR"]] = df_breaks[["BREAK_DATE", "INSTALLED_YEAR"]].astype(int)
else:
    df_breaks = df_breaks.dropna(subset=["BREAK_DATE"])
    df_breaks["BREAK_DATE"] = df_breaks["BREAK_DATE"].astype(int)

df_breaks = df_breaks.sort_values(["PIPE_ID", "BREAK_DATE"]).reset_index(drop=True)
type_diam = df_breaks.groupby("TYPE")["DIAM"].apply(
    lambda x: x.mode().iloc[0] if len(x.mode()) else x.median()).to_dict()
df_breaks.loc[df_breaks["DIAM"].isna(), "DIAM"] = (
    df_breaks.loc[df_breaks["DIAM"].isna(), "TYPE"].map(type_diam))

min_year = int(df_breaks["BREAK_DATE"].min())
max_year = int(df_breaks["BREAK_DATE"].max())

# ---------------------------------------------------------------------------
# 3b. Installation-year rule
# ---------------------------------------------------------------------------
if not LEGACY:
    # ONE rule for every pipe (cases, controls and the full network):
    #   known year   -> kept as recorded (capped at the last data year);
    #                   pre-1956 pipes keep their true age
    #   missing year -> first year of the break record, flagged in YEAR_MISSING
    all_pipes["YEAR_MISSING"] = all_pipes["YEAR"].isna().astype(int)
    all_pipes["INSTALLED_YEAR"] = (all_pipes["YEAR"].fillna(min_year)
                                   .clip(upper=max_year).astype(int))
    df_breaks["INSTALLED_YEAR"] = df_breaks["PIPE_ID"].map(
        dict(zip(all_pipes["PIPE_ID"], all_pipes["INSTALLED_YEAR"])))

    # A break dated before -- or in -- the year the current pipe was installed
    # cannot be attributed to this pipe: break dates are year-level, and a break
    # in the installation year is most often the failure on the old main that
    # triggered the replacement. These breaks belong to an earlier asset.
    pre_install = df_breaks["BREAK_DATE"] <= df_breaks["INSTALLED_YEAR"]
    n_same = (df_breaks["BREAK_DATE"] == df_breaks["INSTALLED_YEAR"]).sum()
    print(f"  Breaks dated in/before the matched pipe's install year (dropped): "
          f"{pre_install.sum():,}  (of which in the install year: {n_same:,})")
    df_breaks = df_breaks[~pre_install].reset_index(drop=True)
    print(f"  Pipes with missing install year: {all_pipes['YEAR_MISSING'].sum():,} "
          f"({all_pipes['YEAR_MISSING'].mean()*100:.1f}%)")

print(f"  {len(df_breaks):,} breaks on {df_breaks['PIPE_ID'].nunique():,} pipes, {min_year}-{max_year}")

# ---------------------------------------------------------------------------
# 4. Weather (loaded for completeness; year-level, NOT a model feature)
# ---------------------------------------------------------------------------
print("[4] Processing weather...")
weather = pd.read_csv(PATH_WEATHER, low_memory=False)
weather["date"] = pd.to_datetime(weather["date"])
weather = weather[weather["date"] >= pd.to_datetime(f"{min_year}-01-01")].set_index("date")
n = max_year - min_year + 1
annual = pd.DataFrame({
    "year": range(min_year, max_year + 1),
    "avg_temp": weather["avg_temperature"].resample("YE").mean().values[:n],
    "std_tmp": weather["avg_temperature"].resample("YE").std().values[:n],
    "HDD": weather["heatdegdays"].resample("YE").sum().values[:n],
    "CDD": weather["cooldegdays"].resample("YE").sum().values[:n],
    "precipitation": weather["precipitation"].resample("YE").sum().values[:n],
    "rain": weather["rain"].resample("YE").sum().values[:n],
    "snow": weather["snow_on_ground"].resample("YE").sum().values[:n],
}).set_index("year")
annual.iloc[-1] = annual.iloc[-2].values
print(f"  Weather: {min_year}-{max_year}")

# ---------------------------------------------------------------------------
# 5. Soil and traffic lookups (per pipe)
# ---------------------------------------------------------------------------
print("[5] Soil and traffic lookups...")
pipe_pts = gpd.GeoDataFrame(
    all_pipes[["PIPE_ID", "X", "Y"]],
    geometry=gpd.points_from_xy(all_pipes["X"], all_pipes["Y"]),
    crs=f"EPSG:{METRIC_CRS}",
)

soil_map = None
if os.path.exists(PATH_SOIL):
    soil = gpd.read_file(PATH_SOIL).to_crs(epsg=METRIC_CRS)
    soil["SOIL_SIMPLE"] = soil["GENESIS"].apply(simplify_soil)
    j = gpd.sjoin(pipe_pts, soil[["SOIL_SIMPLE", "geometry"]], how="left", predicate="within")
    miss = j["SOIL_SIMPLE"].isna()
    if miss.sum() > 0:
        nearest = gpd.sjoin_nearest(pipe_pts[miss], soil[["SOIL_SIMPLE", "geometry"]], how="left")
        j.loc[miss, "SOIL_SIMPLE"] = nearest["SOIL_SIMPLE"].values
    soil_map = dict(zip(j["PIPE_ID"], j["SOIL_SIMPLE"].fillna("Unknown")))
    print(f"  Soil types: {len(set(soil_map.values()))}")
else:
    print("  Soil file not found — soil set to 'Unknown'")

# Traffic is loaded but NOT used as a feature: ~3% coverage (counts cover
# arterials, not the residential streets where most mains run).
traffic = None
if os.path.exists(PATH_TRAFFIC):
    tr_df = pd.read_csv(PATH_TRAFFIC, low_memory=False)
    tr_df["Volume"] = pd.to_numeric(tr_df["Volume"].astype(str).str.replace(",", ""), errors="coerce")
    tr = gpd.GeoDataFrame(tr_df, geometry=gpd.GeoSeries.from_wkt(tr_df["multilinestring"]),
                          crs="EPSG:4326").to_crs(epsg=METRIC_CRS).dropna(subset=["Volume"])
    near = gpd.sjoin_nearest(pipe_pts, tr[["Volume", "geometry"]],
                             distance_col="dist_to_road", how="left", max_distance=200)
    traffic = {
        "volume": dict(zip(near["PIPE_ID"], near["Volume"].fillna(0))),
        "dist": dict(zip(near["PIPE_ID"], near["dist_to_road"].fillna(999))),
    }
    cov = np.mean([v > 0 for v in traffic["volume"].values()]) * 100
    print(f"  Traffic: {cov:.1f}% coverage")
else:
    print("  Traffic file not found — traffic set to 0")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_interim(all_pipes, "all_pipes")
save_interim(df_breaks, "df_breaks")
save_interim(annual, "annual")
save_interim({"soil_map": soil_map, "traffic": traffic}, "spatial_lookups")
save_meta(min_year=min_year, max_year=max_year, mode_diam=mode_diam)
print("Phase 1 done.")
