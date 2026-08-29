"""
Interpretable ML for Water Main Failure Prediction — Calgary
XGBoost + CatBoost with SHAP, spatial block CV, an ablation on break-history
features, and a full-network annual-renewal Lorenz evaluation.

Evaluation design
-----------------
- Sampling frame is fixed at the train cutoff: cases are pipes that broke on or
  before 2018 (identifiable at decision time); controls are sampled from the
  remainder. Pipes that only fail in 2019-2025 are NOT excluded from the control
  pool — that would select controls on the test outcome and inflate every metric.
- Temporal split: train <= 2018, test 2019-2025 (near-future validation, the
  regime that matters operationally; random splits look great but do not
  generalise — Kitchener repository, WDSA/CCWI 2024).
- Headline metric is the annual-renewal Lorenz curve: each year, rank every
  in-service pipe, renew the top X% of TOTAL network length, measure failures
  captured; averaged across test years. This is the evaluation of Forero-Ortiz
  et al. (Applied Water Science, 2026); their Barcelona figures (10.32% @1%,
  30.2% @5%) are quoted as a methodological reference, not a head-to-head claim.
- AUC-ROC is reported but not led on: at <1% base rate it is dominated by the
  negative class (Saito & Rehmsmeier, 2015). PR-AUC lift and the Lorenz are the
  operationally meaningful numbers.

Usage:
  1. Place data files in data/ folder (see README for download links)
  2. Unzip soil shapefile into data/soil_data/
  3. Run: python pipeline.py
  4. Results saved to results/
"""

import os, warnings
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pandas as pd
import geopandas as gpd
from shapely import LineString
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             matthews_corrcoef)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive: never blocks on plt.show() (Windows-safe)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
plt.rcParams['figure.dpi'] = 110
np.random.seed(42)

# CONFIG — update file paths to match your data/ folder
PATH_MAIN    = "data/Public_Water_Main_20240621.csv"
PATH_BREAKS  = "data/Water_Main_Breaks_20260726.csv"
PATH_ZONES   = "data/Water_Pressure_Zones_20260728.csv"
PATH_WEATHER = "data/weatherstats_calgary_daily.csv"
PATH_SOIL    = "data/soil_data/surf_py_ll.shp"
PATH_TRAFFIC = "data/Traffic_Volumes_for_2023_20260729.csv"

PREDICTION_WINDOW = 1
TEST_START_YEAR = 2019
TEST_END_YEAR = 2025
NEGATIVE_SAMPLING_RATIO = 3

def extract_line_segments(gdf):
    """Split MultiLineStrings into straight LineString segments."""
    rows = []
    for _, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None:
            continue
        lines = []
        if geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        elif geom.geom_type == 'LineString':
            lines = [geom]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                r = row.copy()
                r['geometry'] = LineString(coords[i:i+2])
                rows.append(r)
    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)


def simplify_soil(genesis):
    """Classify Alberta Geological Survey genesis codes into broad soil types."""
    if pd.isna(genesis):
        return 'Unknown'
    g = str(genesis).lower()
    if any(k in g for k in ['till', 'glacial', 'mudflow']):
        return 'Glacial_Till'
    if any(k in g for k in ['fluvial', 'channel', 'river']):
        return 'Fluvial'
    if any(k in g for k in ['lacustrine', 'lake', 'pond']):
        return 'Lacustrine'
    if 'eolian' in g:
        return 'Eolian'
    if 'bedrock' in g:
        return 'Bedrock'
    if any(k in g for k in ['colluvial', 'slope']):
        return 'Colluvial'
    if any(k in g for k in ['organic', 'peat']):
        return 'Organic'
    if any(k in g for k in ['alluvial', 'flood']):
        return 'Alluvial'
    return 'Other'


def compute_lorenz(y_true, y_prob, weights=None):
    """Compute Lorenz curve values (Barcelona 2026 primary metric)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights)
    order = np.argsort(y_prob)[::-1]
    cum_net = np.cumsum(w[order]) / w.sum()
    cum_fail = np.cumsum(y_true[order]) / max(y_true.sum(), 1)
    vals = {}
    for r in [0.01, 0.05, 0.10, 0.25]:
        idx = min(np.searchsorted(cum_net, r), len(cum_fail) - 1)
        vals[f'{int(r*100)}pct'] = cum_fail[idx]
    if hasattr(np, 'trapezoid'):
        vals['auc'] = np.trapezoid(cum_fail, cum_net)
    else:
        vals['auc'] = np.trapz(cum_fail, cum_net)
    return vals, cum_net, cum_fail


def annual_lorenz(frame, prob, years, thresholds=(0.01, 0.05, 0.10), min_pos=5):
    """Barcelona-style operational metric (Forero-Ortiz et al. 2026).

    Simulates an ANNUAL renewal decision: for each year, rank every in-service
    pipe by predicted probability, walk down the ranking accumulating pipe LENGTH
    until a given fraction of TOTAL network length is reached, and record the
    fraction of that year's failures captured. Each year is one clean snapshot
    (one row per pipe), then results are averaged across years.

    This avoids the multiple-counting that occurs if several years are pooled into
    a single ranking (a pipe scored high every year would otherwise contribute its
    length many times over). Returns {label: (mean, std)} and the years used.
    """
    prob = np.asarray(prob)
    per_year = {f'{int(t*100)}pct': [] for t in thresholds}
    used = []
    for yr in years:
        m = (frame['year'].values == yr)
        yt = frame['target'].values[m]
        if yt.sum() < min_pos:
            continue
        pr = prob[m]
        w = frame['LENGTH'].values[m]
        order = np.argsort(pr)[::-1]
        cum_len = np.cumsum(w[order]) / w.sum()
        cum_fail = np.cumsum(yt[order]) / yt.sum()
        for t in thresholds:
            idx = min(np.searchsorted(cum_len, t), len(cum_fail) - 1)
            per_year[f'{int(t*100)}pct'].append(cum_fail[idx])
        used.append(int(yr))
    summ = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_year.items() if v}
    return summ, used


# STAGE 1: LOAD MAIN PIPES
print('[1/11] Loading main pipes...')
mains = gpd.read_file(PATH_MAIN, GEOM_POSSIBLE_NAMES='MULTILINESTRING')
mains = mains.drop_duplicates()
mains['DIAM'] = mains['DIAM'].astype(str).str.replace(',', '').astype(float).astype(int)
mode_diam = mains['DIAM'].mode().iloc[0]
mains.loc[mains['DIAM'] == 0, 'DIAM'] = mode_diam
mains.crs = 'EPSG:4326'
mains = mains.to_crs(epsg=3762)

print('  Extracting segments...')
main_segments = extract_line_segments(mains)
main_segments['TYPE'] = 'MAIN'
main_segments['YEAR'] = pd.to_datetime(main_segments['YEAR'], errors='coerce').dt.year
if 'P_ZONE' in main_segments.columns:
    main_segments = main_segments.drop(columns=['P_ZONE'])

cols = ['MATERIAL', 'YEAR', 'DIAM', 'TYPE', 'LENGTH', 'geometry']
all_pipes = main_segments[cols].copy().reset_index(drop=True)
all_pipes['PIPE_ID'] = all_pipes.index
all_pipes = gpd.GeoDataFrame(all_pipes, geometry='geometry', crs='EPSG:3762')
all_pipes['X'] = all_pipes.geometry.centroid.x
all_pipes['Y'] = all_pipes.geometry.centroid.y
print(f'  {len(all_pipes)} pipe segments')

# STAGE 2: ASSIGN REAL PRESSURE ZONES TO ALL PIPES
print('[2/11] Assigning pressure zones to ALL pipes...')
zones = gpd.read_file(PATH_ZONES, GEOM_POSSIBLE_NAMES='MULTIPOLYGON')
zones.crs = 'EPSG:4326'
zones = zones.to_crs(epsg=3762)

all_pipes_zoned = gpd.sjoin_nearest(
    all_pipes, zones[['ZONE', 'geometry']],
    how='left', distance_col='zone_dist'
)
# Drop duplicates — keep closest zone per pipe
all_pipes_zoned = all_pipes_zoned.sort_values('zone_dist').drop_duplicates(
    subset='PIPE_ID', keep='first'
)
all_pipes['ZONE'] = all_pipes['PIPE_ID'].map(
    dict(zip(all_pipes_zoned['PIPE_ID'], all_pipes_zoned['ZONE']))
).fillna('UNKNOWN')
print(f'  Real zones: {(all_pipes["ZONE"] != "UNKNOWN").sum()}/{len(all_pipes)}')

# STAGE 3: SPATIAL JOIN — BREAKS TO PIPES
print('[3/11] Matching breaks to pipes...')
breaks = gpd.read_file(PATH_BREAKS, GEOM_POSSIBLE_NAMES='point')
breaks = breaks.drop_duplicates()
for bt in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S']:
    breaks[bt] = breaks['BREAK_TYPE'].str.contains(bt, na=False).astype(int)
breaks.crs = 'EPSG:4326'
breaks = breaks.to_crs(epsg=3762)

matched = gpd.sjoin_nearest(
    breaks,
    all_pipes[['PIPE_ID', 'MATERIAL', 'YEAR', 'DIAM', 'TYPE', 'LENGTH',
               'X', 'Y', 'ZONE', 'geometry']],
    distance_col='distance', how='inner', max_distance=20,
    lsuffix='brk', rsuffix='pipe'
)
print(f'  {len(matched)} pairs, median dist {matched["distance"].median():.2f}m')
matched = matched.drop(columns=['distance', 'index_pipe'])

df_breaks = pd.DataFrame({
    'PIPE_ID': matched['PIPE_ID'].astype(int),
    'BREAK_DATE': pd.to_datetime(matched['BREAK_DATE'], errors='coerce').dt.year,
    'A': matched['A'], 'B': matched['B'], 'C': matched['C'], 'D': matched['D'],
    'E': matched['E'], 'F': matched['F'], 'G': matched['G'], 'S': matched['S'],
    'MATERIAL': matched['MATERIAL'], 'INSTALLED_YEAR': matched['YEAR'],
    'DIAM': matched['DIAM'], 'TYPE': matched['TYPE'], 'LENGTH': matched['LENGTH'],
    'ZONE': matched['ZONE'], 'X': matched['X'], 'Y': matched['Y']
}).dropna(subset=['BREAK_DATE', 'INSTALLED_YEAR'])
df_breaks[['BREAK_DATE', 'INSTALLED_YEAR']] = df_breaks[['BREAK_DATE', 'INSTALLED_YEAR']].astype(int)
df_breaks = df_breaks.sort_values(['PIPE_ID', 'BREAK_DATE']).reset_index(drop=True)

# Fill missing DIAM
type_diam = df_breaks.groupby('TYPE')['DIAM'].apply(
    lambda x: x.mode().iloc[0] if len(x.mode()) else x.median()
).to_dict()
df_breaks.loc[df_breaks['DIAM'].isna(), 'DIAM'] = (
    df_breaks.loc[df_breaks['DIAM'].isna(), 'TYPE'].map(type_diam)
)
print(f'  {df_breaks["PIPE_ID"].nunique()} unique broken pipes')

# STAGE 4: WEATHER DATA
# NOTE: weather features are loaded but NOT used in the final model.
# Year-level weather has no cross-pipe variation within a year, so it cannot
# discriminate in the annual-renewal ranking. Kept in the panel for potential
# future work with pipe-level frost penetration.
print('[4/11] Processing weather...')
weather = pd.read_csv(PATH_WEATHER, low_memory=False)
weather['date'] = pd.to_datetime(weather['date'])
min_year = int(df_breaks['BREAK_DATE'].min())
max_year = int(df_breaks['BREAK_DATE'].max())
weather = weather[weather['date'] >= pd.to_datetime(f'{min_year}-01-01')].set_index('date')

annual = pd.DataFrame({
    'year': range(min_year, max_year + 1),
    'avg_temp': weather['avg_temperature'].resample('YE').mean().values[:max_year - min_year + 1],
    'std_tmp': weather['avg_temperature'].resample('YE').std().values[:max_year - min_year + 1],
    'HDD': weather['heatdegdays'].resample('YE').sum().values[:max_year - min_year + 1],
    'CDD': weather['cooldegdays'].resample('YE').sum().values[:max_year - min_year + 1],
    'precipitation': weather['precipitation'].resample('YE').sum().values[:max_year - min_year + 1],
    'rain': weather['rain'].resample('YE').sum().values[:max_year - min_year + 1],
    'snow': weather['snow_on_ground'].resample('YE').sum().values[:max_year - min_year + 1],
}).set_index('year')
annual.iloc[-1] = annual.iloc[-2].values
print(f'  Weather: {min_year}-{max_year}')

# STAGE 5: BUILD PANEL — BROKEN PIPES
print('[5/11] Building panel for broken pipes (~20 min)...')
panel_broken_list = []
n_pipes = df_breaks['PIPE_ID'].nunique()

for i, (pid, pipe_df) in enumerate(df_breaks.groupby('PIPE_ID')):
    seq = pipe_df.copy()
    brk_dates, brk_counts = np.unique(seq['BREAK_DATE'], return_counts=True)
    brk_map = dict(zip(brk_dates, brk_counts))
    seq['BREAKED'] = [brk_map[d] for d in seq['BREAK_DATE']]
    seq = seq.drop_duplicates(subset='BREAK_DATE', keep='first').set_index('BREAK_DATE')

    template = seq.iloc[0].copy()
    template[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S', 'BREAKED']] = 0

    inst_year = int(seq['INSTALLED_YEAR'].iloc[0])
    seq = seq[seq.index >= inst_year]
    obs_years = seq.index

    seq = seq.reindex(range(min_year, max_year + 1))
    seq.loc[~seq.index.isin(obs_years)] = template.values
    seq['BREAKS'] = seq['BREAKED'].cumsum()
    seq['TSLF'] = seq.groupby(np.cumsum(seq['BREAKED'] > 0)).cumcount()
    seq.loc[seq['BREAKED'] > 0, 'TSLF'] = 0

    seq.index.name = 'year'
    seq = seq.reset_index().merge(annual, left_on='year', right_index=True, how='inner')
    panel_broken_list.append(seq)

    if (i + 1) % 3000 == 0:
        print(f'  {i+1}/{n_pipes}')

panel_broken = pd.concat(panel_broken_list, axis=0, ignore_index=True)
panel_broken['age'] = (panel_broken['year'] - panel_broken['INSTALLED_YEAR']).clip(lower=0)
panel_broken['LENGTH'] = pd.to_numeric(
    panel_broken['LENGTH'].astype(str).str.replace(',', ''), errors='coerce'
).fillna(0)
print(f'  Broken panel: {len(panel_broken):,} rows, {panel_broken["PIPE_ID"].nunique():,} pipes')

# STAGE 6: NEGATIVE SAMPLING — WITH REAL ZONES
print('[6/11] Adding never-broken pipes (sampling frame fixed at train cutoff)...')
 
TRAIN_END_YEAR = TEST_START_YEAR - 1          # 2018
 
# Cases: pipes with a break recorded ON OR BEFORE the train cutoff.
# These are identifiable using only information the model is allowed to have.
broken_by_train_end = set(
    df_breaks.loc[df_breaks['BREAK_DATE'] <= TRAIN_END_YEAR, 'PIPE_ID'].unique()
)
 
# Controls: sampled from everything else. No test-period information used.
candidate_neg_ids = np.array(sorted(set(all_pipes['PIPE_ID'].values) - broken_by_train_end))
n_sample = min(len(broken_by_train_end) * NEGATIVE_SAMPLING_RATIO, len(candidate_neg_ids))
np.random.seed(42)
sampled_ids = np.random.choice(candidate_neg_ids, size=n_sample, replace=False)
 
broken_ids = set(panel_broken['PIPE_ID'].unique())   # every pipe that ever breaks
 
# Some sampled controls DO fail in 2019-2025. They stay in the panel with their
# real target=1 in that year. This is the whole point — the test set now contains
# failures the model was not told about through sample membership.
future_breakers = sorted(set(sampled_ids) & broken_ids)
true_never      = np.array(sorted(set(sampled_ids) - broken_ids))
 
# Test-only breakers that were NOT sampled must be dropped. Keeping them would be
# selecting cases on the test outcome again.
keep_ids = broken_by_train_end | set(sampled_ids)
panel_broken = panel_broken[panel_broken['PIPE_ID'].isin(keep_ids)].copy()
 
print(f'  Cases (broke <= {TRAIN_END_YEAR}):        {len(broken_by_train_end):,}')
print(f'  Controls sampled:                  {len(sampled_ids):,}')
print(f'    -> fail in {TEST_START_YEAR}-{TEST_END_YEAR}:          {len(future_breakers):,}')
print(f'    -> never fail:                   {len(true_never):,}')
 
# Controls are the genuinely-never-broken pipes only. The future_breakers
# (sampled pipes that fail in 2019-2025) already live in panel_broken with their
# real break rows, so they must NOT be re-added here as target=0 — doing that was
# duplicating them with contradictory labels.
nb = all_pipes.loc[true_never].copy()
nb['INSTALLED_YEAR'] = pd.to_numeric(nb['YEAR'], errors='coerce').fillna(min_year).astype(int)
nb['INSTALLED_YEAR'] = nb['INSTALLED_YEAR'].clip(lower=min_year, upper=max_year)

years = np.arange(min_year, max_year + 1)
nb_expanded = nb.loc[nb.index.repeat(len(years))].reset_index(drop=True)
nb_expanded['year'] = np.tile(years, len(nb))
nb_expanded = nb_expanded[nb_expanded['year'] >= nb_expanded['INSTALLED_YEAR']].copy()

nb_expanded['age'] = (nb_expanded['year'] - nb_expanded['INSTALLED_YEAR']).clip(lower=0)
nb_expanded['TSLF'] = nb_expanded['age']
nb_expanded['BREAKED'] = 0
nb_expanded['BREAKS'] = 0
for c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S']:
    nb_expanded[c] = 0
# ZONE comes from all_pipes
nb_expanded['LENGTH'] = pd.to_numeric(
    nb_expanded['LENGTH'].astype(str).str.replace(',', ''), errors='coerce'
).fillna(0)
nb_expanded['DIAM'] = pd.to_numeric(nb_expanded['DIAM'], errors='coerce').fillna(mode_diam)
nb_expanded = nb_expanded.merge(annual, left_on='year', right_index=True, how='left')

common = [c for c in panel_broken.columns if c in nb_expanded.columns]
panel = pd.concat([panel_broken[common], nb_expanded[common]], axis=0, ignore_index=True)

print(f'  Panel: {len(panel):,} rows, {panel["PIPE_ID"].nunique():,} pipes')
print(f'  ZONE=UNKNOWN: {(panel["ZONE"]=="UNKNOWN").sum()} ({(panel["ZONE"]=="UNKNOWN").mean()*100:.2f}%)')
print(f'  Break rate: {(panel["BREAKED"]>0).mean()*100:.3f}%')

# STAGE 7: ADD SOIL & TRAFFIC
# NOTE: soil IS used in the model. Traffic is loaded but NOT used — coverage is
# only ~3% (Calgary's traffic count dataset covers major arterials, not the
# residential streets where most water mains run), leaving 97% of pipes at
# TRAFFIC_VOLUME=0. A near-constant feature contributes nothing to ranking.
print('[7/11] Adding soil and traffic...')
panel.to_csv('results/calgary_panel_final.csv', index=False)
# Pipe points for spatial joins
pipe_pts = gpd.GeoDataFrame(
    all_pipes[['PIPE_ID', 'X', 'Y']],
    geometry=gpd.points_from_xy(all_pipes['X'], all_pipes['Y']),
    crs='EPSG:3762'
)

# SOIL
if os.path.exists(PATH_SOIL):
    soil = gpd.read_file(PATH_SOIL).to_crs(epsg=3762)
    soil['SOIL_SIMPLE'] = soil['GENESIS'].apply(simplify_soil)
    j = gpd.sjoin(pipe_pts, soil[['SOIL_SIMPLE', 'geometry']], how='left', predicate='within')
    miss = j['SOIL_SIMPLE'].isna()
    if miss.sum() > 0:
        n = gpd.sjoin_nearest(pipe_pts[miss], soil[['SOIL_SIMPLE', 'geometry']], how='left')
        j.loc[miss, 'SOIL_SIMPLE'] = n['SOIL_SIMPLE'].values
    soil_map = dict(zip(j['PIPE_ID'], j['SOIL_SIMPLE'].fillna('Unknown')))
    panel['SOIL_TYPE'] = panel['PIPE_ID'].map(soil_map).fillna('Unknown')
    print(f'  Soil types: {panel["SOIL_TYPE"].nunique()}')
else:
    panel['SOIL_TYPE'] = 'Unknown'
    print('  Soil file not found — skipped')

# TRAFFIC
if os.path.exists(PATH_TRAFFIC):
    tr_df = pd.read_csv(PATH_TRAFFIC, low_memory=False)
    tr_df['Volume'] = pd.to_numeric(
        tr_df['Volume'].astype(str).str.replace(',', ''), errors='coerce'
    )
    tr = gpd.GeoDataFrame(
        tr_df,
        geometry=gpd.GeoSeries.from_wkt(tr_df['multilinestring']),
        crs='EPSG:4326'
    ).to_crs(epsg=3762).dropna(subset=['Volume'])
    near = gpd.sjoin_nearest(
        pipe_pts, tr[['Volume', 'geometry']],
        distance_col='dist_to_road', how='left', max_distance=200
    )
    panel['TRAFFIC_VOLUME'] = panel['PIPE_ID'].map(
        dict(zip(near['PIPE_ID'], near['Volume'].fillna(0)))
    ).fillna(0)
    panel['DIST_TO_ROAD'] = panel['PIPE_ID'].map(
        dict(zip(near['PIPE_ID'], near['dist_to_road'].fillna(999)))
    ).fillna(999)
    print(f'  Traffic: {(panel["TRAFFIC_VOLUME"]>0).mean()*100:.1f}% coverage')
else:
    panel['TRAFFIC_VOLUME'] = 0
    panel['DIST_TO_ROAD'] = 999
    print('  Traffic file not found — skipped')

panel.to_csv('results/calgary_panel_final.csv', index=False)
print(f'  Panel saved: {panel.shape}')

# STAGE 8: FEATURE ENGINEERING
print('[8/11] Feature engineering...')
df = panel.copy()
df = df.sort_values(['PIPE_ID', 'year'], kind='mergesort').reset_index(drop=True)

# 1-year target
df['target'] = (df['BREAKED'] > 0).astype(int)

# Lagged features — prevent target leakage
df['BREAKS_prev'] = df.groupby('PIPE_ID')['BREAKS'].shift(1).fillna(0)
df['TSLF_prev'] = df.groupby('PIPE_ID')['TSLF'].shift(1).fillna(0)
df['broke_last_year'] = df.groupby('PIPE_ID')['target'].shift(1).fillna(0)
df['has_broken_ever'] = (df.groupby('PIPE_ID')['BREAKED'].cumsum().shift(1).fillna(0) > 0).astype(int)

# Encode categoricals
le_mat = LabelEncoder(); df['MATERIAL_enc'] = le_mat.fit_transform(df['MATERIAL'].astype(str))
le_type = LabelEncoder(); df['TYPE_enc'] = le_type.fit_transform(df['TYPE'].astype(str))
le_zone = LabelEncoder(); df['ZONE_enc'] = le_zone.fit_transform(df['ZONE'].astype(str))
le_soil = LabelEncoder(); df['SOIL_enc'] = le_soil.fit_transform(df['SOIL_TYPE'].astype(str))

features = [
    # pipe intrinsics
    'MATERIAL_enc', 'DIAM', 'LENGTH', 'age',
    # location: engineering context + raw coordinates (tie-breakers)
    'ZONE_enc', 'SOIL_enc', 'X', 'Y',
    # break history
    'TSLF_prev', 'BREAKS_prev',
]
feature_names = [
    'Material', 'Diameter', 'Length', 'Age',
    'Pressure Zone', 'Soil Type', 'X (Easting)', 'Y (Northing)',
    'TSLF (prev yr)', 'Cum. Breaks (prev yr)',
]
# Removed after feature-review pass (see README):
#   - INSTALLED_YEAR : perfect linear dependence with `age` within a year.
#   - TYPE_enc       : Calgary open data exposes only mains ('MAIN' for every
#                      pipe), so this is a constant column with zero variance
#                      and zero SHAP importance.
#   - has_broken_ever: exact binarisation of BREAKS_prev>0. Pure duplication.
#   - broke_last_year: recency signal already carried more granularly by TSLF_prev.
#   - TRAFFIC_VOLUME : 97% zero — no discriminative power.
#   - DIST_TO_ROAD   : same source; near-constant.
#   - A..S dummies   : encode PAST break TYPE; zeroed at prediction time in `fn`,
#                      so they leak break-history under a different name. History
#                      is already captured cleanly by TSLF_prev and BREAKS_prev.
#   - weather (avg_temp, std_tmp, HDD, CDD, precipitation, rain, snow):
#                      year-level data, identical across every pipe within a year.
#                      The annual-renewal metric ranks pipes WITHIN a year, so a
#                      constant column contributes zero to that ranking. Local
#                      frost-penetration would help, but wasn't available.

X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['target']

print(f'  Features: {len(features)}')
print(f'  Positive rate: {y.mean()*100:.3f}%')

# Proxy-leakage guard: install year is taken from break records for cases and from
# the mains layer (missing -> min_year) for controls. If the share defaulting to
# min_year differs a lot between groups, INSTALLED_YEAR/age partly encodes the label.
_first_iy = df.groupby('PIPE_ID')['INSTALLED_YEAR'].first()
_is_case = _first_iy.index.isin(broken_by_train_end)
print('  Install-year missingness check (share defaulting to min_year):')
print(f'    cases    (broke <= {TRAIN_END_YEAR}): {_first_iy[_is_case].eq(min_year).mean()*100:5.1f}%')
print(f'    controls                : {_first_iy[~_is_case].eq(min_year).mean()*100:5.1f}%')
print('    -> large gap would mean age is a label proxy; similar shares = OK')

# STAGE 9: TEMPORAL SPLIT
usable_train_end = TEST_START_YEAR - 1  # 2018
usable_test_end = TEST_END_YEAR         # 2025

train_mask = df['year'] <= usable_train_end
test_mask = (df['year'] >= TEST_START_YEAR) & (df['year'] <= usable_test_end)

X_tr, X_te = X[train_mask], X[test_mask]
y_tr, y_te = y[train_mask], y[test_mask]

print(f'  Train: ≤{usable_train_end}, {len(X_tr):,} rows, {y_tr.sum():,} pos ({y_tr.mean()*100:.2f}%)')
print(f'  Test:  {TEST_START_YEAR}-{usable_test_end}, {len(X_te):,} rows, {y_te.sum():,} pos ({y_te.mean()*100:.2f}%)')

# STAGE 10: TRAIN XGBOOST + CATBOOST
print('[9/11] Training models...')

# --- XGBoost ---
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, eval_metric='aucpr', random_state=42,
    n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_tr, y_tr)
xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
xgb_pred = xgb_model.predict(X_te)

xgb_auc = roc_auc_score(y_te, xgb_prob)
xgb_pr = average_precision_score(y_te, xgb_prob)
print(f'  XGBoost:  AUC={xgb_auc:.4f}  PR-AUC={xgb_pr:.4f}')

# --- CatBoost ---
cat_feature_indices = [i for i, f in enumerate(features) if f.endswith('_enc')]
cat_model = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05,
    auto_class_weights='Balanced', loss_function='Logloss',
    eval_metric='PRAUC', random_seed=42, verbose=100,
    cat_features=cat_feature_indices
)
cat_model.fit(X_tr, y_tr)
cat_prob = cat_model.predict_proba(X_te)[:, 1]
cat_pred = cat_model.predict(X_te).astype(int)

cat_auc = roc_auc_score(y_te, cat_prob)
cat_pr = average_precision_score(y_te, cat_prob)
print(f'  CatBoost: AUC={cat_auc:.4f}  PR-AUC={cat_pr:.4f}')

# --- Pick best ---
# Select on PR-AUC: it is the imbalance-appropriate metric (Saito & Rehmsmeier 2015),
# and AUC-ROC differences between these two models are typically within noise here.
if cat_pr > xgb_pr:
    best_model, prob, pred, best_name = cat_model, cat_prob, cat_pred, 'CatBoost'
else:
    best_model, prob, pred, best_name = xgb_model, xgb_prob, xgb_pred, 'XGBoost'
if abs(cat_auc - xgb_auc) < 0.005:
    print(f'  NOTE: AUC-ROC gap ({abs(cat_auc - xgb_auc):.4f}) is within noise — '
          f'selecting on PR-AUC instead.')

# --- Comparison table ---
lz_xgb, _, _ = compute_lorenz(y_te.values, xgb_prob)
lz_cat, _, _ = compute_lorenz(y_te.values, cat_prob)

print(f'\n{"="*60}')
print(f'{"Metric":<20} {"XGBoost":>10} {"CatBoost":>10}')
print(f'{"AUC-ROC":<20} {xgb_auc:>10.4f} {cat_auc:>10.4f}')
print(f'{"PR-AUC":<20} {xgb_pr:>10.4f} {cat_pr:>10.4f}')
print(f'{"Lorenz@1%":<20} {lz_xgb["1pct"]*100:>9.1f}% {lz_cat["1pct"]*100:>9.1f}%')
print(f'{"Lorenz@5%":<20} {lz_xgb["5pct"]*100:>9.1f}% {lz_cat["5pct"]*100:>9.1f}%')
print(f'{"Lorenz@10%":<20} {lz_xgb["10pct"]*100:>9.1f}% {lz_cat["10pct"]*100:>9.1f}%')
print(f'\nBest model: {best_name}')
print('\nBuilding full-network test panel (all pipes, 2019-2025)...')
 
test_years = np.arange(TEST_START_YEAR, TEST_END_YEAR + 1)
all_years  = np.arange(min_year, max_year + 1)
 
# ---- static attributes for every pipe ----
base = all_pipes[['PIPE_ID', 'MATERIAL', 'YEAR', 'DIAM', 'TYPE',
                  'LENGTH', 'X', 'Y', 'ZONE']].copy()
base['INSTALLED_YEAR'] = (pd.to_numeric(base['YEAR'], errors='coerce')
                          .fillna(min_year).astype(int)
                          .clip(lower=min_year, upper=max_year))
base['LENGTH'] = pd.to_numeric(
    base['LENGTH'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
base['DIAM'] = pd.to_numeric(base['DIAM'], errors='coerce').fillna(mode_diam)
 
fn = base.loc[base.index.repeat(len(test_years))].reset_index(drop=True)
fn['year'] = np.tile(test_years, len(base))
fn = fn[fn['year'] >= fn['INSTALLED_YEAR']].reset_index(drop=True)
fn['age'] = (fn['year'] - fn['INSTALLED_YEAR']).clip(lower=0)
 
# ---- break-history matrices (pipe x year), used to build lagged features ----
bk = (df_breaks.groupby(['PIPE_ID', 'BREAK_DATE']).size()
      .unstack(fill_value=0)
      .reindex(columns=all_years, fill_value=0))
 
cum_prev  = bk.cumsum(axis=1).shift(1, axis=1).fillna(0)   # breaks strictly before year
brk_prev1 = bk.shift(1, axis=1).fillna(0)                  # breaks in year-1
 
# year of most recent break at or before each year, then lagged one year
_lastyr = pd.DataFrame(np.where(bk.values > 0, all_years[None, :], np.nan),
                       index=bk.index, columns=all_years).ffill(axis=1)
last_prev = _lastyr.shift(1, axis=1)
 
 
def _to_long(dfw, name):
    out = dfw.stack().rename(name).reset_index()
    out.columns = ['PIPE_ID', 'year', name]
    return out
 
 
fn = fn.merge(_to_long(bk,        'BREAKED'),     on=['PIPE_ID', 'year'], how='left')
fn = fn.merge(_to_long(cum_prev,  'BREAKS_prev'), on=['PIPE_ID', 'year'], how='left')
fn = fn.merge(_to_long(brk_prev1, '_bl'),         on=['PIPE_ID', 'year'], how='left')
fn = fn.merge(_to_long(last_prev, '_last_prev'),  on=['PIPE_ID', 'year'], how='left')
 
fn['BREAKED']         = fn['BREAKED'].fillna(0)
fn['BREAKS_prev']     = fn['BREAKS_prev'].fillna(0)
fn['target']          = (fn['BREAKED'] > 0).astype(int)
fn['broke_last_year'] = (fn['_bl'].fillna(0) > 0).astype(int)
fn['has_broken_ever'] = (fn['BREAKS_prev'] > 0).astype(int)
 
# TSLF convention must match training: never-broken pipes use age, so lagged = age-1
fn['TSLF_prev'] = np.where(fn['_last_prev'].notna(),
                           (fn['year'] - 1) - fn['_last_prev'],
                           fn['age'] - 1)
fn['TSLF_prev'] = fn['TSLF_prev'].clip(lower=0)
 
# ---- break-type dummies: zero at prediction time (not knowable in advance) ----
for _c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S']:
    fn[_c] = 0
 
# ---- weather, soil, traffic ----
fn = fn.merge(annual, left_on='year', right_index=True, how='left')
 
fn['SOIL_TYPE'] = (fn['PIPE_ID'].map(soil_map).fillna('Unknown')
                   if 'soil_map' in globals() else 'Unknown')
 
if 'near' in globals():
    fn['TRAFFIC_VOLUME'] = fn['PIPE_ID'].map(
        dict(zip(near['PIPE_ID'], near['Volume'].fillna(0)))).fillna(0)
    fn['DIST_TO_ROAD'] = fn['PIPE_ID'].map(
        dict(zip(near['PIPE_ID'], near['dist_to_road'].fillna(999)))).fillna(999)
else:
    fn['TRAFFIC_VOLUME'] = 0
    fn['DIST_TO_ROAD'] = 999
 
 
# ---- encode with the SAME fitted encoders; unseen categories -> -1 ----
def _safe_enc(le, s):
    known = {c: i for i, c in enumerate(le.classes_)}
    return s.astype(str).map(known).fillna(-1).astype(int)
 
 
fn['MATERIAL_enc'] = _safe_enc(le_mat,  fn['MATERIAL'])
fn['TYPE_enc']     = _safe_enc(le_type, fn['TYPE'])
fn['ZONE_enc']     = _safe_enc(le_zone, fn['ZONE'])
fn['SOIL_enc']     = _safe_enc(le_soil, fn['SOIL_TYPE'])
 
X_fn = fn[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_fn = fn['target'].values
 
print(f'  Rows: {len(fn):,}   Pipes: {fn["PIPE_ID"].nunique():,}')
print(f'  Positives: {y_fn.sum():,}   Base rate: {y_fn.mean()*100:.3f}%')
 
prob_fn = best_model.predict_proba(X_fn)[:, 1]
 
print('\n' + '=' * 62)
print('FULL-NETWORK EVALUATION — annual renewal simulation')
print('(every in-service pipe ranked each year; one snapshot per year)')
print('=' * 62)
pr_fn_pooled = average_precision_score(y_fn, prob_fn)
print(f'Pooled AUC-ROC: {roc_auc_score(y_fn, prob_fn):.4f}')
print(f'Pooled PR-AUC:  {pr_fn_pooled:.4f}'
      f'   (lift vs base rate {y_fn.mean():.4f}: {pr_fn_pooled/y_fn.mean():.1f}x)')

# Per-year AUC — near-future degradation (model frozen at <= 2018). Utilities and
# the Kitchener repository (WDSA/CCWI 2024) note that future-period performance
# decays relative to random splits; showing the trajectory is the honest view.
print('\nPer-year AUC-ROC (model frozen at <= 2018):')
for yr in test_years:
    m = fn['year'].values == yr
    if y_fn[m].sum() < 5:
        continue
    print(f'  {yr}: AUC={roc_auc_score(y_fn[m], prob_fn[m]):.4f}  '
          f'(failures={int(y_fn[m].sum())})')

# Barcelona-style annual-renewal Lorenz, averaged across test years.
fn_summ, fn_years = annual_lorenz(fn, prob_fn, test_years)
print(f'\nAnnual-renewal Lorenz  (mean +/- sd over {fn_years[0]}-{fn_years[-1]}):')
for k, (mu, sd) in fn_summ.items():
    print(f'  Renew {k:>4} of network length  ->  capture '
          f'{mu*100:5.1f}% +/- {sd*100:4.1f}% of that year\'s failures')
print('[Barcelona 2026 reference: 30.2% at 5%, 10.32% at 1%]')
print('NOTE: reported as methodology adopted from Forero-Ortiz et al. (2026),')
print('      not a like-for-like ranking of the two utilities.')

fn[['PIPE_ID', 'year', 'target']].assign(prob=prob_fn).to_csv(
    'results/full_network_scores.csv', index=False)
# ABLATION STUDY
print('\n' + '='*62)
print('ABLATION STUDY — contribution of break-history features')
print('='*62)
 
HISTORY_FEATURES = ['TSLF_prev', 'BREAKS_prev']

features_no_hist  = [f for f in features if f not in HISTORY_FEATURES]
features_hist_only = HISTORY_FEATURES + ['age']   # age as bare baseline
 
ablation_results = []
 
for label, feat_set in [
    (f'Full model ({len(features)} features)',                     features),
    (f'No break history ({len(features_no_hist)} features)',       features_no_hist),
    (f'History + age only ({len(features_hist_only)} features)',   features_hist_only),
]:
    print(f'\n  Training: {label}...')
 
    X_tr_a = df.loc[train_mask, feat_set].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_te_a = df.loc[test_mask,  feat_set].apply(pd.to_numeric, errors='coerce').fillna(0)
 
    spw_a = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw_a, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, eval_metric='aucpr', random_state=42,
        n_jobs=-1, tree_method='hist'
    )
    m.fit(X_tr_a, y_tr)
    p_a = m.predict_proba(X_te_a)[:, 1]
 
    auc_a  = roc_auc_score(y_te, p_a)
    pr_a   = average_precision_score(y_te, p_a)

    # Full-network scoring for this ablation variant, using the SAME annual-renewal
    # metric as the headline so the numbers are directly comparable.
    X_fn_a = fn[feat_set].apply(pd.to_numeric, errors='coerce').fillna(0)
    p_fn_a = m.predict_proba(X_fn_a)[:, 1]
    summ_a, _ = annual_lorenz(fn, p_fn_a, test_years)
    l1_mu, _ = summ_a['1pct']
    l5_mu, _ = summ_a['5pct']

    ablation_results.append({
        'Feature set':          label,
        'AUC-ROC':              round(auc_a, 4),
        'PR-AUC':               round(pr_a, 4),
        'AnnLorenz@1% (fn)':    round(l1_mu * 100, 1),
        'AnnLorenz@5% (fn)':    round(l5_mu * 100, 1),
    })
    print(f'    AUC={auc_a:.4f}  PR-AUC={pr_a:.4f}  '
          f'AnnualLorenz@5%(full-network)={l5_mu*100:.1f}%')

abl_df = pd.DataFrame(ablation_results)
print('\nABLATION SUMMARY (fn = full network, annual-renewal Lorenz):')
print(abl_df.to_string(index=False))
print('Gap between "Full model" and "No break history" = contribution of the')
print('70-year break record. Expected to be large — consistent with the Calgary')
print('finding that breakage probability rises sharply after a first break.')
abl_df.to_csv('results/ablation_study.csv', index=False)
print('\nAblation saved to results/ablation_study.csv')
print('='*62)
# STAGE 11: FULL EVALUATION
print(f'[10/11] Evaluation...')

# --- Lorenz curve: full-network annual snapshot (first test year) ---
# The headline metric is the annual-renewal Lorenz computed above on the full
# network. Here we draw the curve for one representative snapshot year so the
# figure matches the operational metric (rank the whole network once, renew the
# top X% by length). The earlier case-control / inverse-probability-weighted
# curves were removed: the case-control sample is ~9x enriched in breakers and an
# IPW correction on it is not valid, because controls were selected on outcome.
yr0 = fn_years[0]
m0 = (fn['year'].values == yr0)
p0, yt0, w0 = prob_fn[m0], y_fn[m0], fn['LENGTH'].values[m0]
order0 = np.argsort(p0)[::-1]
cn0 = np.cumsum(w0[order0]) / w0.sum()
cf0 = np.cumsum(yt0[order0]) / max(yt0.sum(), 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(cn0 * 100, cf0 * 100, linewidth=2, color='#2196F3',
        label=f'{best_name} ({yr0} network)')
ax.plot([0, 100], [0, 100], 'k--', alpha=0.4, label='Random')
for r in [1, 5, 10]:
    idx = min(np.searchsorted(cn0, r / 100), len(cf0) - 1)
    ax.scatter([r], [cf0[idx] * 100], s=80, zorder=5)
    ax.annotate(f'{cf0[idx]*100:.1f}%', (r, cf0[idx]*100),
                textcoords='offset points', xytext=(10, 0), fontsize=10)
ax.set_xlabel('% of network renewed (by length)')
ax.set_ylabel('% of failures captured')
ax.set_title(f'Annual-Renewal Lorenz — full network, {yr0} '
             f'(model frozen at <= {TRAIN_END_YEAR})')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0, 50]); ax.set_ylim([0, 100])
plt.tight_layout(); plt.savefig('results/lorenz_curve.png', dpi=150); plt.close()

# --- MCC ---
print('\nMCC across thresholds:')
mcc_rows = []
for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    yp = (prob >= t).astype(int)
    if yp.sum() == 0:
        continue
    mcc_rows.append({
        'Threshold': t, 'MCC': round(matthews_corrcoef(y_te, yp), 4),
        'Precision': round(precision_score(y_te, yp, zero_division=0), 4),
        'Recall': round(recall_score(y_te, yp), 4), 'Flagged': int(yp.sum())
    })
mcc_df = pd.DataFrame(mcc_rows)
print(mcc_df.to_string(index=False))
mcc_df.to_csv('results/mcc_thresholds.csv', index=False)

# --- ROC + PR curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, p, color in [('XGBoost', xgb_prob, '#2196F3'), ('CatBoost', cat_prob, '#FF5722')]:
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_te, p):.3f})',
                 color=color, linewidth=2)
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, label=f'{name} (AP={average_precision_score(y_te, p):.3f})',
                 color=color, linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Precision-Recall'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/roc_pr_curves.png', dpi=150); plt.close()

# --- Spatial block CV ---
print('\nSpatial block CV (5-fold, blocks = pressure zones)...')
df_train_cv = df[train_mask].copy()
groups = df_train_cv['ZONE_enc'].values
gkf = GroupKFold(n_splits=5)
sp_aucs = []

for fold, (tr_i, val_i) in enumerate(gkf.split(X_tr.values, y_tr.values, groups)):
    if y_tr.values[val_i].sum() < 5:
        continue
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, eval_metric='aucpr', random_state=42,
        n_jobs=-1, tree_method='hist'
    )
    m.fit(X_tr.values[tr_i], y_tr.values[tr_i])
    pf = m.predict_proba(X_tr.values[val_i])[:, 1]
    sp_aucs.append(roc_auc_score(y_tr.values[val_i], pf))
    print(f'  Fold {fold+1}: AUC={sp_aucs[-1]:.4f}')

if sp_aucs:
    print(f'  Spatial CV: {np.mean(sp_aucs):.4f} ± {np.std(sp_aucs):.4f}')

# --- Bootstrap CI (on the full-network snapshot, not the enriched sample) ---
print(f'\nBootstrap CI (500 iterations, full network {yr0} snapshot)...')
yb_all, pb_all, wb_all = yt0.copy(), p0.copy(), w0.copy()
n_snap = len(yb_all)
rng = np.random.default_rng(42)
boot = {'AUC-ROC': [], 'PR-AUC': [], 'P@100': [], 'Lorenz@5%': []}

for _ in range(500):
    idx = rng.choice(n_snap, n_snap, replace=True)
    yb, pb, wb = yb_all[idx], pb_all[idx], wb_all[idx]
    if yb.sum() < 2:
        continue
    boot['AUC-ROC'].append(roc_auc_score(yb, pb))
    boot['PR-AUC'].append(average_precision_score(yb, pb))
    o = np.argsort(pb)[::-1]
    boot['P@100'].append(yb[o[:100]].sum() / 100)
    cl = np.cumsum(wb[o]) / wb.sum()
    cfl = np.cumsum(yb[o]) / yb.sum()
    boot['Lorenz@5%'].append(cfl[min(np.searchsorted(cl, 0.05), len(cfl) - 1)])

ci = pd.DataFrame([{
    'Metric': m, 'Mean': round(np.mean(v), 4),
    '2.5%': round(np.percentile(v, 2.5), 4),
    '97.5%': round(np.percentile(v, 97.5), 4)
} for m, v in boot.items()])
print(ci.to_string(index=False))
ci.to_csv('results/bootstrap_ci.csv', index=False)

# --- SHAP ---
# CatBoost's TreeExplainer hangs on Windows for large samples. XGBoost is fast
# and — since both models saw identical features and produce close PR-AUC —
# the SHAP interpretation is materially the same.
print('\nSHAP analysis...')
explainer = shap.TreeExplainer(xgb_model)
X_shap = X_te.sample(min(1000, len(X_te)), random_state=42)
shap_values = explainer.shap_values(X_shap)
X_disp = X_shap.copy(); X_disp.columns = feature_names

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_disp, show=False, max_display=len(feature_names))
plt.title(f'SHAP — XGBoost, 1-Year Window (Temporal Split)')
plt.tight_layout(); plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_disp, plot_type='bar', show=False, max_display=len(feature_names))
plt.tight_layout(); plt.savefig('results/shap_bar.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Material-stratified SHAP ---
print('\nMaterial-stratified SHAP...')
ci_idx = df.loc[test_mask & (df['MATERIAL'] == 'CI')].index
pvc_idx = df.loc[test_mask & (df['MATERIAL'] == 'PVC')].index
ci_s = X.loc[ci_idx].sample(min(500, len(ci_idx)), random_state=42)
pvc_s = X.loc[pvc_idx].sample(min(500, len(pvc_idx)), random_state=42)

shap_ci = explainer.shap_values(ci_s)
shap_pvc = explainer.shap_values(pvc_s)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ci_d = ci_s.copy(); ci_d.columns = feature_names
pvc_d = pvc_s.copy(); pvc_d.columns = feature_names
plt.sca(axes[0]); shap.summary_plot(shap_ci, ci_d, plot_type='bar', show=False, max_display=len(feature_names))
axes[0].set_title('Cast Iron (CI)')
plt.sca(axes[1]); shap.summary_plot(shap_pvc, pvc_d, plot_type='bar', show=False, max_display=len(feature_names))
axes[1].set_title('PVC')
plt.suptitle('Material-Stratified Failure Drivers', fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig('results/shap_ci_vs_pvc.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Material-stratified Lorenz (full-network snapshot) ---
print(f'\nMaterial-stratified Lorenz (full network, {yr0} snapshot):')
snap = fn[fn['year'] == yr0].copy()
snap['prob'] = p0
snap['y_true'] = yt0
mat_lz = []
for mat in snap['MATERIAL'].value_counts().head(10).index:
    m = snap['MATERIAL'] == mat
    if m.sum() < 100 or snap.loc[m, 'y_true'].sum() < 5:
        continue
    lz, _, _ = compute_lorenz(snap.loc[m, 'y_true'].values, snap.loc[m, 'prob'].values,
                              weights=snap.loc[m, 'LENGTH'].values)
    mat_lz.append({'Material': mat, 'Pipes': int(m.sum()),
                   'Failures': int(snap.loc[m, 'y_true'].sum()),
                   'Lorenz_1%': round(lz['1pct'], 4), 'Lorenz_5%': round(lz['5pct'], 4),
                   'Lorenz_10%': round(lz['10pct'], 4)})
mat_lz_df = pd.DataFrame(mat_lz).sort_values('Lorenz_5%', ascending=False)
print(mat_lz_df.to_string(index=False))
mat_lz_df.to_csv('results/material_lorenz.csv', index=False)

# --- Top-100 pipes: rank once at horizon start, confirm over the whole window ---
print(f'\n[11/11] Top-100 risky pipes (ranked at {yr0}, confirmed {TEST_START_YEAR}-{TEST_END_YEAR})...')
broke_in_window = set(df_breaks.loc[
    (df_breaks['BREAK_DATE'] >= TEST_START_YEAR) &
    (df_breaks['BREAK_DATE'] <= TEST_END_YEAR), 'PIPE_ID'].unique())
snap100 = fn[fn['year'] == yr0].copy()
snap100['risk_score'] = p0
snap100['confirmed'] = snap100['PIPE_ID'].isin(broke_in_window).astype(int)
top100 = snap100.nlargest(100, 'risk_score')[
    ['PIPE_ID', 'MATERIAL', 'age', 'DIAM', 'LENGTH', 'ZONE',
     'SOIL_TYPE', 'TRAFFIC_VOLUME', 'BREAKS_prev', 'TSLF_prev',
     'has_broken_ever', 'risk_score', 'confirmed']].round(4)
print(f'  Confirmed breaks within horizon: {top100["confirmed"].sum()}/100 '
      f'({top100["confirmed"].mean()*100:.1f}%)')
print(f'  Network base rate over horizon:  '
      f'{len(broke_in_window)/fn["PIPE_ID"].nunique()*100:.1f}%  '
      f'(lift: {top100["confirmed"].mean()/(len(broke_in_window)/fn["PIPE_ID"].nunique()):.1f}x)')
print(f'\n  Top 10:')
print(top100.head(10).to_string(index=False))
top100.to_csv('results/top100_risky_pipes.csv', index=False)

# FINAL SUMMARY
print(f'\n{"="*60}')
print('FINAL RESULTS  (full-network, annual-renewal evaluation)')
print(f'{"="*60}')
print(f'Best model: {best_name}  (selected on PR-AUC)')
print(f'Pooled AUC-ROC (full network): {roc_auc_score(y_fn, prob_fn):.4f}')
print(f'Pooled PR-AUC  (full network): {pr_fn_pooled:.4f}  '
      f'({pr_fn_pooled/y_fn.mean():.1f}x base rate)')
_l1 = fn_summ.get("1pct", (float("nan"),))[0]
_l5 = fn_summ.get("5pct", (float("nan"),))[0]
print(f'Annual-renewal Lorenz@1%: {_l1*100:.1f}%   @5%: {_l5*100:.1f}%  '
      f'(mean over {fn_years[0]}-{fn_years[-1]})')
print(f'[Barcelona 2026 reference: 10.32% at 1%, 30.2% at 5% — methodology, not head-to-head]')
if sp_aucs:
    print(f'Spatial block CV AUC: {np.mean(sp_aucs):.4f} +/- {np.std(sp_aucs):.4f}')
print('\nLimitations recorded in run:')
print('  - segment-level unit: mains are split into 2-point segments')
print('  - traffic ~3% coverage and weather is year-level (both near-flat features)')
print('  - output is a renewal-prioritisation ranking, not a calibrated probability')
print(f'\nResults saved to results/')
