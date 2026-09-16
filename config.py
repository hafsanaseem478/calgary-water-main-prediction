"""
Shared configuration for every phase of the Calgary water main pipeline.

Every phase script imports from here, so paths, study-design constants,
model settings and the feature list are defined in exactly one place.
"""
import os

# Input data — update paths to match your data/ folder
PATH_MAIN    = "data/Public_Water_Main_20240621.csv"
PATH_BREAKS  = "data/Water_Main_Breaks_20260726.csv"
PATH_ZONES   = "data/Water_Pressure_Zones_20260728.csv"
PATH_WEATHER = "data/weatherstats_calgary_daily.csv"
PATH_SOIL    = "data/soil_data/surf_py_ll.shp"
PATH_TRAFFIC = "data/Traffic_Volumes_for_2023_20260729.csv"

# Study design
PREDICTION_WINDOW = 1
TEST_START_YEAR = 2019
TEST_END_YEAR = 2025
TRAIN_END_YEAR = TEST_START_YEAR - 1          # 2018
# Model selection uses a validation period inside the training era:
# fit on <= 2016, compare models on the full network in 2017-2018, then
# retrain the chosen model through 2018. 2019-2025 is touched only once.
VALIDATION_START_YEAR = 2017
NEGATIVE_SAMPLING_RATIO = 3                   # controls sampled per case
RANDOM_STATE = 42
BREAK_MATCH_MAX_DIST_M = 20                   # break -> nearest pipe, metres
METRIC_CRS = 3762                             # NAD83(CSRS) / UTM 11N
N_BOOTSTRAP = 500
N_SPATIAL_FOLDS = 5


PANEL_MODE = os.environ.get("PANEL_MODE", "fixed").strip().lower()
if PANEL_MODE not in ("fixed", "legacy"):
    raise ValueError(f"PANEL_MODE must be 'fixed' or 'legacy', got {PANEL_MODE!r}")
LEGACY = PANEL_MODE == "legacy"

_suffix = "_legacy" if LEGACY else ""
RESULTS_DIR = f"results{_suffix}"     # figures and tables
INTERIM_DIR = f"interim{_suffix}"     # hand-off files between phases
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)

# Features
FEATURES = [
    # pipe intrinsics
    "MATERIAL_enc", "DIAM", "LENGTH", "age",
    # location: engineering context + raw coordinates
    "ZONE_enc", "SOIL_enc", "X", "Y",
    # break history (lagged one year)
    "TSLF_prev", "BREAKS_prev",
]
FEATURE_NAMES = [
    "Material", "Diameter", "Length", "Age",
    "Pressure Zone", "Soil Type", "X (Easting)", "Y (Northing)",
    "TSLF (prev yr)", "Cum. Breaks (prev yr)",
]
HISTORY_FEATURES = ["TSLF_prev", "BREAKS_prev"]
# Removed after the feature review (see README): INSTALLED_YEAR, TYPE_enc,
# has_broken_ever, broke_last_year, TRAFFIC_VOLUME, DIST_TO_ROAD, break-type
# dummies A..S, and all year-level weather variables.

LORENZ_THRESHOLDS = (0.01, 0.05, 0.10)

# Models
XGB_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric="aucpr", random_state=RANDOM_STATE,
    n_jobs=-1, tree_method="hist",
)
CAT_PARAMS = dict(
    iterations=400, depth=6, learning_rate=0.05,
    auto_class_weights="Balanced", loss_function="Logloss",
    eval_metric="PRAUC", random_seed=RANDOM_STATE, verbose=100,
)
