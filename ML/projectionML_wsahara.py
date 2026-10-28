"""
Western Sahara Climate Prediction - XGBoost multi-horizon forecast 2025-2050
Trained on TerraClimate 1981-2024 + elevation (6 variables, no fog), same method as the original
northern-Morocco projectionML.py, run on the Western Sahara grid from download/download_terraclimate.py.

Changes vs the original projectionML.py:
  - region: Western Sahara (south of the original grid, 27.74 N, to ~20.8 N), files wsahara_*.tif
  - no 50 mm/year minimum-precipitation filter: most of Western Sahara gets less than that
  - sampling strata re-cut for a low, hyper-arid region: elevation 150 / 400 m, precipitation 30 / 60 mm
  - summary regions: north/south of 25 N, coastal strip, interior, lowlands, plateau
  - files are matched by exact year and month (no wrong-year fallback); incomplete years stay missing
  - vectorized feature building, threaded GeoTIFF reads, XGBoost on CUDA when an NVIDIA GPU is visible
  - outputs go to predictions_wsahara_ML/ (annual + monthly GeoTIFFs)
"""


import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import linregress
from scipy.ndimage import zoom
warnings.filterwarnings('ignore')


# ==================== CENTRALIZED PARAMETERS ====================

class CONFIG:
    """Centralized configuration - modify these to tune model performance."""

    # ===== DIRECTORIES =====
    # Script lives in <project_root>/ML/, data lives in <project_root>/data/
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    # Set WSAHARA_CLIMATE_DATA to use a data folder somewhere else
    DATA_DIR = Path(os.environ.get("WSAHARA_CLIMATE_DATA", PROJECT_ROOT / "data"))
    OUTPUT_DIR = PROJECT_ROOT / "predictions_wsahara_ML"
    ELEVATION_FILE = DATA_DIR / "wsahara_elev.tif"   # written by download/download_terraclimate.py
    FILE_PREFIX = "wsahara"                            # data/{year}/{var}/wsahara_{var}_{year}_{MM}.tif

    # ===== PREDICTION SETTINGS =====
    LOOKBACK_WINDOW = 15  # years of history used for each recursive prediction step

    # ===== TIME PERIODS =====
    YEARS_TRAIN = list(range(1981, 2025))   # training period 
    YEARS_VAL = list(range(2015, 2018))     # validation period
    YEARS_TEST = list(range(2018, 2025))    # test period
    YEARS_ALL = list(range(1981, 2025))
    YEARS_PREDICT = list(range(2025, 2051)) # 2025-2050

    # ===== VARIABLES =====
    VARIABLES_RAW = ['tmax', 'tmin', 'prec', 'vap', 'ws', 'def']
    VARIABLES = ['tmax', 'tmin', 'prec', 'vap', 'ws', 'def']

    AGGREGATION = {
        'tmax': 'mean', 'tmin': 'mean', 'prec': 'sum', 'vap': 'mean',
        'ws': 'mean', 'soil': 'mean', 'srad': 'mean', 'aet': 'sum', 'def': 'sum',
    }

    # ===== SPATIAL FILTERING =====
    MIN_ANNUAL_PREC = 0      # the northern project used 50 mm; most of Western Sahara is below it
    MAX_ELEVATION = 4000

    # ===== SAMPLING STRATEGY =====
    STRATIFIED_SAMPLING = True
    N_PIXELS_PER_STRATA = 300   # fewer strata in a flat desert, so more pixels per stratum

    # Stratification thresholds
    ELEVATION_THRESHOLDS = [150, 400]         # coastal lowlands / plains / Zemmour-Adrar plateaus
    PRECIPITATION_THRESHOLDS = [30, 60]       # mm/year: hyper-arid / arid / less arid

    # ===== XGBOOST HYPERPARAMETERS (OPTIMIZED FOR ACCURACY) =====
    XGBOOST_PARAMS = {
        'prec': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.8,
            'reg_lambda': 1.5,
            'random_state': 42,
            'n_jobs': -1
        },
        'tmax': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'tmin': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'default': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.05,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    }

    # ===== HARDWARE =====
    # Tuned for Ryzen 9 5950X (16 cores) + 64 GB RAM + RTX 3090 24 GB. Works on smaller machines too.
    IO_THREADS = 16          # parallel GeoTIFF reads
    USE_GPU = "auto"         # "auto": train XGBoost on CUDA when an NVIDIA GPU is visible; True/False to force

    # ===== TRAINING SETTINGS =====
    EARLY_STOPPING_ROUNDS = 80
    VERBOSE_TRAINING = False

    # ===== FEATURE ENGINEERING =====
    N_LAG_FEATURES = 5
    WINDOW_5Y_STATS = True
    WINDOW_10Y_STATS = True
    SEASONAL_FEATURES = True
    TREND_FEATURES = True

    # ===== PREDICTION CONSTRAINTS =====
    TMAX_RANGE = (-10, 50)
    TMIN_RANGE = (-20, 40)
    PREC_MIN = 0
    VAP_MIN = 0
    WS_RANGE = (0, 20)
    DEF_MIN = 0

    # ===== VALIDATION CSV =====
    # Regions used for the structured prediction dump
    # (definitions applied later once lat/lon grids exist)
    VALIDATION_CSV_NAME = "predictions_summary.csv"


# Create output directory
CONFIG.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WESTERN SAHARA CLIMATE PREDICTION - XGBoost (fast: vectorized + GPU)")
print("=" * 80)
print(f"Data directory     : {CONFIG.DATA_DIR}")
print(f"Output directory   : {CONFIG.OUTPUT_DIR}")
print(f"Training period    : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]} ({len(CONFIG.YEARS_TRAIN)} years)")
print(f"Validation period  : {CONFIG.YEARS_VAL[0]}-{CONFIG.YEARS_VAL[-1]} ({len(CONFIG.YEARS_VAL)} years)")
print(f"Test period        : {CONFIG.YEARS_TEST[0]}-{CONFIG.YEARS_TEST[-1]} ({len(CONFIG.YEARS_TEST)} years)")
print(f"Prediction period  : {CONFIG.YEARS_PREDICT[0]}-{CONFIG.YEARS_PREDICT[-1]}")
print(f"Pixels per strata  : {CONFIG.N_PIXELS_PER_STRATA}")
print("=" * 80)


# ==================== LOAD DATA ====================
def load_geotiff(filepath):
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return {'data': data, 'bounds': src.bounds,
                    'transform': src.transform, 'crs': src.crs}
    except Exception:
        return None


def get_variable_path(year, variable, month):
    year_dir = CONFIG.DATA_DIR / str(year) / variable
    if not year_dir.exists():
        return None

    # Exact year + month only. The original fallback pattern "*_{MM}*" silently loaded another
    # year's file when one was missing (repo data: 2002/prec holds 2006 files, 1969 holds 1968, ...).
    pattern = f"{CONFIG.FILE_PREFIX}_{variable}_{year}_{month:02d}.tif"
    path = year_dir / pattern
    return path if path.exists() else None


def load_monthly_data_to_annual(years):
    print(f"\nLoading {years[0]}-{years[-1]}...")

    sample_path = get_variable_path(years[0], 'tmax', 1)
    sample_data = load_geotiff(sample_path)
    shape = sample_data['data'].shape

    n_years = len(years)

    annual_data_raw = np.full((n_years, shape[0], shape[1], len(CONFIG.VARIABLES_RAW)), np.nan)
    monthly_data_full = np.full((n_years, 12, shape[0], shape[1], len(CONFIG.VARIABLES_RAW)), np.nan,
                                dtype=np.float32)

    # Parallel reads: rasterio releases the GIL, so threads load several files at once.
    jobs = [(year_idx, year, month, var_idx, var)
            for year_idx, year in enumerate(years) for month in range(1, 13)
            for var_idx, var in enumerate(CONFIG.VARIABLES_RAW)]

    def read_one(job):
        year_idx, year, month, var_idx, var = job
        path = get_variable_path(year, var, month)
        return job, (load_geotiff(path) if path else None)

    with ThreadPoolExecutor(max_workers=CONFIG.IO_THREADS) as ex:
        for (year_idx, year, month, var_idx, var), var_data in tqdm(
                ex.map(read_one, jobs), total=len(jobs), desc="Loading"):
            if var_data:
                monthly_data_full[year_idx, month - 1, :, :, var_idx] = var_data['data']

    # A year is only valid if all 12 months exist; otherwise nansum would turn a missing year
    # into 0 mm and nanmean would average an incomplete year.
    for var_idx, var in enumerate(CONFIG.VARIABLES_RAW):
        monthly_var = monthly_data_full[:, :, :, :, var_idx]
        incomplete = np.isnan(monthly_var).any(axis=1)
        with np.errstate(invalid='ignore'):
            if CONFIG.AGGREGATION[var] == 'sum':
                annual = np.nansum(monthly_var, axis=1)
            else:
                annual = np.nanmean(monthly_var, axis=1)
        annual_data_raw[:, :, :, var_idx] = np.where(incomplete, np.nan, annual)
    for var in CONFIG.VARIABLES:
        v = CONFIG.VARIABLES_RAW.index(var)
        bad_years = [years[k] for k in range(n_years)
                     if np.isnan(annual_data_raw[k, :, :, v]).all()]
        if bad_years:
            print(f"   {var}: no complete data for {bad_years} (left as missing)")

    annual_data = np.full((n_years, shape[0], shape[1], len(CONFIG.VARIABLES)), np.nan)
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        raw_idx = CONFIG.VARIABLES_RAW.index(var)
        annual_data[:, :, :, var_idx] = annual_data_raw[:, :, :, raw_idx]

    n_dry_months = np.full((n_years, shape[0], shape[1]), np.nan)
    for year_idx in range(n_years):
        monthly_prec = monthly_data_full[year_idx, :, :, :, CONFIG.VARIABLES_RAW.index('prec')]
        n_dry_months[year_idx] = np.sum(monthly_prec < 10, axis=0)

    prec_mean = np.nanmean(annual_data[:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)
    land_mask_basic = ~np.isnan(prec_mean)

    print(f"Shape: {annual_data.shape}, Variables: {CONFIG.VARIABLES}")

    return {
        'data': annual_data, 'n_dry_months': n_dry_months,
        'land_mask_basic': land_mask_basic, 'bounds': sample_data['bounds'],
        'transform': sample_data['transform'], 'crs': sample_data['crs'],
        'shape': shape, 'monthly_data': monthly_data_full
    }


print("\nLoading elevation...")
elevation_data = load_geotiff(CONFIG.ELEVATION_FILE)
elevation_original = elevation_data['data']

all_data = load_monthly_data_to_annual(CONFIG.YEARS_ALL)

# Resize elevation to match climate grid
print("\nResizing elevation...")
zoom_factors = (all_data['shape'][0] / elevation_original.shape[0],
                all_data['shape'][1] / elevation_original.shape[1])
elevation = zoom(elevation_original, zoom_factors, order=1)

# Strict land mask
print("\nCreating strict land mask...")
prec_mean_historical = np.nanmean(all_data['data'][:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)

land_mask = (
    all_data['land_mask_basic'] &
    (prec_mean_historical >= CONFIG.MIN_ANNUAL_PREC) &
    (elevation < CONFIG.MAX_ELEVATION) &
    (elevation >= -10)
)

print(f"   Pixels: {np.sum(land_mask):,}")
all_data['land_mask'] = land_mask


# ==================== STRATIFIED SAMPLING ====================
if CONFIG.STRATIFIED_SAMPLING:
    print("\nStratified sampling by climate zone...")

    land_pixels = np.argwhere(all_data['land_mask'])

    strates = []
    for i, j in land_pixels:
        elev_val = elevation[i, j]
        prec_val = prec_mean_historical[i, j]

        if elev_val < CONFIG.ELEVATION_THRESHOLDS[0]:
            elev_zone = 'lowland'
        elif elev_val < CONFIG.ELEVATION_THRESHOLDS[1]:
            elev_zone = 'plain'
        else:
            elev_zone = 'plateau'

        if prec_val < CONFIG.PRECIPITATION_THRESHOLDS[0]:
            prec_zone = 'hyper_arid'
        elif prec_val < CONFIG.PRECIPITATION_THRESHOLDS[1]:
            prec_zone = 'arid'
        else:
            prec_zone = 'less_arid'

        strates.append(f"{elev_zone}_{prec_zone}")

    strates = np.array(strates)
    unique_strates = np.unique(strates)

    print(f"   Found {len(unique_strates)} climate strata:")
    for strate in unique_strates:
        count = np.sum(strates == strate)
        print(f"      {strate}: {count:,} pixels")

    np.random.seed(42)
    sampled_indices = []
    for strate in unique_strates:
        strate_indices = np.where(strates == strate)[0]
        n_sample = min(CONFIG.N_PIXELS_PER_STRATA, len(strate_indices))
        sampled = np.random.choice(strate_indices, n_sample, replace=False)
        sampled_indices.extend(sampled)

    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (stratified)")
else:
    land_pixels = np.argwhere(all_data['land_mask'])
    np.random.seed(42)
    n_sample = min(3000, len(land_pixels))
    sampled_indices = np.random.choice(len(land_pixels), n_sample, replace=False)
    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (random)")


# ==================== FEATURE ENGINEERING (vectorized) ====================
# One function builds the features for MANY pixels at once. It is used both to build the
# training set and at every step of the 2025-2050 forecast, so training and prediction use
# exactly the same code. Values are identical to the original per-pixel version
# (create_improved_features), which looped over one pixel and one year at a time.

def nan_trend(block):
    """OLS slope along axis 1 of a (P, L) block, ignoring NaN. 0 where fewer than 3 valid values
    (same rule as the original calculate_trend, which used scipy.stats.linregress)."""
    L = block.shape[1]
    x = np.arange(L, dtype=np.float64)[None, :]
    valid = ~np.isnan(block)
    n = valid.sum(axis=1).astype(np.float64)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, block, 0.0)
    sx, sy = xv.sum(1), yv.sum(1)
    sxx, sxy = (xv * xv).sum(1), (xv * yv).sum(1)
    den = n * sxx - sx * sx
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = (n * sxy - sx * sy) / den
    return np.where((n >= 3) & (den > 0), slope, 0.0)


def _nanstat(fn, block):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return fn(block, axis=1)


def create_features_batch(data, monthly, year_idx):
    """
    data    : (P, T, V) annual values for P pixels
    monthly : (P, T, 12, V) monthly values
    year_idx: 'current' index; features only use years < year_idx (as in the original)
    Returns an ordered dict {feature_name: array of length P}.
    """
    feats = {}
    P = data.shape[0]
    for var_idx, var_name in enumerate(CONFIG.VARIABLES):
        history = data[:, :year_idx, var_idx]                          # (P, year_idx)
        enough = (~np.isnan(history)).sum(axis=1) >= 5                 # original: skip var if < 5 values
        cols = {}

        for lag in range(1, CONFIG.N_LAG_FEATURES + 1):
            if year_idx - lag >= 0:
                cols[f'{var_name}_lag{lag}'] = data[:, year_idx - lag, var_idx]

        if CONFIG.WINDOW_5Y_STATS and year_idx >= 5:
            w5 = data[:, year_idx - 5:year_idx, var_idx]
            cols[f'{var_name}_mean_5y'] = _nanstat(np.nanmean, w5)
            cols[f'{var_name}_std_5y'] = _nanstat(np.nanstd, w5)

        if CONFIG.WINDOW_10Y_STATS and year_idx >= 10:
            w10 = data[:, year_idx - 10:year_idx, var_idx]
            cols[f'{var_name}_mean_10y'] = _nanstat(np.nanmean, w10)
            cols[f'{var_name}_std_10y'] = _nanstat(np.nanstd, w10)
            cols[f'{var_name}_min_10y'] = _nanstat(np.nanmin, w10)
            cols[f'{var_name}_max_10y'] = _nanstat(np.nanmax, w10)

        if CONFIG.TREND_FEATURES:
            if year_idx >= 10:
                trend_recent = nan_trend(history[:, -10:])
                cols[f'{var_name}_trend_10y'] = trend_recent
            if year_idx >= 20:
                cols[f'{var_name}_trend_accel'] = nan_trend(history[:, -10:]) - nan_trend(history[:, -20:-10])

        if CONFIG.SEASONAL_FEATURES and var_name in ['prec', 'tmax', 'tmin'] and year_idx > 0:
            m = monthly[:, year_idx - 1, :, var_idx]                      # (P, 12)
            has = ~np.isnan(m).all(axis=1)
            winter = _nanstat(np.nanmean, m[:, [11, 0, 1]])
            summer = _nanstat(np.nanmean, m[:, [5, 6, 7]])
            spring = _nanstat(np.nanmean, m[:, [2, 3, 4]])
            autumn = _nanstat(np.nanmean, m[:, [8, 9, 10]])
            mx, mn = _nanstat(np.nanmax, m), _nanstat(np.nanmin, m)
            nan = np.full(P, np.nan)
            cols[f'{var_name}_winter_mean'] = np.where(has, winter, nan)
            cols[f'{var_name}_summer_mean'] = np.where(has, summer, nan)
            cols[f'{var_name}_spring_mean'] = np.where(has, spring, nan)
            cols[f'{var_name}_autumn_mean'] = np.where(has, autumn, nan)
            with np.errstate(invalid='ignore', divide='ignore'):
                cols[f'{var_name}_ratio_summer_winter'] = np.where(has & (winter > 0), summer / winter, nan)
            cols[f'{var_name}_seasonal_amplitude'] = np.where(has, mx - mn, nan)
            if var_name == 'prec':
                cols['prec_driest_month'] = np.where(has, mn, nan)
                cols['prec_wettest_month'] = np.where(has, mx, nan)

        for k, v in cols.items():                                       # skipped variable -> missing
            feats[k] = np.where(enough, v, np.nan)
    return feats


def feature_frame(feats, year_idx_value, year_value, horizon, elev, lat, lon):
    """Assemble one block of rows with the same columns, in the same order, as the original."""
    P = len(elev)
    cols = {'year_idx': np.full(P, year_idx_value), 'year': np.full(P, year_value),
            'elevation': elev, 'lat': lat, 'lon': lon, 'horizon': np.full(P, horizon)}
    cols.update(feats)
    for var_name in CONFIG.VARIABLES:                                   # topography interactions
        lag1 = feats.get(f'{var_name}_lag1')
        if lag1 is not None:
            cols[f'elev_x_{var_name}'] = elev * lag1
    return cols


def pixel_cube(pixels):
    """Annual (P, T, V) and monthly (P, T, 12, V) arrays for a list of (i, j) pixels."""
    I, J = pixels[:, 0], pixels[:, 1]
    data = np.transpose(all_data['data'][:, I, J, :], (1, 0, 2)).astype(np.float64)
    raw_idx = [CONFIG.VARIABLES_RAW.index(v) for v in CONFIG.VARIABLES]
    monthly = all_data['monthly_data'][:, :, I, J, :][..., raw_idx]          # (T, 12, P, V)
    monthly = np.transpose(monthly, (2, 0, 1, 3)).astype(np.float64)        # (P, T, 12, V)
    return data, monthly


# ==================== BUILD DATASET ====================
print("\nBuilding dataset (vectorized)...")

lat_grid = np.linspace(all_data['bounds'].top, all_data['bounds'].bottom, all_data['shape'][0])
lon_grid = np.linspace(all_data['bounds'].left, all_data['bounds'].right, all_data['shape'][1])

S_data, S_monthly = pixel_cube(land_pixels_sampled)
S_elev = elevation[land_pixels_sampled[:, 0], land_pixels_sampled[:, 1]].astype(np.float64)
S_lat = lat_grid[land_pixels_sampled[:, 0]]
S_lon = lon_grid[land_pixels_sampled[:, 1]]
n_years_all = S_data.shape[1]

blocks = []
for year_idx in tqdm(range(10, n_years_all - 6), desc="Building"):
    feats = create_features_batch(S_data, S_monthly, year_idx)
    for horizon in range(1, 7):
        cols = feature_frame(feats, year_idx, CONFIG.YEARS_ALL[year_idx], horizon, S_elev, S_lat, S_lon)
        target_idx = year_idx + horizon
        for var_idx, var_name in enumerate(CONFIG.VARIABLES):
            cols[f'{var_name}_target'] = S_data[:, target_idx, var_idx]
        blocks.append(pd.DataFrame(cols))

df = pd.concat(blocks, ignore_index=True)
print(f"Dataset: {len(df):,} samples, {df.shape[1]} features")

# Split
train_mask = (df['year'] >= CONFIG.YEARS_TRAIN[0]) & (df['year'] <= CONFIG.YEARS_TRAIN[-1])
val_mask = (df['year'] >= CONFIG.YEARS_VAL[0]) & (df['year'] <= CONFIG.YEARS_VAL[-1])
test_mask = (df['year'] >= CONFIG.YEARS_TEST[0]) & (df['year'] <= CONFIG.YEARS_TEST[-1])

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(f"   Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")


# ==================== TRAIN MODELS ====================
def detect_xgb_device():
    if CONFIG.USE_GPU is False:
        return 'cpu'
    has_cuda_build = bool(xgb.build_info().get('USE_CUDA', False))
    has_gpu = shutil.which('nvidia-smi') is not None
    if CONFIG.USE_GPU is True or (has_cuda_build and has_gpu):
        try:
            xgb.XGBRegressor(n_estimators=1, device='cuda', tree_method='hist').fit(
                np.random.rand(64, 3), np.random.rand(64))
            return 'cuda'
        except Exception as e:
            print(f"   CUDA training unavailable ({e}); using CPU")
    return 'cpu'


XGB_DEVICE = detect_xgb_device()
print(f"\nTraining XGBoost models on {XGB_DEVICE.upper()}...")

target_cols = [f'{var}_target' for var in CONFIG.VARIABLES]
meta_cols = ['year', 'year_idx']
feature_cols = [col for col in df_train.columns if col not in target_cols + meta_cols]

print(f"   Features: {len(feature_cols)}")

# NaN = missing: XGBoost learns a default branch for it (filling with 0 would mean '0 mm', '0 °C')
X_train = df_train[feature_cols].values.astype(np.float32)
X_val = df_val[feature_cols].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)

models = {}
test_metrics = {}  # store metrics for later reporting

for var in CONFIG.VARIABLES:
    print(f"\n   Training: {var}")

    y_train = df_train[f'{var}_target'].values
    y_val = df_val[f'{var}_target'].values
    y_test = df_test[f'{var}_target'].values

    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    test_valid = ~np.isnan(y_test)

    params = CONFIG.XGBOOST_PARAMS.get(var, CONFIG.XGBOOST_PARAMS['default']).copy()

    print(f"   Using: n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, lr={params['learning_rate']}")

    params['early_stopping_rounds'] = CONFIG.EARLY_STOPPING_ROUNDS
    params['tree_method'] = 'hist'
    params['device'] = XGB_DEVICE

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train[train_valid], y_train[train_valid],
        eval_set=[(X_val[val_valid], y_val[val_valid])],
        verbose=CONFIG.VERBOSE_TRAINING
    )
    model.set_params(device='cpu')   # predictions below run on CPU arrays; avoids device-mismatch copies
    models[var] = model

    # Model evaluation on test set
    y_pred = model.predict(X_test[test_valid])
    y_true_test = y_test[test_valid]

    mae = mean_absolute_error(y_true_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred))
    r2 = r2_score(y_true_test, y_pred)

    # Baselines on the same test subset
    # Persistence: predict target = lag1 (last observed year)
    lag1_col = f'{var}_lag1'
    if lag1_col in df_test.columns:
        y_pers = df_test.loc[test_valid, lag1_col].fillna(0).values
        mae_pers = mean_absolute_error(y_true_test, y_pers)
        r2_pers = r2_score(y_true_test, y_pers)
    else:
        mae_pers, r2_pers = np.nan, np.nan

    # Climatology: predict target = 10-year rolling mean (proxy for climatology)
    mean10_col = f'{var}_mean_10y'
    if mean10_col in df_test.columns:
        y_clim = df_test.loc[test_valid, mean10_col].fillna(0).values
        mae_clim = mean_absolute_error(y_true_test, y_clim)
        r2_clim = r2_score(y_true_test, y_clim)
    else:
        mae_clim, r2_clim = np.nan, np.nan

    skill_pers = 1 - mae / mae_pers if mae_pers and not np.isnan(mae_pers) else np.nan
    skill_clim = 1 - mae / mae_clim if mae_clim and not np.isnan(mae_clim) else np.nan

    test_metrics[var] = {
        'mae_xgb': mae, 'rmse_xgb': rmse, 'r2_xgb': r2,
        'mae_persistence': mae_pers, 'r2_persistence': r2_pers,
        'mae_climatology': mae_clim, 'r2_climatology': r2_clim,
        'skill_vs_persistence': skill_pers, 'skill_vs_climatology': skill_clim,
    }

    print(f"   XGBoost    -> MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")
    print(f"   Persistence-> MAE: {mae_pers:.3f}  R2: {r2_pers:.3f}  (skill vs pers: {skill_pers:+.3f})")
    print(f"   Climatology-> MAE: {mae_clim:.3f}  R2: {r2_clim:.3f}  (skill vs clim: {skill_clim:+.3f})")
    print(f"   Best iteration: {model.best_iteration}")

# Per-horizon breakdown
print("\nWalk-forward validation (multi-horizon):")
for horizon in [1, 3, 6]:
    test_horizon = df_test[df_test['horizon'] == horizon]
    if len(test_horizon) > 0:
        print(f"\n   Horizon {horizon} year(s):")
        X_test_h = test_horizon[feature_cols].values.astype(np.float32)

        for var in CONFIG.VARIABLES:
            y_test_h = test_horizon[f'{var}_target'].values
            valid = ~np.isnan(y_test_h)
            if np.sum(valid) > 0:
                y_pred_h = models[var].predict(X_test_h[valid])
                mae_h = mean_absolute_error(y_test_h[valid], y_pred_h)
                r2_h = r2_score(y_test_h[valid], y_pred_h)
                print(f"      {var}: MAE={mae_h:.2f}, R2={r2_h:.3f}")

with open(CONFIG.OUTPUT_DIR / 'models_improved.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'feature_cols': feature_cols,
        'test_metrics': test_metrics,
    }, f)

print("\nModels trained.")


# ==================== RECURSIVE PREDICTION (all pixels at once) ====================
# Same algorithm as the original predict_multiyear_recursive, but each forecast year is computed
# for every land pixel in one matrix: features for all pixels -> one predict call per variable.
print("\nRecursive prediction 2025-2050 (all pixels per year)...")

CLIP = {
    'tmax': lambda p: np.clip(p, *CONFIG.TMAX_RANGE),
    'tmin': lambda p: np.clip(p, *CONFIG.TMIN_RANGE),
    'prec': lambda p: np.maximum(p, CONFIG.PREC_MIN),
    'vap': lambda p: np.maximum(p, CONFIG.VAP_MIN),
    'ws': lambda p: np.clip(p, *CONFIG.WS_RANGE),
    'def': lambda p: np.maximum(p, CONFIG.DEF_MIN),
}


def predict_recursive_batch(models, feature_cols, data, monthly, elev, lat, lon, n_years, lookback_window):
    """data (P, T, V), monthly (P, T, 12, V) -> {var: (n_years, P)}"""
    P, T, V = data.shape
    out = {var: np.full((n_years, P), np.nan, dtype=np.float32) for var in CONFIG.VARIABLES}
    # Pre-allocated history (observed + forecast years), filled in place: memory stays flat.
    ext = np.empty((P, T + n_years, V)); ext[:, :T] = data
    ext_m = np.empty((P, T + n_years, 12, V)); ext_m[:, :T] = monthly
    reference = monthly[:, -1, :, :].copy()                            # (P, 12, V) last observed year
    ref_annual = {}
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ref_annual[var] = (np.nansum(reference[:, :, var_idx], axis=1) if var == 'prec'
                               else np.nanmean(reference[:, :, var_idx], axis=1))

    for year_offset in tqdm(range(n_years), desc="Predicting"):
        end = T + year_offset                                          # years available so far
        win = ext[:, end - lookback_window:end, :]
        win_m = ext_m[:, end - lookback_window:end, :, :]
        year_idx = win.shape[1] - 1
        horizon = min(year_offset + 1, 6)
        feats = create_features_batch(win, win_m, year_idx)
        cols = feature_frame(feats, year_idx, 0, horizon, elev, lat, lon)
        X = pd.DataFrame(cols).reindex(columns=feature_cols).values.astype(np.float32)   # unseen columns -> NaN

        new_row = np.empty((P, V))
        for var_idx, var in enumerate(CONFIG.VARIABLES):
            pred = CLIP[var](models[var].predict(X).astype(np.float64))
            out[var][year_offset] = pred
            new_row[:, var_idx] = pred

        # monthly values of the predicted year: last observed year's pattern scaled to the new
        # annual value for prec / tmax / tmin, flat for the other variables (as in the original)
        new_m = np.repeat(new_row[:, None, :], 12, axis=1)
        for var in ['prec', 'tmax', 'tmin']:
            v = CONFIG.VARIABLES.index(var)
            ra = ref_annual[var]
            ok = ra > 0.01
            with np.errstate(invalid='ignore', divide='ignore'):
                scaled = reference[:, :, v] * (new_row[:, v] / ra)[:, None]
            new_m[:, :, v] = np.where(ok[:, None], scaled, (new_row[:, v] / 12)[:, None])

        ext[:, end] = new_row
        ext_m[:, end] = new_m
    return out


n_future_years = len(CONFIG.YEARS_PREDICT)
shape = all_data['shape']
land_pixels_all = np.argwhere(all_data['land_mask'])
del df, df_train, df_val, df_test, X_train, X_val, X_test, blocks       # free training memory

A_data, A_monthly = pixel_cube(land_pixels_all)
A_elev = elevation[land_pixels_all[:, 0], land_pixels_all[:, 1]].astype(np.float64)
A_lat = lat_grid[land_pixels_all[:, 0]]
A_lon = lon_grid[land_pixels_all[:, 1]]

preds_batch = predict_recursive_batch(models, feature_cols, A_data, A_monthly, A_elev, A_lat, A_lon,
                                      n_future_years, CONFIG.LOOKBACK_WINDOW)

predictions_all = {
    var: np.full((n_future_years, shape[0], shape[1]), np.nan, dtype=np.float32)
    for var in CONFIG.VARIABLES
}
for var in CONFIG.VARIABLES:
    predictions_all[var][:, land_pixels_all[:, 0], land_pixels_all[:, 1]] = preds_batch[var]
del A_data, A_monthly

print("Predictions complete.")


# ==================== MONTHLY DISAGGREGATION ====================
print("\nMonthly disaggregation...")

# 1991-2020 climatological baseline (indices relative to 1958 start of TerraClimate,
# but here we use 1981 indexing consistent with YEARS_ALL)
clim_start_idx = max(0, 1991 - CONFIG.YEARS_ALL[0])
clim_end_idx = min(len(CONFIG.YEARS_ALL), 2020 - CONFIG.YEARS_ALL[0] + 1)

# prec: monthly share of the annual total (1/12 where the 1991-2020 climatology is 0 mm)
MONTHLY_OUTPUT_VARS = ['tmax', 'tmin', 'prec']
MULTIPLICATIVE_VARS = {'prec'}

monthly_patterns = {}
for var in MONTHLY_OUTPUT_VARS:
    var_idx = CONFIG.VARIABLES_RAW.index(var)

    monthly_clim = np.nanmean(
        all_data['monthly_data'][clim_start_idx:clim_end_idx, :, :, :, var_idx],
        axis=0
    )

    annual_clim = np.nansum(monthly_clim, axis=0) if CONFIG.AGGREGATION[var] == 'sum' else np.nanmean(monthly_clim, axis=0)

    pattern = np.zeros_like(monthly_clim)
    for month in range(12):
        with np.errstate(divide='ignore', invalid='ignore'):
            if var in MULTIPLICATIVE_VARS:
                pattern[month] = np.where(annual_clim > 0, monthly_clim[month] / annual_clim, 1.0 / 12)
            else:
                pattern[month] = monthly_clim[month] - annual_clim

    monthly_patterns[var] = pattern

predictions_monthly = {}
for var in MONTHLY_OUTPUT_VARS:
    predictions_monthly[var] = np.full(
        (n_future_years, 12, shape[0], shape[1]), np.nan, dtype=np.float32
    )

    for future_idx in range(n_future_years):
        annual_pred = predictions_all[var][future_idx]

        for month in range(12):
            if var in MULTIPLICATIVE_VARS:
                predictions_monthly[var][future_idx, month] = (
                    annual_pred * monthly_patterns[var][month]
                )
            else:
                predictions_monthly[var][future_idx, month] = (
                    annual_pred + monthly_patterns[var][month]
                )

print("Monthly disaggregation complete.")


# ==================== EXPORT GEOTIFFS ====================
print("\nExporting GeoTIFFs...")


def save_geotiff(data, year, variable, metadata):
    year_dir = CONFIG.OUTPUT_DIR / str(year) / variable
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"{CONFIG.FILE_PREFIX}_{variable}_{year}.tif"

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=metadata['shape'][0], width=metadata['shape'][1], count=1,
        dtype=data.dtype, crs=metadata['crs'], transform=metadata['transform'],
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)


def save_monthly_geotiff(data, year, month, variable, metadata):
    year_dir = CONFIG.OUTPUT_DIR / str(year) / variable
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"{CONFIG.FILE_PREFIX}_{variable}_{year}_{month:02d}.tif"

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=metadata['shape'][0], width=metadata['shape'][1], count=1,
        dtype=data.dtype, crs=metadata['crs'], transform=metadata['transform'],
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)


for future_idx, year in enumerate(tqdm(CONFIG.YEARS_PREDICT, desc="Exporting")):
    for var in CONFIG.VARIABLES:
        save_geotiff(predictions_all[var][future_idx], year, var, all_data)

    tavg = (predictions_all['tmax'][future_idx] + predictions_all['tmin'][future_idx]) / 2
    save_geotiff(tavg, year, 'tavg', all_data)

    for month in range(1, 13):
        for var in MONTHLY_OUTPUT_VARS:
            save_monthly_geotiff(
                predictions_monthly[var][future_idx, month - 1],
                year, month, var, all_data
            )


# ==================== PREDICTIONS SUMMARY CSV ====================
# Structured dump of predicted values aggregated by region, with persistence
# and climatology baselines embedded in the same file. The 'observed' column
# stays blank and can be filled in later (e.g. from CHIRPS or TerraClimate)
# as ground truth for each future year becomes available.
print("\nBuilding predictions summary CSV...")

# Build 2D lat/lon grids for regional masking
lat_grid_2d = np.broadcast_to(lat_grid[:, None], shape)
lon_grid_2d = np.broadcast_to(lon_grid[None, :], shape)

# Region masks
# Coastal strip: cells within 0.5 deg (~50 km) of the westernmost land cell of their row.
# The coast runs roughly north-south and the sea is on the west, so this is a good proxy.
coast_lon = np.full(shape[0], np.nan)
for _r in range(shape[0]):
    _cols = np.where(land_mask[_r])[0]
    if len(_cols):
        coast_lon[_r] = lon_grid[_cols[0]]
coastal = land_mask & ((lon_grid_2d - coast_lon[:, None]) <= 0.5)

regions = {
    'western_sahara': land_mask,
    'north_of_25N':   land_mask & (lat_grid_2d >= 25),
    'south_of_25N':   land_mask & (lat_grid_2d < 25),
    'coastal_strip':  coastal,
    'interior':       land_mask & ~coastal,
    'lowlands':       land_mask & (elevation < 150),
    'plateau':        land_mask & (elevation >= 400),
}

# Report region sizes
print("   Region pixel counts:")
for r, m in regions.items():
    print(f"      {r}: {int(np.sum(m)):,}")

# Precompute climatology (1981-2014 training-only mean) and 2024 observed fields
# per variable, to avoid recomputing inside the loop.
clim_years_end = 2015 - CONFIG.YEARS_ALL[0]  # exclusive index = first year NOT in climatology
climatology_fields = {}
last_observed_fields = {}
for var in CONFIG.VARIABLES:
    var_idx = CONFIG.VARIABLES.index(var)
    climatology_fields[var] = np.nanmean(
        all_data['data'][:clim_years_end, :, :, var_idx], axis=0
    )
    last_observed_fields[var] = all_data['data'][-1, :, :, var_idx]

summary_rows = []
for future_idx, year in enumerate(CONFIG.YEARS_PREDICT):
    for var in CONFIG.VARIABLES:
        xgb_field = predictions_all[var][future_idx]
        clim_field = climatology_fields[var]
        pers_field = last_observed_fields[var]  # flat persistence: repeat 2024

        for region_name, mask in regions.items():
            if not np.any(mask):
                continue
            summary_rows.append({
                'year': year,
                'variable': var,
                'region': region_name,
                'n_pixels': int(np.sum(mask)),
                'xgb_pred': float(np.nanmean(xgb_field[mask])),
                'persistence_2024': float(np.nanmean(pers_field[mask])),
                'climatology_1981_2014': float(np.nanmean(clim_field[mask])),
                'observed': np.nan,  # fill in later as ground truth becomes available
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = CONFIG.OUTPUT_DIR / CONFIG.VALIDATION_CSV_NAME
summary_df.to_csv(summary_path, index=False)
print(f"   Summary CSV written: {summary_path}")
print(f"   Rows: {len(summary_df):,}")

# Also dump the test-set metrics table for easy reference
metrics_rows = []
for var, m in test_metrics.items():
    metrics_rows.append({'variable': var, **m})
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = CONFIG.OUTPUT_DIR / "test_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Test metrics CSV written: {metrics_path}")


# ==================== DONE ====================
print("\n" + "=" * 80)
print("OPTIMIZED PREDICTIONS COMPLETE")
print("=" * 80)
print("\nConfiguration used:")
print(f"   Pixels per strata   : {CONFIG.N_PIXELS_PER_STRATA}")
print(f"   Lag features        : {CONFIG.N_LAG_FEATURES}")
print(f"   XGBoost trees (prec): {CONFIG.XGBOOST_PARAMS['prec']['n_estimators']}")
print(f"   Early stopping      : {CONFIG.EARLY_STOPPING_ROUNDS} rounds")
print(f"   Lookback window     : {CONFIG.LOOKBACK_WINDOW} years")
print(f"\nOutput directory: {CONFIG.OUTPUT_DIR}")
print("=" * 80)