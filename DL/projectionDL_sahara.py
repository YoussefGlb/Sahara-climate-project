"""
Western Sahara Climate Prediction - LSTM multi-horizon forecast 2025-2050
=========================================================================

Same method as the original northern-Morocco projectionDL.py (2-layer LSTM + static branch +
multi-output head, 6 variables, no fog), run on the Western Sahara grid from
download/download_terraclimate.py.

Changes vs the original projectionDL.py:
    - region: Western Sahara (south of the original grid, 27.74 N, to ~20.8 N), files wsahara_*.tif
    - no 50 mm/year minimum-precipitation filter: most of Western Sahara gets less than that
    - sampling strata re-cut for a low, hyper-arid region: elevation 150 / 400 m, precipitation 30 / 60 mm
    - summary regions: north/south of 25 N, coastal strip, interior, lowlands, plateau
    - files are matched by exact year and month; isolated missing years are filled for the INPUT
      sequences only, targets stay observed only
    - vectorized sample building, GPU-resident dataset, mixed precision; outputs to predictions_wsahara_DL/

Deep learning counterpart to projectionML_wsahara.py (XGBoost): same data, same splits, same
output format (GeoTIFFs + summary CSV with baselines), so the two can be compared one-to-one.
"""


import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import linregress
from scipy.ndimage import zoom
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch - the deep learning framework
# If this import fails: py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ==================== CENTRALIZED PARAMETERS ====================

class CONFIG:
    """Centralized configuration - modify these to tune model behavior."""

    # ===== DIRECTORIES =====
    # Script lives in <project_root>/DL/, data lives in <project_root>/data/
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    # Set WSAHARA_CLIMATE_DATA to use a data folder somewhere else
    DATA_DIR = Path(os.environ.get("WSAHARA_CLIMATE_DATA", PROJECT_ROOT / "data"))
    OUTPUT_DIR = PROJECT_ROOT / "predictions_wsahara_DL"
    ELEVATION_FILE = DATA_DIR / "wsahara_elev.tif"   # written by download/download_terraclimate.py
    FILE_PREFIX = "wsahara"                            # data/{year}/{var}/wsahara_{var}_{year}_{MM}.tif

    # ===== TIME PERIODS =====
    # DL gets more data than XGBoost because it benefits more from it.
    YEARS_ALL = list(range(1958, 2025))       # full data range 1958-2024
    YEARS_TRAIN = list(range(1958, 2015))     # training period (1958-2014)
    YEARS_VAL = list(range(2015, 2018))       # validation period
    YEARS_TEST = list(range(2018, 2025))      # test period
    YEARS_PREDICT = list(range(2025, 2051))   # prediction period

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

    ELEVATION_THRESHOLDS = [150, 400]         # coastal lowlands / plains / Zemmour-Adrar plateaus
    PRECIPITATION_THRESHOLDS = [30, 60]       # mm/year: hyper-arid / arid / less arid

    # ===== LSTM ARCHITECTURE =====
    SEQUENCE_LENGTH = 15              # past years fed into the LSTM at each step
    LSTM_HIDDEN_SIZE = 128            # dimensionality of the hidden state
    LSTM_NUM_LAYERS = 2               # stacked LSTM layers
    LSTM_DROPOUT = 0.2                # dropout between LSTM layers
    HEAD_HIDDEN_SIZE = 64             # width of the dense layers after LSTM
    HEAD_DROPOUT = 0.2                # dropout in the dense head

    # ===== TRAINING =====
    BATCH_SIZE = 512                  # samples per gradient step
    LEARNING_RATE = 1e-3              # Adam starting LR (classic default)
    WEIGHT_DECAY = 1e-5               # tiny L2 regularization
    MAX_EPOCHS = 100                  # hard cap - early stopping usually kicks in earlier
    EARLY_STOPPING_PATIENCE = 15      # epochs with no val improvement before stopping
    LR_SCHEDULER_PATIENCE = 5         # epochs with no val improvement before halving LR
    GRAD_CLIP = 1.0                   # clip gradient norm (LSTMs can explode without this)

    # ===== HARDWARE =====
    # Tuned for Ryzen 9 5950X (16 cores) + 64 GB RAM + RTX 3090 24 GB. Works on smaller machines too.
    IO_THREADS = 16           # parallel GeoTIFF reads
    USE_AMP = True            # mixed precision for training on CUDA (ignored on CPU)

    # ===== PREDICTION =====
    LOOKBACK_WINDOW = 15              # must equal SEQUENCE_LENGTH for consistency
    MAX_HORIZON = 6                   # model trained on 1-6 year horizons

    # ===== PREDICTION CONSTRAINTS (same as ML) =====
    TMAX_RANGE = (-10, 50)
    TMIN_RANGE = (-20, 40)
    PREC_MIN = 0
    VAP_MIN = 0
    WS_RANGE = (0, 20)
    DEF_MIN = 0

    # ===== MISC =====
    RANDOM_SEED = 42
    VALIDATION_CSV_NAME = "predictions_summary.csv"


# Select device (GPU if available, CPU fallback)
# NOTE: on GTX 1060 3GB, the LSTM+head I've sized will use ~1GB VRAM. Safe.
# If CUDA init fails you'll see a clear error; the fallback to CPU is automatic.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True                 # fastest LSTM kernels for fixed shapes
    torch.backends.cuda.matmul.allow_tf32 = True          # Ampere (RTX 30xx) tensor cores for fp32 matmuls
    torch.backends.cudnn.allow_tf32 = True

# Create output directory
CONFIG.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Set seeds for reproducibility
np.random.seed(CONFIG.RANDOM_SEED)
torch.manual_seed(CONFIG.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG.RANDOM_SEED)

print("=" * 80)
print("WESTERN SAHARA CLIMATE PREDICTION - DEEP LEARNING (LSTM, fast: vectorized + GPU-resident)")
print("=" * 80)
print(f"Device             : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"VRAM available     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Data directory     : {CONFIG.DATA_DIR}")
print(f"Output directory   : {CONFIG.OUTPUT_DIR}")
print(f"Training period    : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]} ({len(CONFIG.YEARS_TRAIN)} years)")
print(f"Validation period  : {CONFIG.YEARS_VAL[0]}-{CONFIG.YEARS_VAL[-1]} ({len(CONFIG.YEARS_VAL)} years)")
print(f"Test period        : {CONFIG.YEARS_TEST[0]}-{CONFIG.YEARS_TEST[-1]} ({len(CONFIG.YEARS_TEST)} years)")
print(f"Full data range    : {CONFIG.YEARS_ALL[0]}-{CONFIG.YEARS_ALL[-1]} ({len(CONFIG.YEARS_ALL)} years)")
print(f"Sequence length    : {CONFIG.SEQUENCE_LENGTH} years")
print(f"LSTM hidden size   : {CONFIG.LSTM_HIDDEN_SIZE}")
print("=" * 80)


# ==================== DATA LOADING (mirror of ML script) ====================
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

    prec_mean = np.nanmean(annual_data[:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)
    land_mask_basic = ~np.isnan(prec_mean)

    print(f"Shape: {annual_data.shape}, Variables: {CONFIG.VARIABLES}")

    return {
        'data': annual_data,
        'land_mask_basic': land_mask_basic,
        'bounds': sample_data['bounds'],
        'transform': sample_data['transform'],
        'crs': sample_data['crs'],
        'shape': shape,
        'monthly_data': monthly_data_full
    }


print("\nLoading elevation...")
elevation_data = load_geotiff(CONFIG.ELEVATION_FILE)
elevation_original = elevation_data['data']

all_data = load_monthly_data_to_annual(CONFIG.YEARS_ALL)

print("\nResizing elevation...")
zoom_factors = (all_data['shape'][0] / elevation_original.shape[0],
                all_data['shape'][1] / elevation_original.shape[1])
elevation = zoom(elevation_original, zoom_factors, order=1)

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


# ==================== GAP FILLING FOR INPUTS ====================
# Missing years (see "no complete data" above) are spread so that almost every 15-year window
# contains one, which would leave no complete training sequence. Inputs therefore use a copy
# where each missing year is linearly interpolated from its neighbours (per pixel, per variable).
# TARGETS always come from the original data, so the model is never trained to predict a filled year.
def fill_time_gaps(data):
    filled = data.copy()
    n_t = data.shape[0]
    flat = filled.reshape(n_t, -1)
    t = np.arange(n_t)
    for col in np.where(np.isnan(flat).any(axis=0) & ~np.isnan(flat).all(axis=0))[0]:
        series = flat[:, col]
        ok = ~np.isnan(series)
        flat[~ok, col] = np.interp(t[~ok], t[ok], series[ok])
    return flat.reshape(data.shape)


print("\nFilling isolated missing years for model inputs (targets stay observed)...")
all_data['data_filled'] = fill_time_gaps(all_data['data'])
n_filled = int(np.sum(np.isnan(all_data['data'][:, land_mask]) & ~np.isnan(all_data['data_filled'][:, land_mask])))
print(f"   Filled values on land: {n_filled:,}")


# ==================== STRATIFIED SAMPLING (mirror of ML script) ====================
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
    n_sample = min(3000, len(land_pixels))
    sampled_indices = np.random.choice(len(land_pixels), n_sample, replace=False)
    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (random)")


# ==================== FEATURE COMPUTATION FOR STATIC BRANCH (vectorized) ====================
# The static vector per sample is [elevation, lat, lon, horizon] + for each variable
# [10-year trend, trend acceleration, 10-year mean]. It is computed for all pixels at once;
# values are identical to the original per-sample compute_static_features (linregress-based).
def nan_trend(block):
    """OLS slope along axis 1 of a (P, L) block ignoring NaN; 0 where < 3 valid values."""
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


def static_features_batch(data, year_idx, elev, lat, lon, horizon):
    """data (P, T, V) -> (P, 4 + 3V) float32, history = years < year_idx."""
    P, _, V = data.shape
    out = np.zeros((P, 4 + 3 * V), dtype=np.float64)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = elev, lat, lon, float(horizon)
    for v in range(V):
        hist = data[:, :year_idx, v]
        if year_idx >= 10:
            trend_recent = nan_trend(hist[:, -10:])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mean_10y = np.nanmean(hist[:, -10:], axis=1)
        else:
            trend_recent = np.zeros(P)
            mean_10y = np.zeros(P)
        accel = trend_recent - nan_trend(hist[:, -20:-10]) if year_idx >= 20 else np.zeros(P)
        out[:, 4 + 3 * v] = trend_recent
        out[:, 5 + 3 * v] = accel
        out[:, 6 + 3 * v] = mean_10y
    return out.astype(np.float32)


N_STATIC_FEATURES = 4 + 3 * len(CONFIG.VARIABLES)
print(f"\nStatic feature vector length: {N_STATIC_FEATURES}")


def pixel_arrays(pixels, key):
    """(P, T, V) float64 array of all_data[key] for a list of (i, j) pixels."""
    return np.transpose(all_data[key][:, pixels[:, 0], pixels[:, 1], :], (1, 0, 2)).astype(np.float64)


# ==================== BUILD SEQUENCE DATASET (vectorized) ====================
# Each sample: sequence (SEQUENCE_LENGTH, n_vars) of gap-filled inputs, static (N_STATIC,),
# target (n_vars,) of OBSERVED values. Built one year_idx at a time for all sampled pixels.
print("\nBuilding sequence dataset (vectorized)...")

lat_grid = np.linspace(all_data['bounds'].top, all_data['bounds'].bottom, all_data['shape'][0])
lon_grid = np.linspace(all_data['bounds'].left, all_data['bounds'].right, all_data['shape'][1])

seq_len = CONFIG.SEQUENCE_LENGTH
n_years = len(CONFIG.YEARS_ALL)
n_vars = len(CONFIG.VARIABLES)

S_fill = pixel_arrays(land_pixels_sampled, 'data_filled')
S_obs = pixel_arrays(land_pixels_sampled, 'data')
S_elev = elevation[land_pixels_sampled[:, 0], land_pixels_sampled[:, 1]].astype(np.float64)
S_lat = lat_grid[land_pixels_sampled[:, 0]]
S_lon = lon_grid[land_pixels_sampled[:, 1]]
keep_pixel = np.isnan(S_obs[:, :, 0]).sum(axis=1) <= n_years * 0.1       # same 10% rule as before

seq_parts, stat_parts, tgt_parts, year_parts, hor_parts = [], [], [], [], []
for year_idx in tqdm(range(seq_len, n_years - CONFIG.MAX_HORIZON), desc="Building"):
    seq = S_fill[:, year_idx - seq_len:year_idx, :]
    ok_seq = keep_pixel & ~np.isnan(seq).any(axis=(1, 2))
    static = static_features_batch(S_fill, year_idx, S_elev, S_lat, S_lon, 0)
    for horizon in range(1, CONFIG.MAX_HORIZON + 1):
        target = S_obs[:, year_idx + horizon, :]
        ok = ok_seq & ~np.isnan(target).any(axis=1)
        if not ok.any():
            continue
        st = static[ok].copy()
        st[:, 3] = horizon
        seq_parts.append(seq[ok].astype(np.float32))
        stat_parts.append(st)
        tgt_parts.append(target[ok].astype(np.float32))
        year_parts.append(np.full(ok.sum(), CONFIG.YEARS_ALL[year_idx]))
        hor_parts.append(np.full(ok.sum(), horizon))

sequences_arr = np.concatenate(seq_parts)
statics_arr = np.concatenate(stat_parts)
targets_arr = np.concatenate(tgt_parts)
years_arr = np.concatenate(year_parts)
horizons_arr = np.concatenate(hor_parts)

print(f"Dataset: {len(sequences_arr):,} samples")
print(f"   Sequence shape: {sequences_arr.shape}")
print(f"   Static shape:   {statics_arr.shape}")
print(f"   Target shape:   {targets_arr.shape}")


# ==================== TRAIN/VAL/TEST SPLIT ====================
# Split by year to avoid temporal leakage.
train_mask = np.isin(years_arr, CONFIG.YEARS_TRAIN)
val_mask = np.isin(years_arr, CONFIG.YEARS_VAL)
test_mask = np.isin(years_arr, CONFIG.YEARS_TEST)

print(f"\n   Train: {np.sum(train_mask):,}, Val: {np.sum(val_mask):,}, Test: {np.sum(test_mask):,}")


# ==================== NORMALIZATION ====================
# LSTMs train vastly better when inputs are roughly standardized (mean=0, std=1).
# We fit scalers ONLY on training data to avoid leaking test statistics.
# Scalers are saved with the model and used at inference time to transform
# new inputs and un-transform predictions back to physical units.
print("\nComputing normalization statistics (training data only)...")

# Sequence: per-variable stats (6 vars)
seq_mean = sequences_arr[train_mask].reshape(-1, n_vars).mean(axis=0)  # (6,)
seq_std = sequences_arr[train_mask].reshape(-1, n_vars).std(axis=0) + 1e-6  # (6,)

# Static: per-feature stats (22 features)
stat_mean = statics_arr[train_mask].mean(axis=0)  # (22,)
stat_std = statics_arr[train_mask].std(axis=0) + 1e-6  # (22,)

# Target: per-variable stats (6 vars)
tgt_mean = targets_arr[train_mask].mean(axis=0)  # (6,)
tgt_std = targets_arr[train_mask].std(axis=0) + 1e-6  # (6,)


def normalize_seq(x):
    return (x - seq_mean) / seq_std


def normalize_stat(x):
    return (x - stat_mean) / stat_std


def normalize_tgt(x):
    return (x - tgt_mean) / tgt_std


def denormalize_tgt(x):
    return x * tgt_std + tgt_mean


print(f"   Target means  : {tgt_mean}")
print(f"   Target stds   : {tgt_std}")


# ==================== GPU-RESIDENT DATA ====================
# The whole dataset is a few hundred MB, so it lives on the GPU for the entire run.
# Batches are index slices: no DataLoader, no host-to-device copies per batch.
def to_device(seqs, stats, tgts):
    return (torch.tensor(normalize_seq(seqs), dtype=torch.float32, device=DEVICE),
            torch.tensor(normalize_stat(stats), dtype=torch.float32, device=DEVICE),
            torch.tensor(normalize_tgt(tgts), dtype=torch.float32, device=DEVICE))


train_split = to_device(sequences_arr[train_mask], statics_arr[train_mask], targets_arr[train_mask])
val_split = to_device(sequences_arr[val_mask], statics_arr[val_mask], targets_arr[val_mask])
test_split = to_device(sequences_arr[test_mask], statics_arr[test_mask], targets_arr[test_mask])
if DEVICE.type == 'cuda':
    print(f"   Dataset on GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ==================== LSTM MODEL ====================
# Architecture overview:
#
#   sequence (batch, 15, 6)  -->  LSTM (2 layers, 128 hidden)  -->  last_hidden (batch, 128)
#                                                                        |
#                                                                        v
#                                                            concat with static (batch, 22)
#                                                                        |
#                                                                        v
#                                                       Dense(128+22 -> 64) + ReLU + Dropout
#                                                                        |
#                                                                        v
#                                                       Dense(64 -> 64) + ReLU + Dropout
#                                                                        |
#                                                                        v
#                                                       Dense(64 -> 6)  <-- predicts 6 vars
#
# WHY THIS SHAPE:
# - LSTM extracts temporal features from the raw 15-year sequence
# - Static branch injects hard-to-learn long-term signals (trend, climatology)
# - Shared dense head produces all 6 outputs jointly (enforces soft consistency)
# - ~200k parameters total, comfortably under GPU limits
class LSTMClimateModel(nn.Module):
    def __init__(self, n_vars, n_static, hidden_size=128, num_layers=2,
                 head_hidden=64, lstm_dropout=0.2, head_dropout=0.2):
        super().__init__()

        # The LSTM layer. batch_first=True means input shape is (batch, seq, features)
        # which is more intuitive than the PyTorch default of (seq, batch, features).
        self.lstm = nn.LSTM(
            input_size=n_vars,        # 6 (tmax, tmin, prec, vap, ws, def)
            hidden_size=hidden_size,  # 128
            num_layers=num_layers,    # 2
            dropout=lstm_dropout,     # dropout between layers
            batch_first=True,
        )

        # After LSTM we have (batch, 128). We concat with static (batch, 22).
        # The dense head maps (128 + 22) -> 6 outputs.
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_vars),
        )

    def forward(self, seq, static):
        # seq shape: (batch, 15, 6). static shape: (batch, 22).
        lstm_out, (h_n, c_n) = self.lstm(seq)
        # lstm_out shape: (batch, 15, 128) - output at every timestep
        # h_n shape:      (num_layers, batch, 128) - last hidden state per layer

        # We use the last hidden state of the top layer as the sequence summary.
        # Equivalent to lstm_out[:, -1, :].
        seq_summary = h_n[-1]  # (batch, 128)

        # Fuse sequence summary with static features
        fused = torch.cat([seq_summary, static], dim=1)  # (batch, 128 + 22)

        return self.head(fused)  # (batch, 6)


model = LSTMClimateModel(
    n_vars=n_vars,
    n_static=N_STATIC_FEATURES,
    hidden_size=CONFIG.LSTM_HIDDEN_SIZE,
    num_layers=CONFIG.LSTM_NUM_LAYERS,
    head_hidden=CONFIG.HEAD_HIDDEN_SIZE,
    lstm_dropout=CONFIG.LSTM_DROPOUT,
    head_dropout=CONFIG.HEAD_DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")


# ==================== TRAINING LOOP ====================
# The canonical PyTorch training loop:
#   1. Set model to train mode
#   2. For each batch:
#      a. Move tensors to GPU
#      b. Forward pass (model predicts)
#      c. Compute loss (MSE)
#      d. Zero gradients, backprop, clip gradients, optimizer step
#   3. Set model to eval mode, compute validation loss
#   4. Check early stopping and LR scheduler
criterion = nn.MSELoss()  # plain MSE on normalized targets
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG.LEARNING_RATE,
    weight_decay=CONFIG.WEIGHT_DECAY,
)
# Reduce LR by half when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=CONFIG.LR_SCHEDULER_PATIENCE,
)


USE_AMP = bool(CONFIG.USE_AMP) and DEVICE.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)


def run_epoch(split, training=True):
    """One epoch over a GPU-resident split. Mixed precision (fp16 autocast) on CUDA for training speed;
    loss and gradient clipping stay in float32."""
    seqs, stats, tgts = split
    n = seqs.shape[0]
    model.train(training)
    order = torch.randperm(n, device=DEVICE) if training else torch.arange(n, device=DEVICE)
    total = torch.zeros((), device=DEVICE)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, n, CONFIG.BATCH_SIZE):
            ix = order[start:start + CONFIG.BATCH_SIZE]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                pred = model(seqs[ix], stats[ix])
            loss = criterion(pred.float(), tgts[ix])
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)   # LSTMs need this
                scaler.step(optimizer)
                scaler.update()
            total += loss.detach() * ix.shape[0]
    return float(total) / n                       # one GPU->CPU sync per epoch


print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float('inf')
best_epoch = 0
epochs_without_improvement = 0
training_log = []

for epoch in range(1, CONFIG.MAX_EPOCHS + 1):
    train_loss = run_epoch(train_split, training=True)
    val_loss = run_epoch(val_split, training=False)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    training_log.append({
        'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': current_lr
    })

    improved = val_loss < best_val_loss
    marker = " *" if improved else ""
    print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.1e}{marker}")

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0
        # Save best weights
        torch.save(model.state_dict(), CONFIG.OUTPUT_DIR / 'best_model.pt')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CONFIG.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best was epoch {best_epoch}.")
            break

# Reload best weights for evaluation and prediction
print(f"\nReloading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CONFIG.OUTPUT_DIR / 'best_model.pt', map_location=DEVICE))
model.eval()


# ==================== TEST SET EVALUATION + BASELINES ====================
# Same metric framework as the ML script: XGBoost MAE/R2 + persistence +
# climatology for each variable. Here instead of XGBoost we evaluate the LSTM.
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

# Gather all test predictions in one pass (in normalized space)
model.eval()
with torch.no_grad():
    t_seq, t_stat, t_tgt = test_split
    test_preds_norm = np.concatenate([model(t_seq[s:s + 65536], t_stat[s:s + 65536]).float().cpu().numpy()
                                      for s in range(0, t_seq.shape[0], 65536)], axis=0)
    test_targets_norm = t_tgt.cpu().numpy()

# De-normalize to physical units for reporting
test_preds = denormalize_tgt(test_preds_norm)
test_targets = denormalize_tgt(test_targets_norm)

# Baselines computed from the test subset of sequences/statics directly
# Persistence: target = last value in sequence = sequences[:, -1, :]
# Climatology: target = mean_10y feature = statics[:, 4 + 2 + var*3 : ...]
# Actually mean_10y is stored as the 3rd feature per variable in the static vector:
#   static = [elev, lat, lon, horizon, trend_0, accel_0, mean10y_0, trend_1, accel_1, mean10y_1, ...]
# Index of mean_10y for variable v: 4 + v*3 + 2 = 6 + 3*v
test_sequences = sequences_arr[test_mask]
test_statics = statics_arr[test_mask]

persistence_preds = test_sequences[:, -1, :]  # (N_test, 6)
climatology_preds = np.stack([test_statics[:, 6 + 3 * v] for v in range(n_vars)], axis=1)  # (N_test, 6)

# Per-variable metrics
test_metrics = {}
for var_idx, var in enumerate(CONFIG.VARIABLES):
    y_true = test_targets[:, var_idx]
    y_lstm = test_preds[:, var_idx]
    y_pers = persistence_preds[:, var_idx]
    y_clim = climatology_preds[:, var_idx]

    mae_lstm = mean_absolute_error(y_true, y_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_true, y_lstm))
    r2_lstm = r2_score(y_true, y_lstm)

    mae_pers = mean_absolute_error(y_true, y_pers)
    r2_pers = r2_score(y_true, y_pers)

    mae_clim = mean_absolute_error(y_true, y_clim)
    r2_clim = r2_score(y_true, y_clim)

    skill_pers = 1 - mae_lstm / mae_pers if mae_pers > 0 else np.nan
    skill_clim = 1 - mae_lstm / mae_clim if mae_clim > 0 else np.nan

    test_metrics[var] = {
        'mae_lstm': mae_lstm, 'rmse_lstm': rmse_lstm, 'r2_lstm': r2_lstm,
        'mae_persistence': mae_pers, 'r2_persistence': r2_pers,
        'mae_climatology': mae_clim, 'r2_climatology': r2_clim,
        'skill_vs_persistence': skill_pers, 'skill_vs_climatology': skill_clim,
    }

    print(f"\n{var}:")
    print(f"   LSTM        -> MAE: {mae_lstm:.3f}  RMSE: {rmse_lstm:.3f}  R2: {r2_lstm:.3f}")
    print(f"   Persistence -> MAE: {mae_pers:.3f}  R2: {r2_pers:.3f}  (skill: {skill_pers:+.3f})")
    print(f"   Climatology -> MAE: {mae_clim:.3f}  R2: {r2_clim:.3f}  (skill: {skill_clim:+.3f})")

# Per-horizon breakdown
print("\n" + "-" * 80)
print("Per-horizon test MAE (LSTM only)")
print("-" * 80)
test_horizons = horizons_arr[test_mask]
for horizon in [1, 3, 6]:
    mask = test_horizons == horizon
    if not np.any(mask):
        continue
    print(f"\n   Horizon {horizon} year(s):")
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        mae_h = mean_absolute_error(test_targets[mask, var_idx], test_preds[mask, var_idx])
        r2_h = r2_score(test_targets[mask, var_idx], test_preds[mask, var_idx])
        print(f"      {var}: MAE={mae_h:.2f}, R2={r2_h:.3f}")


# ==================== RECURSIVE PREDICTION 2025-2050 (all pixels per year) ====================
print("\n" + "=" * 80)
print("RECURSIVE PREDICTION 2025-2050 (all pixels per year on " + DEVICE.type.upper() + ")")
print("=" * 80)

CLIP = {
    'tmax': lambda p: np.clip(p, *CONFIG.TMAX_RANGE),
    'tmin': lambda p: np.clip(p, *CONFIG.TMIN_RANGE),
    'prec': lambda p: np.maximum(p, CONFIG.PREC_MIN),
    'vap': lambda p: np.maximum(p, CONFIG.VAP_MIN),
    'ws': lambda p: np.clip(p, *CONFIG.WS_RANGE),
    'def': lambda p: np.maximum(p, CONFIG.DEF_MIN),
}


def predict_recursive_lstm_batch(model, data, elev, lat, lon, n_years_ahead, seq_len, chunk=65536):
    """Same algorithm as the original per-pixel predictor, run for every pixel at once.
    data (P, T, V) gap-filled history -> {var: (n_years_ahead, P)}. Inference stays in float32."""
    P, T, V = data.shape
    out = {var: np.full((n_years_ahead, P), np.nan, dtype=np.float32) for var in CONFIG.VARIABLES}
    ext = np.empty((P, T + n_years_ahead, V)); ext[:, :T] = data      # pre-allocated, filled in place
    model.eval()
    for year_offset in tqdm(range(n_years_ahead), desc="Predicting"):
        end = T + year_offset
        seq = ext[:, end - seq_len:end, :]
        ok = ~np.isnan(seq).any(axis=(1, 2))
        static = static_features_batch(ext[:, :end], end, elev, lat, lon, 1)
        pred_phys = np.full((P, V), np.nan)
        idx = np.where(ok)[0]
        with torch.no_grad():
            for s in range(0, len(idx), chunk):
                b = idx[s:s + chunk]
                st = torch.tensor(normalize_seq(seq[b]), dtype=torch.float32, device=DEVICE)
                sa = torch.tensor(normalize_stat(static[b]), dtype=torch.float32, device=DEVICE)
                pred_phys[b] = denormalize_tgt(model(st, sa).float().cpu().numpy())
        new_row = np.full((P, V), np.nan)
        for var_idx, var in enumerate(CONFIG.VARIABLES):
            p = CLIP[var](pred_phys[:, var_idx])
            out[var][year_offset] = p
            new_row[:, var_idx] = p
        # the original appended a float32 row; keep the same rounding so results match exactly
        ext[:, end] = new_row.astype(np.float32).astype(np.float64)
    return out


n_future_years = len(CONFIG.YEARS_PREDICT)
shape = all_data['shape']
land_pixels_all = np.argwhere(all_data['land_mask'])

A_fill = pixel_arrays(land_pixels_all, 'data_filled')
A_keep = np.isnan(A_fill[:, :, 0]).sum(axis=1) <= n_years * 0.1
A_elev = elevation[land_pixels_all[:, 0], land_pixels_all[:, 1]].astype(np.float64)
A_lat = lat_grid[land_pixels_all[:, 0]]
A_lon = lon_grid[land_pixels_all[:, 1]]

preds_batch = predict_recursive_lstm_batch(model, A_fill[A_keep], A_elev[A_keep], A_lat[A_keep], A_lon[A_keep],
                                           n_future_years, seq_len)

predictions_all = {
    var: np.full((n_future_years, shape[0], shape[1]), np.nan, dtype=np.float32)
    for var in CONFIG.VARIABLES
}
kept = land_pixels_all[A_keep]
for var in CONFIG.VARIABLES:
    predictions_all[var][:, kept[:, 0], kept[:, 1]] = preds_batch[var]
del A_fill

print("Predictions complete.")


# ==================== MONTHLY DISAGGREGATION (same as ML) ====================
print("\nMonthly disaggregation...")

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


# ==================== EXPORT GEOTIFFS (same as ML) ====================
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


# ==================== PREDICTIONS SUMMARY CSV (same structure as ML) ====================
print("\nBuilding predictions summary CSV...")

lat_grid_2d = np.broadcast_to(lat_grid[:, None], shape)
lon_grid_2d = np.broadcast_to(lon_grid[None, :], shape)

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

print("   Region pixel counts:")
for r, m in regions.items():
    print(f"      {r}: {int(np.sum(m)):,}")

# Climatology from training years only (1958-2014)
clim_years_end = len(CONFIG.YEARS_TRAIN)  # exclusive
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
        lstm_field = predictions_all[var][future_idx]
        clim_field = climatology_fields[var]
        pers_field = last_observed_fields[var]

        for region_name, mask in regions.items():
            if not np.any(mask):
                continue
            summary_rows.append({
                'year': year,
                'variable': var,
                'region': region_name,
                'n_pixels': int(np.sum(mask)),
                'lstm_pred': float(np.nanmean(lstm_field[mask])),
                'persistence_2024': float(np.nanmean(pers_field[mask])),
                'climatology_1958_2014': float(np.nanmean(clim_field[mask])),
                'observed': np.nan,
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = CONFIG.OUTPUT_DIR / CONFIG.VALIDATION_CSV_NAME
summary_df.to_csv(summary_path, index=False)
print(f"   Summary CSV written: {summary_path}")
print(f"   Rows: {len(summary_df):,}")

# Test metrics CSV
metrics_rows = []
for var, m in test_metrics.items():
    metrics_rows.append({'variable': var, **m})
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = CONFIG.OUTPUT_DIR / "test_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Test metrics CSV written: {metrics_path}")

# Training log CSV (loss curves)
training_log_df = pd.DataFrame(training_log)
training_log_df.to_csv(CONFIG.OUTPUT_DIR / "training_log.csv", index=False)

# Save scalers + model together for later reuse
with open(CONFIG.OUTPUT_DIR / 'model_bundle.pkl', 'wb') as f:
    pickle.dump({
        'seq_mean': seq_mean, 'seq_std': seq_std,
        'stat_mean': stat_mean, 'stat_std': stat_std,
        'tgt_mean': tgt_mean, 'tgt_std': tgt_std,
        'config': {k: v for k, v in CONFIG.__dict__.items() if not k.startswith('_')},
        'n_static_features': N_STATIC_FEATURES,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
    }, f)

print("\n" + "=" * 80)
print("DEEP LEARNING PIPELINE COMPLETE")
print("=" * 80)
print(f"\nConfiguration used:")
print(f"   Sequence length  : {CONFIG.SEQUENCE_LENGTH}")
print(f"   LSTM layers      : {CONFIG.LSTM_NUM_LAYERS}")
print(f"   LSTM hidden size : {CONFIG.LSTM_HIDDEN_SIZE}")
print(f"   Total parameters : {n_params:,}")
print(f"   Best val epoch   : {best_epoch}")
print(f"   Training range   : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]}")
print(f"\nOutput directory: {CONFIG.OUTPUT_DIR}")
print("=" * 80)
