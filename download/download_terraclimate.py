"""
Download TerraClimate monthly data for WESTERN SAHARA and write GeoTIFFs in the same layout and
units as the original northern-Morocco project:

    data/{year}/{var}/wsahara_{var}_{year}_{MM}.tif      (float32, nodata -9999, EPSG:4326)
    data/wsahara_elev.tif                                (WorldClim 2.1 30" elevation, averaged to the same grid)

Region: everything south of the original northern grid (its bottom edge, 27.7436 N) down to Lagouira
(~20.8 N) and west of 8.67 W, cut to the Western Sahara borders (Algeria and Mauritania excluded).
The grid is TerraClimate's native 1/24 degree grid (~4 km), so no resampling is applied to the
climate values. The two regions tile together: northern grid above 27.7436 N, this grid below it.

Source: TerraClimate (Abatzoglou et al. 2018) from the Climatology Lab THREDDS server.
  1st choice : NetCDF Subset Service, one small file per variable and year (plain HTTP, only the box)
  fallback   : OPeNDAP on the 1958-present aggregation (needs netCDF4 with DAP support)

Units (TerraClimate native, same as the original project's files):
  tmax, tmin degC | prec mm/month | vap kPa | ws m/s | def mm/month

Usage (from the project root):
    python download/download_terraclimate.py                         # 1958-2024, 6 variables
    python download/download_terraclimate.py --years 2015 2024       # test range first
    python download/download_terraclimate.py --elev-file D:/wc2.1_30s_elev.tif   # reuse a global DEM you have
It resumes: years whose 12 files already exist are skipped. Failures go to data/download_failures.txt.
"""
import argparse
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds


class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PREFIX = "wsahara"
    YEARS = (1958, 2024)
    VARIABLES = ["tmax", "tmin", "prec", "vap", "ws", "def"]
    TC_NAME = {"prec": "ppt"}            # project name -> TerraClimate name (the others are identical)
    NODATA = -9999.0

    # ---- grid: TerraClimate native cells whose centres fall inside this box ----
    RES = 1.0 / 24.0
    NORTH = 27.7436    # bottom edge of the original northern grid -> no overlap, no gap
    SOUTH = 20.70
    WEST = -17.25
    EAST = -8.50

    # Western Sahara land borders (lon, lat). The ocean side is a far-west line: TerraClimate is
    # already empty over the sea. Borders follow the straight-line treaty boundaries; expect
    # +/- one 4 km cell along them.
    POLYGON = [(-20.0, 27.80), (-8.6667, 27.80), (-8.6667, 26.0), (-12.0, 26.0), (-12.0, 23.4467),
               (-13.0, 21.3333), (-17.05, 21.3333), (-17.05, 20.70), (-20.0, 20.70), (-20.0, 27.80)]

    NCSS = ("http://thredds.northwestknowledge.net:8080/thredds/ncss/TERRACLIMATE_ALL/data/"
            "TerraClimate_{tc}_{year}.nc")
    OPENDAP = ("http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
               "agg_terraclimate_{tc}_1958_CurrentYear_GLOBE.nc")
    WORLDCLIM_ELEV = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_elev.zip"

    TIMEOUT = 300      # seconds per HTTP request
    RETRIES = 3


NC_LOCK = threading.Lock()   # the netCDF C library is not thread-safe: downloads run in parallel, parsing does not


# ==================== GRID + REGION ====================
def target_grid():
    r = CONFIG.RES
    i0 = int(np.ceil((CONFIG.WEST + 180) / r - 0.5)); i1 = int(np.floor((CONFIG.EAST + 180) / r - 0.5))
    j0 = int(np.ceil((90 - CONFIG.NORTH) / r - 0.5)); j1 = int(np.floor((90 - CONFIG.SOUTH) / r - 0.5))
    lons = -180 + (np.arange(i0, i1 + 1) + 0.5) * r
    lats = 90 - (np.arange(j0, j1 + 1) + 0.5) * r          # north -> south
    transform = from_origin(lons[0] - r / 2, lats[0] + r / 2, r, r)
    return lats, lons, transform


def region_mask(shape, transform):
    geom = {"type": "Polygon", "coordinates": [CONFIG.POLYGON]}
    return geometry_mask([geom], out_shape=shape, transform=transform, invert=True)   # True inside


def out_path(data_dir, var, year, month):
    return Path(data_dir) / str(year) / var / f"{CONFIG.PREFIX}_{var}_{year}_{month:02d}.tif"


def write_tif(path, arr, transform, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.isfinite(arr), arr, CONFIG.NODATA).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=CONFIG.NODATA,
                       compress="deflate") as dst:
        dst.write(data, 1)
        if tags:
            dst.update_tags(**tags)


# ==================== DOWNLOAD ====================
def http_get(url, target):
    last = None
    for attempt in range(1, CONFIG.RETRIES + 1):
        try:
            tmp = Path(str(target) + ".part")
            with urllib.request.urlopen(url, timeout=CONFIG.TIMEOUT) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(target)
            return target
        except Exception as e:
            last = e
            time.sleep(10 * attempt)
    raise RuntimeError(f"download failed after {CONFIG.RETRIES} tries: {last}")


def select_months(da, year, lats, lons):
    """(time, lat, lon) DataArray -> {month: 2-D float32 array on the target grid}."""
    import xarray as xr  # noqa: F401
    idx = np.where(da["time"].dt.year.values == year)[0]
    da = da.isel(time=idx).sel(lat=lats, lon=lons, method="nearest", tolerance=CONFIG.RES / 4)
    vals = da.values.astype(np.float32)
    return {int(m): vals[k] for k, m in enumerate(da["time"].dt.month.values)}


def fetch_ncss(tc, year, lats, lons, tmp_dir):
    pad = CONFIG.RES
    q = {"var": tc, "north": f"{lats[0] + pad:.4f}", "south": f"{lats[-1] - pad:.4f}",
         "west": f"{lons[0] - pad:.4f}", "east": f"{lons[-1] + pad:.4f}",
         "disableProjSubset": "on", "horizStride": 1,
         "time_start": f"{year}-01-01T00:00:00Z", "time_end": f"{year}-12-31T23:59:59Z",
         "timeStride": 1, "accept": "netcdf"}
    url = CONFIG.NCSS.format(tc=tc, year=year) + "?" + urllib.parse.urlencode(q)
    path = Path(tmp_dir) / f"{tc}_{year}.nc"
    http_get(url, path)
    import xarray as xr
    try:
        with NC_LOCK:
            with xr.open_dataset(path) as ds:
                return select_months(ds[tc], year, lats, lons)
    finally:
        path.unlink(missing_ok=True)


def fetch_opendap(tc, year, lats, lons, tmp_dir=None):
    import xarray as xr
    with NC_LOCK:
        with xr.open_dataset(CONFIG.OPENDAP.format(tc=tc)) as ds:
            pad = CONFIG.RES
            da = ds[tc].sel(lat=slice(lats[0] + pad, lats[-1] - pad), lon=slice(lons[0] - pad, lons[-1] + pad))
            return select_months(da, year, lats, lons)


def process(var, year, grid, region, args):
    lats, lons, transform = grid
    if not args.overwrite and all(out_path(args.data_dir, var, year, m).exists() for m in range(1, 13)):
        return "skip"
    tc = CONFIG.TC_NAME.get(var, var)
    sources = {"ncss": [fetch_ncss], "opendap": [fetch_opendap], "auto": [fetch_ncss, fetch_opendap]}[args.source]
    errors = []
    for fetch in sources:
        try:
            months = fetch(tc, year, lats, lons, args.tmp_dir)
            break
        except Exception as e:
            errors.append(f"{fetch.__name__}: {e}")
    else:
        raise RuntimeError(" | ".join(errors))
    if sorted(months) != list(range(1, 13)):
        raise RuntimeError(f"server returned months {sorted(months)} instead of 1-12")
    means = []
    for m in range(1, 13):
        a = np.where(region, months[m], np.nan)
        write_tif(out_path(args.data_dir, var, year, m), a, transform,
                  tags={"variable": var, "source": f"TerraClimate {tc}", "year": year, "month": m})
        means.append(np.nanmean(a))
    return f"ok mean={np.nanmean(means):.2f}"


# ==================== ELEVATION ====================
def build_elevation(args, grid, region):
    lats, lons, transform = grid
    target = Path(args.data_dir) / f"{CONFIG.PREFIX}_elev.tif"
    if target.exists() and not args.overwrite:
        print(f"Elevation     : {target.name} exists, kept")
        return
    if args.elev_file:
        src_path = str(args.elev_file)
    else:
        cache = Path(args.data_dir) / "_cache"
        cache.mkdir(parents=True, exist_ok=True)
        zpath = cache / "wc2.1_30s_elev.zip"
        if not zpath.exists():
            print("Elevation     : downloading WorldClim 2.1 30\" elevation (a few hundred MB, once)...", flush=True)
            http_get(CONFIG.WORLDCLIM_ELEV, zpath)
        with zipfile.ZipFile(zpath) as z:
            tif = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
        src_path = f"/vsizip/{zpath.as_posix()}/{tif}"
    r = CONFIG.RES
    w, e = lons[0] - r / 2 - 0.1, lons[-1] + r / 2 + 0.1
    s, n = lats[-1] - r / 2 - 0.1, lats[0] + r / 2 + 0.1
    with rasterio.open(src_path) as src:
        win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
        win_transform = src.window_transform(win)
        src_crs = src.crs or "EPSG:4326"
    dst = np.full(region.shape, np.nan, dtype=np.float32)
    reproject(source=data, destination=dst, src_transform=win_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    dst[~region] = np.nan
    write_tif(target, dst, transform, tags={"variable": "elevation", "units": "m",
                                            "source": "WorldClim 2.1 30s, averaged to the TerraClimate grid"})
    print(f"Elevation     : {target.name} written (min {np.nanmin(dst):.0f} m, max {np.nanmax(dst):.0f} m)")


# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser(description="Download TerraClimate for Western Sahara as monthly GeoTIFFs")
    ap.add_argument("--data-dir", type=Path, default=CONFIG.DATA_DIR)
    ap.add_argument("--years", type=int, nargs=2, default=CONFIG.YEARS, metavar=("FIRST", "LAST"))
    ap.add_argument("--vars", nargs="+", default=CONFIG.VARIABLES,
                    help="project names: tmax tmin prec vap ws def (also soil srad aet pet ...)")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (4-6 is polite)")
    ap.add_argument("--source", choices=["auto", "ncss", "opendap"], default="auto")
    ap.add_argument("--elev-file", type=Path, default=None,
                    help="existing GLOBAL WorldClim elevation GeoTIFF to use instead of downloading it")
    ap.add_argument("--skip-elevation", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.tmp_dir = Path(args.data_dir) / "_tmp"
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    grid = target_grid()
    lats, lons, transform = grid
    region = region_mask((len(lats), len(lons)), transform)
    jobs = [(v, y) for y in range(args.years[0], args.years[1] + 1) for v in args.vars]
    print(f"Grid          : {len(lons)} x {len(lats)} cells (1/24 deg), "
          f"lon {lons[0] - CONFIG.RES / 2:.4f}..{lons[-1] + CONFIG.RES / 2:.4f}, "
          f"lat {lats[-1] - CONFIG.RES / 2:.4f}..{lats[0] + CONFIG.RES / 2:.4f}")
    print(f"Region        : {int(region.sum()):,} cells inside the Western Sahara borders (land and sea)")
    print(f"Jobs          : {len(jobs)} ({len(args.vars)} variables x {args.years[1] - args.years[0] + 1} years), "
          f"workers={args.workers}, source={args.source}")
    print(f"Output        : {args.data_dir}/{{year}}/{{var}}/{CONFIG.PREFIX}_{{var}}_{{year}}_{{MM}}.tif\n")

    if not args.skip_elevation:
        try:
            build_elevation(args, grid, region)
        except Exception as e:
            print(f"Elevation     : FAILED ({e}). Re-run, or pass --elev-file with a global WorldClim DEM.")

    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process, v, y, grid, region, args): (v, y) for v, y in jobs}
        for f in as_completed(futs):
            v, y = futs[f]
            done += 1
            try:
                msg = f.result()
            except Exception as e:
                msg = f"FAILED: {e}"
                failures.append(f"{y} {v}  {e}")
                traceback.print_exc(limit=1)
            print(f"[{done:4d}/{len(jobs)}] {y} {v:<5} {msg}", flush=True)

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    log = Path(args.data_dir) / "download_failures.txt"
    if failures:
        log.write_text("\n".join(sorted(failures)))
        print(f"\n{len(failures)} downloads failed; listed in {log}. Re-run the same command to retry only those.")
    else:
        log.unlink(missing_ok=True)
        print("\nAll downloads done.")


if __name__ == "__main__":
    main()
