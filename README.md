# Climate Prediction —  Sahara

A companion to `climate-prediction-morocco-2050`. It covers the area the northern project left out: ** Sahara, from 27.74° N (the bottom edge of the northern grid) down to Lagouira (~20.8° N)**. It uses the same method: TerraClimate monthly data from 1958–2024 with 6 variables and no fog, an XGBoost forecast, an LSTM forecast, and 2025–2050 outputs in the same GeoTIFF layout.

 Sahara is a different climate regime (hyper-arid, Atlantic trade winds, rare autumn storms). That's why it gets its own model, and isn't mixed into the northern training data.

## What's inside

```
download/download_terraclimate.py   TerraClimate -> data/{year}/{var}/wsahara_{var}_{year}_{MM}.tif + data/wsahara_elev.tif
ML/projectionML_wsahara.py          XGBoost pipeline (6 variables)       -> predictions_wsahara_ML/
DL/projectionDL_wsahara.py          LSTM pipeline (6 variables)          -> predictions_wsahara_DL/
requirements.txt
```

## Region and grid

- **Grid:** TerraClimate's native 1/24° grid (~4 km), 210 × 169 cells. Lon −17.25 to −8.50, lat 20.71 to 27.75. Climate values are copied cell for cell, with no resampling.
- **Mask:** the  Sahara land borders (8°40′ W, 26° N, 12° W, then the line to 21°20′ N, and the Cap Blanc peninsula). Algeria and Mauritania are excluded. TerraClimate is already empty over the sea. Along the straight-line borders, expect ±1 cell.
- **Fits with the northern grid:** this grid's top edge is at 27.75° N and the first cell centres are below 27.7436° N. The northern grid stops at 27.7436° N. Together the two cover Morocco and  Sahara without overlap.
- **Same units as the northern files:** tmax/tmin in °C, prec in mm/month, vap in kPa, ws in m/s, def in mm/month. Files are float32 with nodata −9999.

## Run order (Windows cmd)

```bat
cd C:\path\to\climate-prediction--sahara
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**1. Download the data.** Try a short range first, then the full range:
```bat
python download/download_terraclimate.py --years 2020 2024
python download/download_terraclimate.py
```
- This makes 402 small requests (6 variables × 67 years), each covering only the  Sahara box. Time depends on the TerraClimate server, not on your PC.
- It also builds `data/wsahara_elev.tif`. It downloads the WorldClim 2.1 30″ elevation once (a few hundred MB) and averages it onto the grid. If you already have the global `wc2.1_30s_elev.tif`, use it instead with `--elev-file C:\path\wc2.1_30s_elev.tif`. The northern project's copy won't work, because it's cropped to the north.
- It resumes: years whose 12 files exist are skipped. Failures are listed in `data/download_failures.txt`. Run the same command again to retry them.
- **Source:** the Climatology Lab THREDDS server (`thredds.northwestknowledge.net:8080`). It uses the NetCDF Subset Service, and falls back to OPeNDAP. Force one with `--source ncss` or `--source opendap`.

**2. Train and forecast**
```bat
python ML/projectionML_wsahara.py
python DL/projectionDL_wsahara.py
```
To keep the data somewhere else, set it first with `set WSAHARA_CLIMATE_DATA=D:\wsahara\data`.

## Outputs

Each pipeline writes:
- `predictions_wsahara_{ML|DL}/{year}/{var}/wsahara_{var}_{year}.tif` for 2025–2050, for all 6 variables plus `tavg`.
- Monthly files `wsahara_{var}_{year}_{MM}.tif` for tmax, tmin and prec.
- `predictions_summary.csv`: regional means with persistence and climatology baselines. Regions are the whole area, north and south of 25° N, a coastal strip (≤ 0.5° from the coast), the interior, lowlands (< 150 m) and plateau (≥ 400 m).
- `test_metrics.csv`.

## What changed from the northern pipelines, and why

| Setting | North |  Sahara | Why |
|---|---|---|---|
| `MIN_ANNUAL_PREC` | 50 mm | 0 | Most of  Sahara gets less than 50 mm/year, so the filter would delete it. |
| `ELEVATION_THRESHOLDS` | 500 / 1500 m | 150 / 400 m | There are no mountains. The strata become coastal lowlands, plains and plateaus. |
| `PRECIPITATION_THRESHOLDS` | 200 / 400 mm | 30 / 60 mm | Everything would otherwise land in the single "arid" stratum. |
| `N_PIXELS_PER_STRATA` | 150 | 300 | Fewer distinct strata, so more pixels are taken from each one. |
| File prefix | `morocco_` | `wsahara_` | Keeps the two regions' files apart. |
| Monthly prec split | ÷ annual climatology | same, with 1/12 where the climatology is 0 mm | Avoids a divide-by-zero in the driest cells. |

Everything else is the same as the fast northern pipelines. That includes the features, the model architectures and hyperparameters, the splits (XGBoost 1981–2024 and LSTM 1958–2024, with validation 2015–2017 and test 2018–2024), exact year and month file matching, and GPU use on the RTX 3090.

## Things to know

- **Rain forecasts here will be weak.** Many months have 0 mm, and a whole year's rain can come from one storm. Compare prec skill against the climatology baseline in `test_metrics.csv` before trusting it. Temperature, vapour pressure, wind and deficit should behave much better.
- **TerraClimate is thinner in  Sahara.** There are very few weather stations, so the values lean more on reanalysis and interpolation than in the north. Treat fine spatial detail with caution.
- **Tested here:** with a mock server returning synthetic TerraClimate files, the full 1958–2024 download ran and produced 4,824 monthly files plus elevation on the right grid, with no cells outside the borders. Both pipelines then ran end to end on that data (the LSTM with 2 epochs, on CPU). The real THREDDS server couldn't be reached from the test environment, so the first real download is the first live test.
