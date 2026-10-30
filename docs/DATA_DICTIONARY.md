# Data Dictionary

This document records every column used in the pipeline: its source,
units, timezone, missing-value convention, and the role it plays in
the model. It satisfies Section 4 (data inventory) of the scope
document.

---

## 1. Raw layer — `data/raw/`

Raw CSV files fetched directly from the NOAA CO-OPS Data API. Each
file is one (product, station, ~30-day chunk) triple. The columns
below are the **normalised** schema written by `src/data/ingest.py`;
NOAA's verbose original headers (`Date Time`, `Water Level`,
`Prediction`, `Pressure`, `Speed`, `Direction`, etc.) are renamed on
save.

| File pattern | Columns (after normalisation) |
|--------------|------------------------------|
| `water_level_<station>_<dates>.csv` | `t, v, s, f, q` |
| `predictions_<station>_<dates>.csv` | `t, v` |
| `wind_<station>_<dates>.csv`        | `t, s, d, g` |
| `air_pressure_<station>_<dates>.csv` | `t, v` |
| `air_temperature_<station>_<dates>.csv` | `t, v` |
| `water_temperature_<station>_<dates>.csv` | `t, v` |

### Raw column meanings

| Column | Type | Unit | Meaning |
|--------|------|------|---------|
| `t` | string (`YYYY-MM-DD HH:MM`) | UTC | Timestamp of the observation |
| `v` | float | product-specific (see below) | The measured value |
| `s` | float | m | Sigma (1σ uncertainty of the water-level measurement) |
| `f` | string | — | NOAA quality flag string (comma-separated) |
| `q` | string | — | NOAA quality: `v` = verified, `p` = preliminary, `r` = rejected |
| `s` (wind) | float | m/s | Wind speed |
| `d` (wind) | float | degrees | Wind direction (0=N, 90=E) |
| `g` (wind) | float | m/s | Wind gust |

### Units per product

| Product | Unit | Notes |
|---------|------|-------|
| `water_level` | metres | above MLLW datum |
| `predictions` | metres | above MLLW datum (NOAA tide model) |
| `wind` (speed) | m/s | |
| `wind` (direction) | degrees | 0–360 |
| `air_pressure` | hPa | |
| `air_temperature` | °C | |
| `water_temperature` | °C | |

### Manifest — `data/raw/manifest.json`

For every raw file we record:
- `product`, `station_id`, `begin`, `end`
- `status` (`ok` / `failed`)
- `rows` (count)
- `path` (relative to project root)
- `url` (the exact NOAA API request that produced this file)
- `retrieved_at` (UTC ISO timestamp, or `cached`)
- `source` (`NOAA CO-OPS Data API` or `synthetic_generator_v1`)
- `license` (`Public domain (NOAA)` or `CC0`)
- `station_name`, `datum`, `units`, `timezone`

---

## 2. Processed layer — `data/processed/`

### `validated.parquet`

Outer-join of every product on timestamp. Index = UTC datetime.
Per-product columns include the measurement plus a `_suspect` boolean
(True if outside the plausible physical bounds) and a shared
`_noaa_quality` string.

| Column | Type | Unit | Bounds | Source |
|--------|------|------|--------|--------|
| `water_level_m` | float | m | [-2.5, 3.5] | tide_gauge_noaa_8518750 |
| `water_level_m_suspect` | bool | — | — | derived |
| `predicted_tide_m` | float | m | [-2.0, 3.0] | noaa_tide_model_8518750 |
| `predicted_tide_m_suspect` | bool | — | — | derived |
| `wind_speed_ms` | float | m/s | [0, 60] | anemometer_noaa_8516945 |
| `wind_speed_ms_suspect` | bool | — | — | derived |
| `wind_direction_deg` | float | deg | [0, 360] | anemometer_noaa_8516945 |
| `wind_direction_deg_suspect` | bool | — | — | derived |
| `air_pressure_hpa` | float | hPa | [950, 1060] | barometer_noaa_8518750 |
| `air_pressure_hpa_suspect` | bool | — | — | derived |
| `air_temperature_c` | float | °C | [-25, 45] | thermometer_noaa_8518750 |
| `air_temperature_c_suspect` | bool | — | — | derived |
| `water_temperature_c` | float | °C | [-3, 35] | thermometer_noaa_8518750 |
| `water_temperature_c_suspect` | bool | — | — | derived |
| `_noaa_quality` | string | — | `v`/`p`/`r`/`missing` | NOAA |

### `aligned.parquet`

Same columns as `validated.parquet`, but reindexed onto a strict
hourly UTC grid covering `history_start` → `history_end`. Missing
values inside a 3-hour window are linearly interpolated; longer gaps
remain NaN. Each measurement column gains a `<col>_was_missing`
boolean that records whether that hour was inserted by reindexing.

### `features.parquet`

Adds the storm-surge target and engineered features:

| Column | Type | Unit | How derived |
|--------|------|------|-------------|
| `storm_surge_residual` | float | m | **target** = `water_level_m − predicted_tide_m` |
| `wind_u_ms` | float | m/s | `wind_speed_ms * cos(deg2rad(wind_direction_deg))` |
| `wind_v_ms` | float | m/s | `wind_speed_ms * sin(deg2rad(wind_direction_deg))` |
| `pressure_change_3h_hpa` | float | hPa | `air_pressure_hpa.diff(periods=3)` |
| `water_level_lag1_m` | float | m | `water_level_m.shift(1)` |
| `surge_lag1_m` | float | m | `storm_surge_residual.shift(1)` |
| `hour_sin` / `hour_cos` | float | — | `sin(2π·hour/24)`, `cos(2π·hour/24)` |
| `doy_sin` / `doy_cos` | float | — | `sin(2π·day_of_year/365.25)`, `cos(2π·day_of_year/365.25)` |

Every engineered column also gets a `<col>_was_missing` boolean.

### `test_predictions.parquet`

Held-out test-period predictions, one row per (window, model):

| Column | Type | Unit | Meaning |
|--------|------|------|---------|
| `t` | datetime (UTC) | — | Forecast issue time + horizon |
| `y_true` | float | m | Observed surge at forecast time |
| `lstm_pred` | float | m | LSTM point forecast (mean of MC ensemble) |
| `lstm_lo80` / `lstm_hi80` | float | m | 80% prediction interval (10th / 90th percentile of MC ensemble) |
| `lstm_lo95` / `lstm_hi95` | float | m | 95% prediction interval (2.5 / 97.5 percentile) |
| `persistence_pred` | float | m | Persistence baseline forecast |
| `linear_pred` | float | m | Linear ridge baseline forecast |

---

## 3. Model artifacts — `models/`

| File | Type | Contents |
|------|------|----------|
| `lstm_model.keras` | TF/Keras model | The trained Deep LSTM (architecture in `config.yaml`) |
| `baseline_linear.joblib` | joblib | `StandardScaler` + `Ridge` regressor + lookback/horizon |
| `scalers.joblib` | joblib | Feature means/stds (train-fitted), target mean/std, lookback/horizon, feature column order |
| `windows.npz` | numpy archive | `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, plus the datetime index for each split |
| `training_history.json` | JSON | Per-epoch `loss` / `val_loss` / `mae` / `val_mae` / `rmse` / `val_rmse` / `learning_rate` |

---

## 4. Reports — `reports/`

| File | Contents |
|------|----------|
| `metrics.json` | Final evaluation metrics for LSTM + both baselines (MAE, RMSE, bias, peak-event MAE, coverage, skill scores) |
| `timeseries.png` | Test-period observed vs LSTM forecast with 80%/95% prediction intervals |
| `scatter.png` | Predicted vs observed scatter (LSTM and persistence overlaid) |
| `residuals.png` | Residual histogram + residuals-vs-observed scatter |
| `peak_events.csv` | Per-event peak surge summary (event_id, start, n_hours, peak_true, peak_pred, mae) |
| `environment.json` | Package versions of every library used at training time |
| `evaluation_report.md` | Human-readable evaluation narrative (see `docs/EVALUATION_REPORT.md`) |
| `model_card.md` | Model card per scope Section 8 (see `docs/MODEL_CARD.md`) |

---

## 5. Missing-value conventions

- **Raw:** missing values appear as gaps in the time series (rows absent
  from the NOAA response) or as empty cells. NOAA uses the `q` column
  to flag rejected observations (`q = "r"`).
- **Validated:** every column is present on every retained timestamp;
  suspect values are flagged in `<col>_suspect`, never silently removed.
- **Aligned:** the hourly grid is dense. Missing values are kept as
  NaN, with `<col>_was_missing = True`. Linear interpolation is
  applied only inside 3-hour gaps; longer gaps propagate as NaN and
  the affected windows are dropped at window-creation time.
- **Windows:** a window is only emitted if every feature and the
  target are non-NaN across all 24 lookback steps AND the target at
  `t + horizon` is non-NaN.

---

## 6. Source licenses and attribution

| Source | License | URL |
|--------|---------|-----|
| NOAA CO-OPS Data API | Public domain | https://api.tidesandcurrents.noaa.gov/api/prod/ |
| NOAA station 8518750 (The Battery) | Public domain | https://tidesandcurrents.noaa.gov/stationhome.html?id=8518750 |
| NOAA station 8516945 (Kings Point) | Public domain | https://tidesandcurrents.noaa.gov/stationhome.html?id=8516945 |
| Synthetic generator (project) | CC0 | `src/data/ingest.py::fetch_synthetic` |

Wind and water-level data are sourced from **different physical
stations** (Kings Point vs The Battery, ~23 km apart). This is
documented explicitly in the telemetry panel of the dashboard and in
the model card.
