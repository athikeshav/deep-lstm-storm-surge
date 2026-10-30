# Model Card — Deep LSTM Storm-Surge Forecaster

> Per scope Section 8: training period, test period, features, lookback,
> forecast horizon, metrics, baseline comparison, known limitations,
> and dataset source.

---

## 1. Model details

| Field | Value |
|-------|-------|
| **Name** | Deep LSTM Storm-Surge Forecaster |
| **Version** | 1.0.0 |
| **Date trained** | 2026-10-29 |
| **Model type** | Stacked Long Short-Term Memory (Deep LSTM) |
| **Framework** | TensorFlow / Keras 2.21 |
| **Parameters** | ~31,500 (small by design — see Section 6) |
| **License** | MIT (code) · CC0 (weights) |

## 2. Intended use

| Field | Value |
|-------|-------|
| **Primary intended uses** | Educational / research demonstration of time-series forecasting for storm surge; investigation of how LSTM-based deep learning behaves on a real coastal problem |
| **Out-of-scope uses** | Operational warning service; evacuation decisions; public safety; insurance pricing; commercial navigation; any deployment where false negatives have severe consequences |
| **Primary intended users** | Students, researchers, coastal-risk analysts exploring the workflow |
| **Out-of-scope users** | Emergency managers without supplementary expert review |

## 3. Study site and target

- **Coastal region:** The Battery, NY (New York Harbor), NOAA station 8518750 (40.7006° N, 74.0142° W)
- **Target variable:** `storm_surge_residual` (metres) = observed water level (MLLW datum) − NOAA-predicted astronomical tide
- **Why this site:** Long, high-quality tide-gauge record; affected by
  tropical cyclones (Ida remnants Sep 2021), nor'easters, and
  extra-tropical systems; publicly available data.
- **Why this target:** Storm-surge residual isolates the weather-driven
  abnormal rise above the regular tidal cycle. It is the quantity
  relevant to coastal flooding risk. **It is not total water level**
  (which also includes waves and other effects), and it is not
  significant wave height.

## 4. Data

| Source | Station | Cadence | Period used | Role |
|--------|---------|---------|-------------|------|
| NOAA CO-OPS water_level | 8518750 (The Battery) | 6-min | 2021-09-01 → 2023-12-31 | Basis for target |
| NOAA CO-OPS predictions (astronomical tide) | 8518750 | 6-min | same | Subtracted to compute residual |
| NOAA CO-OPS wind | 8516945 (Kings Point) | 6-min | same | Wind speed & direction |
| NOAA CO-OPS air_pressure | 8518750 | 6-min | same | Barometric pressure |
| NOAA CO-OPS air_temperature | 8518750 | 6-min | same | Air temperature |
| NOAA CO-OPS water_temperature | 8518750 | 6-min | same | Water temperature |

### Coverage

- 197,549 raw rows per product at the 6-min cadence
- After hourly aggregation: 20,425 hourly rows; 19,786 (96.9%) fully complete
- Missingness: 2.7–3.0% per feature, mostly attributable to routine NOAA
  maintenance gaps

### Splits (chronological — **no random shuffling across time**)

| Split | Start | End | Hours |
|-------|-------|-----|-------|
| Train | 2021-09-01 | 2022-12-31 | ~11,232 |
| Validation | 2023-01-01 | 2023-06-30 | ~4,344 |
| Test | 2023-07-01 | 2023-12-31 | ~4,416 |

The test period contains **18 distinct storm events** reaching the 95th
percentile surge threshold of 0.417 m (213 hourly samples).

## 5. Features

The model is given a **24-hour lookback window** of **16 features**:

| Feature | Unit | Description |
|---------|------|-------------|
| `water_level_m` | m | Observed water level at The Battery |
| `predicted_tide_m` | m | NOAA astronomical tide prediction |
| `wind_speed_ms` | m/s | Wind speed (Kings Point) |
| `wind_direction_deg` | deg | Wind direction (Kings Point) |
| `air_pressure_hpa` | hPa | Barometric pressure |
| `air_temperature_c` | °C | Air temperature |
| `water_temperature_c` | °C | Water temperature |
| `wind_u_ms` | m/s | East-west wind component |
| `wind_v_ms` | m/s | North-south wind component |
| `pressure_change_3h_hpa` | hPa | 3-hour pressure change (storm approach rate) |
| `water_level_lag1_m` | m | Water level lagged 1 hour |
| `surge_lag1_m` | m | Surge residual lagged 1 hour |
| `hour_sin` / `hour_cos` | — | Hour-of-day cyclical encoding |
| `doy_sin` / `doy_cos` | — | Day-of-year cyclical encoding |

Every feature is standardized using **training-period mean and std**
only. The same parameters are persisted in `models/scalers.joblib`
and applied verbatim to validation, test, and inference inputs.

## 6. Architecture and training

```
Input [batch, 24, 16]
  ├─ LSTM(64, return_sequences=True) ─ Dropout(0.2)
  ├─ LSTM(32, return_sequences=False) ─ Dropout(0.2)
  ├─ Dense(16, activation='relu')
  └─ Dense(1, activation='linear')   ← predicted surge residual (standardized)
```

- **Loss:** Huber (robust to storm-peak outliers)
- **Optimizer:** Adam, initial learning rate 1e-3
- **LR schedule:** ReduceLROnPlateau (factor 0.5, patience 3, min_lr 1e-5)
- **Early stopping:** patience 6, restores best weights
- **Batch size:** 64
- **Epochs:** 40 max, **24 actually run** (early stop at epoch 18 best)
- **Training time:** ~99 seconds on CPU (no GPU required)
- **Random seed:** 42 (NumPy, Python, TensorFlow all seeded)

## 7. Forecast horizon

**3 hours ahead**, fixed. Given the 24-hour input window ending at
time `t`, the model predicts the surge residual at time `t + 3h`.

This horizon is short enough that the most recent observations
dominate, but long enough to be useful operationally (a real
forecasting service would also offer 6h, 12h, 24h horizons; the scope
recommends starting with one).

## 8. Evaluation metrics

**Test period:** 2023-07-02 → 2023-12-31 · **n = 4,201 hourly samples**

| Metric | LSTM | Persistence | Linear regression |
|--------|------|-------------|-------------------|
| **MAE (m)** | **0.0540** | 0.0655 | 0.0517 |
| **RMSE (m)** | 0.0789 | 0.0989 | 0.0762 |
| **Bias (m)** | +0.0057 | -0.0000 | -0.0026 |
| **Peak-event MAE (m)** | 0.1133 | 0.1010 | 0.0982 |
| **Coverage 80%** | 27.1% | — | — |
| **Coverage 95%** | 37.8% | — | — |

### Skill scores

- **vs persistence:** +17.5% — the LSTM **does** beat persistence
- **vs linear regression:** −4.5% — linear ridge slightly outperforms the LSTM

A complex model that does not beat a baseline is not a successful
result. **The LSTM beats persistence but not linear regression.** This
is reported honestly, with possible reasons in Section 10.

## 9. Baseline comparison

### Persistence baseline
- Definition: `ŷ(t+3h) = surge(t)` — forecast equals the latest observed surge
- MAE: 0.0655 m
- This is the simplest possible forecast and a fair minimum bar.

### Linear regression baseline
- Definition: ridge regression on the flattened 24-hour × 16-feature window
- MAE: 0.0517 m
- This baseline is strong because surge is autocorrelated and a
  linear map from the recent window captures most of the signal.

The LSTM's advantage over persistence (but not over linear regression)
is consistent with the literature on short-horizon surge forecasting:
deep models tend to dominate linear models only at longer horizons
(6h+).

## 10. Known limitations

1. **Prediction intervals are under-calibrated.** MC dropout (30
   forward passes) produces empirical 80% / 95% intervals that cover
   only 27% / 38% of true values. This means the model is over-confident
   about its forecasts. Real deployment should use conformal prediction
   or quantile regression for calibrated intervals.

2. **Linear regression slightly outperforms the LSTM.** The model has
   not yet earned its complexity. Hypotheses for why:
   - The training period (~16 months) is short; LSTMs typically need
     multi-year data to learn the rare-event structure of surge.
   - The Huber loss, while robust, may be too forgiving at peak events.
   - Architectural choices (layer size, dropout rate) are starting
     values from the scope; further tuning is warranted.

3. **Wind comes from a different station.** Kings Point (8516945) is
   ~23 km from The Battery. Local wind forcing at The Battery is
   approximated by Kings Point wind, introducing representativeness
   error. A better setup would use a colocated anemometer or model
   reanalysis wind fields.

4. **Few major surge events in the test set.** Only 18 distinct storm
   events reach the 95th-percentile threshold. Storm-surge forecasting
   is fundamentally a rare-event problem; the test period may not
   contain the largest events the model would face in deployment.

5. **No tropical cyclone direct hits in the test period.** Hurricane
   Ida (Sep 2021) occurred in the *training* period. The test period
   contains extra-tropical storms (notably Ophelia Sep 2023) but no
   Category 2+ cyclone direct hits. Model behavior under extreme
   tropical cyclone forcing is unverified.

6. **Not an operational warning system.** This prototype lacks the
   redundancy, real-time data ingestion, failover, and human review
   required for operational use. Decisions about evacuation or
   public safety require official forecasts.

7. **No causal reasoning.** Feature ablation (in the dashboard's
   Explainability view) reveals which features the model relies on,
   but **correlation is not causation**. The ablation table should
   not be interpreted as a causal explanation of surge physics.

## 11. Reproducibility

| Field | Value |
|-------|-------|
| Random seed | 42 |
| NumPy version | see `reports/environment.json` |
| TensorFlow version | see `reports/environment.json` |
| Training log | `logs/training_log.csv` |
| Per-epoch history | `models/training_history.json` |
| Package versions snapshot | `reports/environment.json` |
| Configuration file | `config/config.yaml` |

Re-running `scripts/05_train_lstm.py` with the same seed, environment,
and data should produce a model with the same architecture and
training trajectory (small numerical differences are possible due to
non-deterministic CPU floating-point operations; setting
`TF_ENABLE_ONEDNN_OPTS=0` reduces these).

## 12. Contact

This is an educational prototype. For questions about the project
scope or methodology, refer to the project scope document. For
operational coastal flood forecasting, consult official NOAA
forecasting services.
