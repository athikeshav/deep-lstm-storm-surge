# Evaluation Report

> Per scope Section 7: time-series plot, predicted-vs-observed scatter,
> residual distribution, baseline comparison, event-wise peak analysis.
> Ordinary and peak periods evaluated separately. State test dates and
> number of relevant events.

---

## 1. Test setup

| Field | Value |
|-------|-------|
| Test period | 2023-07-02 02:00 UTC → 2023-12-31 00:00 UTC |
| Number of test samples | 4,201 hourly forecasts |
| Number of peak events (≥95th percentile) | 18 distinct storm events, 213 hourly samples |
| 95th-percentile surge threshold | 0.417 m |
| Forecast horizon | 3 hours ahead |
| Lookback window | 24 hours |
| Target | `storm_surge_residual` (m) = observed WL − predicted astronomical tide |

---

## 2. Overall metrics

| Model | MAE (m) | RMSE (m) | Bias (m) | Skill vs Persistence |
|-------|---------|----------|----------|----------------------|
| **LSTM** | **0.0540** | **0.0789** | +0.0057 | **+17.5%** |
| Persistence | 0.0655 | 0.0989 | -0.0000 | (baseline) |
| Linear regression (Ridge) | 0.0517 | 0.0762 | -0.0026 | -0.8% (i.e. just slightly better than persistence) |

**Reading the table:**

- The LSTM beats persistence by **17.5%** on MAE. This is a
  meaningful improvement: the model is doing more than echoing the
  most recent observation.
- Linear regression is the **strongest baseline**, with a MAE of
  0.0517 m. The LSTM does not yet beat it (skill score −4.5%). The
  LSTM has not earned its complexity on this dataset / horizon.
- All three models have small bias (|bias| < 6 mm), indicating no
  systematic over- or under-forecasting at ordinary surge levels.

## 3. Peak-event analysis

| Model | Peak-event MAE (m) | Notes |
|-------|-------------------|-------|
| LSTM | 0.1133 | Larger error at peaks than overall; the model under-resolves surge maxima |
| Persistence | 0.1010 | Persistence is competitive at peaks because surge is autocorrelated |
| Linear regression | 0.0982 | Best peak-event performance |

**Per-event peak surge table** (`reports/peak_events.csv`): 18 distinct
events identified by clustering hourly peaks above the 95th-percentile
threshold with a 6-hour gap criterion. For each event we record:
- start time
- duration in hours
- peak observed surge
- peak predicted surge
- mean absolute error over the event

The LSTM's peak-event MAE (11.3 cm) is roughly 2× its overall MAE
(5.4 cm). This is the expected failure mode for regression-based
forecasters on rare events — the loss surface rewards being close on
ordinary samples, and there are too few peak samples to dominate the
loss.

## 4. Prediction intervals

| Method | Target | Empirical coverage | Calibration status |
|--------|--------|-------------------|--------------------|
| LSTM 80% PI (MC dropout) | 80% | **27.1%** | Under-calibrated |
| LSTM 95% PI (MC dropout) | 95% | **37.8%** | Under-calibrated |

The Monte-Carlo dropout intervals are too narrow. This is a known
limitation of MC dropout as an uncertainty quantifier — it tends to
underestimate epistemic uncertainty and ignores aleatoric noise. For
production use we would recommend:

1. **Conformal prediction** on top of the LSTM point forecasts to
   guarantee nominal coverage.
2. **Quantile regression** (pinball loss) with separate heads for
   10th / 50th / 90th percentiles.
3. **Ensemble of independently trained LSTMs** to capture model
   uncertainty across random initializations.

## 5. Visualisations

All plots are saved as PNG in `reports/`:

### `reports/timeseries.png`
Observed surge and LSTM forecast across the full test period, with
80% and 95% prediction-interval bands. The plot shows:
- LSTM forecast (red) tracks observed surge (dark blue) closely at
  ordinary levels.
- The model under-resolves the largest peaks (events where observed
  surge > 0.6 m are predicted lower).
- The 95% PI band is too narrow to contain the observed values at
  peak times.

### `reports/scatter.png`
Predicted-vs-observed scatter for the LSTM (dark blue) and persistence
(grey). The dashed red line is `y = x`. The LSTM scatter is tighter
around the diagonal than persistence, especially at moderate surge
levels (0.2–0.5 m).

### `reports/residuals.png`
Two panels:
1. Residual histogram (forecast − observed). The distribution is
   approximately Gaussian centered at +0.0057 m (small positive bias)
   with heavier-than-Gaussian tails at the extremes (driven by missed
   peaks).
2. Residuals vs observed surge. The residuals slope upward at high
   observed values — the model systematically under-predicts large
   surges.

## 6. Replay demonstration

The dashboard's "⏪ Replay" mode lets a reviewer pick any historical
timestamp in the test period, run the LSTM using only data knowable up
to that point, and then reveal the observed surge value.

A worked example: **issue time 2023-09-29 18:00 UTC** (during
Tropical Storm Ophelia's NYC impact):

| Quantity | Value |
|----------|-------|
| Forecast time (t + 3h) | 2023-09-29 21:00 UTC |
| LSTM forecast | +0.258 m |
| 80% prediction interval | [+0.233, +0.283] |
| 95% prediction interval | [+0.228, +0.289] |
| **Observed surge at forecast time** | **+0.270 m** |
| **Forecast error** | **−0.012 m (1.2 cm)** |

The forecast is within 1.2 cm of the observed value, well inside
the 95% PI. Replay is the strongest demonstration that the system
is more than a static UI — it shows the actual workflow end-to-end
with no information leakage.

## 7. Honest interpretation

The scope's central claim is that *a carefully defined, evaluated
model can turn real historical coastal and weather observations into a
measurable forecast*. This project demonstrates that end-to-end:

1. ✅ The LSTM beats persistence on MAE (+17.5% skill) — the simplest
   baseline.
2. ❌ The LSTM does not yet beat linear regression. This is reported
   honestly and the model card lists reasons (short training period,
   starting architecture values, etc.).
3. ✅ The pipeline is fully reproducible: real NOAA data, chronological
   splits, train-only scalers, seeded RNGs, pinned environment.
4. ✅ Every step is inspectable: a JSON-lines trace records the
   transformation from raw rows to model output.
5. ✅ Replay mode demonstrates no-leakage inference on a historical
   timestamp, with the observed truth hidden until the user reveals it.
6. ⚠️ Prediction intervals are not calibrated — this is a real
   limitation, not hidden.

## 8. Recommendations for further work

- **Longer training history** (3–5+ years) to expose the LSTM to more
  rare events.
- **Conformal prediction** for calibrated intervals.
- **Tropical cyclone event hold-out** to specifically evaluate the
  model on the events that matter most operationally.
- **Multi-horizon forecasting head** (1h, 3h, 6h, 12h, 24h) using
  a shared encoder.
- **Add satellite-derived sea-surface height anomaly** as a supplementary
  feature — explicitly per the scope's "satellite role" guidance.
- **Tune LSTM architecture** (units, dropout, layer count) using
  Bayesian optimisation on the validation period.
