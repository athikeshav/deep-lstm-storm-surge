# Deep LSTM-Based Storm-Surge Height Forecasting Using Satellite and Ocean-Weather Telemetry

> A transparent, end-to-end forecasting system that turns time-stamped
> ocean and weather observations into a short-horizon storm-surge height
> forecast, exposes how the model reached its output, and communicates
> uncertainty and data quality — not just a number on a screen.
>
> **Safety boundary:** this is an **educational / research prototype**,
> **not an operational warning service**. Do not use it for evacuation
> or public-safety decisions.

---

## 1. What this project delivers

| Component | Where | What |
|-----------|-------|------|
| **Pipeline** | `src/data/`, `src/model/`, `src/evaluate/`, `src/inference/` | Ingestion → validation → time alignment → feature engineering → windowing → training → evaluation → inference, every stage logged to a JSON-lines trace |
| **Trained model + baselines** | `models/lstm_model.keras`, `models/baseline_linear.joblib`, `models/scalers.joblib` | Deep LSTM (LSTM-64 → LSTM-32 → Dense-16 → Dense-1), trained on real NOAA data; persistence + linear ridge baselines |
| **Evaluation artifacts** | `reports/metrics.json`, `reports/timeseries.png`, `reports/scatter.png`, `reports/residuals.png`, `reports/peak_events.csv` | MAE, RMSE, bias, peak-event MAE, prediction-interval coverage, baseline skill scores, per-event peak table |
| **Dashboard** | `dashboard/app.py` | Streamlit app with replay mode, observed-vs-predicted, telemetry panel, behind-the-scenes trace, model card, explainability ablation |
| **Documentation** | `README.md`, `docs/MODEL_CARD.md`, `docs/EVALUATION_REPORT.md`, `docs/DATA_DICTIONARY.md`, `docs/REPRODUCIBILITY.md`, `docs/DEMO_SCRIPT.md` | Full handoff documentation |
| **Tests** | `tests/test_pipeline.py` | Smoke tests for the major pipeline stages |
| **Reproducibility** | `reports/environment.json`, `config/config.yaml`, `src/utils/reproducibility.py` | Pinned environment snapshot, central config, seeded RNGs |

## 2. Study site and target

- **Station:** The Battery, NY (NOAA CO-OPS station **8518750**, lat 40.7006, lon -74.0142)
- **Target variable:** `storm_surge_residual = observed_water_level_m − predicted_astronomical_tide_m`
  - Both quantities are observed/predicted at the same MLLW datum and the same 6-min cadence
  - The difference removes the regular tidal cycle and isolates the **weather-driven abnormal rise**
  - **This is NOT total water level** and NOT significant wave height
- **Forecast horizon:** 3 hours ahead
- **Lookback window:** 24 hours
- **Cadence:** hourly
- **History window:** 2021-09-01 → 2023-12-31 (~28 months)

## 3. Data sources (honest attribution)

| Variable | Source station | Reason |
|----------|----------------|--------|
| `water_level_m` | The Battery (8518750) | Tide-gauge observation; basis for the target |
| `predicted_tide_m` | The Battery (8518750) | NOAA tide model prediction; subtracted to compute surge residual |
| `wind_speed_ms`, `wind_direction_deg` | **Kings Point (8516945)** — 23 km away | The Battery has no anemometer; Kings Point is the nearest NOAA met station |
| `air_pressure_hpa`, `air_temperature_c`, `water_temperature_c` | The Battery (8518750) | Barometric pressure is the dominant non-wind surge driver (inverse barometer) |

All data are fetched from the **NOAA CO-OPS Data API**
(`https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`) and saved
as raw CSV with a complete provenance manifest at `data/raw/manifest.json`.

## 4. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (re-fetches real NOAA data; takes ~5 min)
python scripts/01_fetch_data.py
python scripts/02_validate_align.py
python scripts/03_create_features.py
python scripts/05_train_lstm.py     # trains baselines + LSTM (~2 min on CPU)
python scripts/06_evaluate.py

# 3. Launch the dashboard
bash scripts/07_run_dashboard.sh
# Open http://localhost:8501
```

If the NOAA API is unreachable (offline demo), force the synthetic
generator that produces a physically plausible surrogate series with the
exact same schema:

```bash
python scripts/01_fetch_data.py --synthetic
```

Already-fetched chunks are **cached** — re-running `01_fetch_data.py`
skips files that already exist locally.

## 5. Pipeline architecture (Section 5 of the scope)

```
                Stage 1                Stage 2                Stage 3
            ┌─────────────┐       ┌──────────────┐       ┌──────────────┐
raw feeds ─▶│ Ingestion   │──────▶│ Validation   │──────▶│ Time align   │
            │ (NOAA API)  │       │ (schema,     │       │ (hourly grid)│
            │             │       │  gaps, flags)│       │              │
            └─────────────┘       └──────────────┘       └──────────────┘
                                                                 │
                                                                 ▼
            ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
            │ Presentation │◀──────│ Inference    │◀──────│ Features     │
            │ (Streamlit)  │       │ (replay)     │       │ (wind u/v,   │
            │              │       │              │       │  Δp, lags,   │
            └──────────────┘       └──────────────┘       │  cyclical)   │
                    ▲                                       └──────────────┘
                    │                                              │
            ┌──────────────┐       ┌──────────────┐               │
            │ Evaluation   │◀──────│ Training     │◀──────────────┘
            │ (metrics,    │       │ (LSTM +      │
            │  plots,      │       │  baselines)  │
            │  peak events)│       │              │
            └──────────────┘       └──────────────┘
```

Every stage writes a JSON-lines trace record into `logs/pipeline_trace.jsonl`
so a reviewer can inspect what happened at each step.

## 6. Model summary (Section 6 of the scope)

**Deep LSTM** (TensorFlow/Keras):

```
Input [batch, 24 hours, 16 features]
  → LSTM(64, return_sequences=True)   → Dropout(0.2)
  → LSTM(32, return_sequences=False) → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1, linear)
```

- **Loss:** Huber (robust to storm-peak outliers)
- **Optimizer:** Adam, lr=1e-3, ReduceLROnPlateau
- **Batch size:** 64 · **Epochs:** 40 with early stopping (patience=6)
- **No shuffling across time** — chronological splits only
- **Scalers fit on training data only**; the same parameters are applied to
  validation, test, and inference (saved in `models/scalers.joblib`)
- **Uncertainty:** Monte-Carlo dropout (30 stochastic forward passes;
  80% / 95% prediction intervals from the empirical quantiles)

**Baselines** (the LSTM must beat these to be useful):

- **Persistence:** `ŷ(t+3h) = surge(t)` (the latest observed surge)
- **Linear regression:** ridge regression on the flattened 24h window

## 7. Evaluation results (Section 7 of the scope)

**Test period:** 2023-07-02 → 2023-12-31 · **n = 4,201 samples**

| Model              | MAE (m) | RMSE (m) | Bias (m)  | Peak-event MAE (m) | Coverage 80% | Coverage 95% |
|--------------------|---------|----------|-----------|--------------------|--------------|--------------|
| **LSTM**           | **0.0540** | 0.0789 | +0.0057 | 0.1133 | 27.1% | 37.8% |
| Persistence        | 0.0655 | 0.0989 | -0.0000 | 0.1010 | — | — |
| Linear regression  | 0.0517 | 0.0762 | -0.0026 | 0.0982 | — | — |

- **Skill score vs persistence:** **+17.5%** (the LSTM beats persistence)
- **Skill score vs linear regression:** **-4.5%** (linear ridge slightly beats the LSTM — honest reporting)
- **Peak-event MAE:** top 5% of surge events (≥0.417 m, 213 hours); 18 distinct storm events identified

### Honest limitations (also see `docs/MODEL_CARD.md`)

1. **Prediction intervals are under-calibrated.** MC dropout gives 27% /
   38% coverage for the nominal 80% / 95% intervals. Treat the
   interval bounds as approximate, not calibrated.
2. **The LSTM does not yet beat linear regression.** With a longer
   training history (we used ~16 months) and more storm events, the LSTM
   is expected to improve relative to baselines.
3. **Wind comes from a different station.** Kings Point (8516945) is
   23 km from The Battery; spatial separation introduces representativeness
   error.
4. **Few major surge events.** Only 18 distinct storm events reached the
   95th-percentile threshold in the test period. Storm-surge forecasting
   is fundamentally a low-signal, rare-event problem.
5. **Not an operational warning system.** This is a research prototype.
   Public safety requires official forecasts and validated local data.

## 8. Project structure

```
deep-lstm-storm-surge/
├── README.md                       ← you are here
├── requirements.txt
├── config/
│   └── config.yaml                 ← central, single-source-of-truth config
├── data/
│   ├── raw/                        ← NOAA CSV chunks + manifest.json
│   ├── processed/                  ← validated/aligned/features/predictions parquet
│   └── DATA_DICTIONARY.md
├── src/
│   ├── config.py                   ← path + config loader
│   ├── data/                       ← ingest, validate, align, features
│   ├── model/                      ← dataset, lstm, baseline, train
│   ├── evaluate/                   ← metrics, evaluate
│   ├── inference/                  ← replay forecast + data-quality status
│   └── utils/                      ← logging, reproducibility, timeseries
├── models/                         ← lstm_model.keras, baseline_linear.joblib,
│                                     scalers.joblib, windows.npz, training_history.json
├── reports/                        ← metrics.json, *.png plots, peak_events.csv,
│                                     environment.json, evaluation_report.md, model_card.md
├── dashboard/
│   └── app.py                      ← Streamlit dashboard with replay mode
├── scripts/                        ← 01_fetch_data.py … 07_run_dashboard.sh
├── tests/
│   └── test_pipeline.py
└── logs/                           ← pipeline.log, pipeline_trace.jsonl, training_log.csv
```

## 9. Definition of Done (Section 13 of the scope)

| Criterion | Status |
|-----------|--------|
| Another person can run it from documented instructions | ✅ — see Quick Start |
| Inspect the data provenance | ✅ — `data/raw/manifest.json`, `docs/DATA_DICTIONARY.md` |
| Reproduce preprocessing and model inference | ✅ — `scripts/01-06`, seeded RNGs, pinned env |
| See a genuine held-out evaluation against a baseline | ✅ — `reports/metrics.json`, `docs/EVALUATION_REPORT.md` |
| Replay a historical forecast without leakage | ✅ — dashboard "⏪ Replay" mode |
| Understand the model's limitations | ✅ — `docs/MODEL_CARD.md`, dashboard model-card view |

## 10. License

- **Source code:** MIT
- **Data:** Public domain (NOAA CO-OPS)
- **Model weights & derived artifacts:** CC0

## 11. Citation

If you use this work in an academic context, please cite as:

> Deep LSTM-Based Storm-Surge Height Forecasting Using Satellite and Ocean-Weather Telemetry.
> Educational research prototype, v1.0.0. The Battery, NY study site, 2021-09 → 2023-12.
