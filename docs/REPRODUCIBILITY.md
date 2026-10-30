# Reproducibility Checklist

> Per scope Section 13: another person can run it from documented
> instructions, inspect data provenance, reproduce preprocessing and
> inference, see genuine held-out evaluation, replay a historical
> forecast without leakage, and understand limitations.

---

## 1. Environment

```bash
# Python 3.10+ recommended (tested with 3.12)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A snapshot of the exact package versions used to train the shipped
model is in `reports/environment.json`. Re-running `scripts/05_train_lstm.py`
on the same data with the same environment should produce a model with
the same architecture and a nearly-identical training trajectory.

## 2. Data sources

| Source | How to verify |
|--------|--------------|
| Raw CSV chunks | `data/raw/<product>_<station>_<dates>.csv` |
| Provenance manifest | `data/raw/manifest.json` — every file has a NOAA API URL, retrieval timestamp, license, and station attribution |
| Validation report | `data/processed/validation_report.json` — per-product row counts, duplicates, out-of-bounds flags, time gaps |

## 3. Reproduce preprocessing

```bash
# (Re)fetch raw data — uses cache; only fetches missing chunks
python scripts/01_fetch_data.py

# Validate, align, build features
python scripts/02_validate_align.py
python scripts/03_create_features.py
```

Verify by inspecting `logs/pipeline_trace.jsonl`: each stage writes a
JSON record with row counts, coverage, and the output parquet path.

## 4. Reproduce training

```bash
python scripts/05_train_lstm.py
```

What this script does:

1. Sets random seeds for `random`, `numpy`, `tensorflow` (`seed = 42`).
2. Calls `build_windows()` which:
   - loads `data/processed/features.parquet`
   - filters to chronological train/val/test splits
   - fits scalers on the training period only
   - persists `models/scalers.joblib`
3. Trains the LSTM (40 epochs max, early stopping patience 6).
4. Saves `models/lstm_model.keras` and `models/training_history.json`.

Per-epoch loss / val_loss / mae / rmse are written to
`logs/training_log.csv` and `models/training_history.json`.

## 5. Reproduce evaluation

```bash
python scripts/06_evaluate.py
```

Outputs:

- `reports/metrics.json` — final MAE/RMSE/bias/peak-MAE/coverage for all 3 models
- `reports/timeseries.png`, `reports/scatter.png`, `reports/residuals.png`
- `reports/peak_events.csv` — per-event peak surge analysis
- `data/processed/test_predictions.parquet` — every test-window forecast

## 6. Replay a historical forecast (no-leakage)

```bash
# Headless: forecast for a chosen historical timestamp
PYTHONPATH=. python3 -m src.inference.forecast 2023-09-29T18:00
```

Or via the dashboard:

```bash
bash scripts/07_run_dashboard.sh
# open http://localhost:8501 → "⏪ Replay a historical forecast"
```

Replay guarantees:

- The input window is built from `data/processed/features.parquet`
  using **only rows up to and including `t_input_end`**.
- No future tide, water level, or target value is used.
- The forecast timestamp is `t_input_end + 3h`.
- The user can then reveal the observed value at that future time for
  honest comparison.

## 7. Test the pipeline

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Smoke tests verify that:
- Config loads and exposes expected attributes
- Synthetic ingestion produces the expected schema
- Window creation respects lookback + horizon
- Metrics computation produces finite values
- Replay forecast returns a structured result

## 8. Versioning

| Artifact | Where to find the version |
|----------|---------------------------|
| Code | `git log` or `src/config.py::CONFIG["project"]["version"]` |
| Trained model | `models/lstm_model.keras` (Keras format embeds version metadata) |
| Configuration | `config/config.yaml` |
| Environment | `reports/environment.json` |
| Pipeline trace | `logs/pipeline_trace.jsonl` (one record per stage per run) |

## 9. Known sources of nondeterminism

Even with seeds set, small numerical differences may arise from:

1. **CPU floating-point operations** — setting `TF_ENABLE_ONEDNN_OPTS=0`
   reduces (but does not eliminate) these.
2. **Parallelism in TensorFlow's thread pool** — operation execution
   order is not deterministic across runs.
3. **NOAA API** — re-fetching data may produce slightly different
   rows if NOAA updates preliminary data (`q = "p"`) to verified
   (`q = "v"`). Use the shipped `data/raw/` for bit-exact
   reproduction.
4. **Synthetic data fallback** — uses NumPy default_rng(42) which is
   deterministic, but the synthetic series is a substitute for real
   data.

For bit-exact reproduction of the shipped model, **do not re-fetch
data** — use the `data/raw/` directory as shipped, then re-run training.

## 10. Definition of Done — checklist

| Item | Status |
|------|--------|
| Source code | ✅ under `src/` |
| Data dictionary and provenance | ✅ `docs/DATA_DICTIONARY.md`, `data/raw/manifest.json` |
| Preprocessing script | ✅ `scripts/01-03` |
| Baseline training script | ✅ `scripts/04_train_baseline.py` (also called by `05_train_lstm.py`) |
| LSTM training script | ✅ `scripts/05_train_lstm.py` |
| Saved model | ✅ `models/lstm_model.keras` |
| Evaluation report | ✅ `docs/EVALUATION_REPORT.md`, `reports/metrics.json` |
| Dashboard / replay demo | ✅ `dashboard/app.py` (run via `scripts/07_run_dashboard.sh`) |
| Model card | ✅ `docs/MODEL_CARD.md` |
| Demonstration recording / scripted walkthrough | ✅ `docs/DEMO_SCRIPT.md` |
| Reproducibility checklist | ✅ this file |
