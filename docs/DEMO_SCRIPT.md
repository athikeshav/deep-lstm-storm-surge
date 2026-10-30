# Demonstration Script

> Per scope Section 8: a strong live demo of the actual workflow. This
> script walks a reviewer through the system in 8 minutes.

---

## Setup (before the demo)

```bash
# Ensure all artifacts are in place (data fetched, model trained)
ls models/lstm_model.keras reports/metrics.json dashboard/app.py

# Launch the dashboard
bash scripts/07_run_dashboard.sh
# Open http://localhost:8501 in a browser
```

If the model is not yet trained, run:

```bash
python scripts/01_fetch_data.py     # ~5 min (uses cache if data exists)
python scripts/02_validate_align.py
python scripts/03_create_features.py
python scripts/05_train_lstm.py     # ~2 min on CPU
python scripts/06_evaluate.py
```

---

## Demo script (8 minutes)

### Part 1 — The forecast problem (1 min)

> "This is a Deep LSTM that forecasts storm-surge residual height 3
> hours ahead at The Battery, NY — the southern tip of Manhattan.
> Storm surge is the weather-driven rise of water above the regular
> astronomical tide. We isolate it by subtracting NOAA's tide model
> prediction from the observed water level. This is **not** total
> water level and not wave height.
>
> The forecast horizon is 3 hours. The model uses the previous 24
> hours of ocean and weather observations as input."

Point to: dashboard top banner (site, station, version, safety boundary).

### Part 2 — Pipeline trace (1 min)

Switch sidebar to **🔍 Behind-the-scenes trace**.

> "Every pipeline stage writes a JSON-lines record. Here we can see:
> ingestion produced 197,549 raw rows per product; validation dropped
> duplicates and flagged suspect values; alignment produced 20,425
> hourly rows; feature engineering added 9 derived features; windowing
> created supervised tensors. **No step is hidden.**"

Point to: the trace table showing ingestion → validate → align →
features → train → evaluate stages.

### Part 3 — Held-out evaluation (2 min)

Switch sidebar to **📊 Forecast & evaluation**.

> "The test period is July 2023 through December 2023 — 4,201 hourly
> forecasts. The LSTM achieves a mean absolute error of 5.4 cm, which
> is a 17.5% improvement over the persistence baseline. **However**,
> linear regression is slightly better — we report this honestly in
> the model card and discuss reasons."

Point to: the metric cards, then the time-series chart, then the
scatter + baseline table.

> "Notice the 95% prediction interval band on the time series —
> empirical coverage is only 38%, well below the nominal 95%. This is
> a known limitation of Monte-Carlo dropout; we discuss it in the
> model card."

Point to: the under-coverage in the model card table.

### Part 4 — Telemetry panel (1 min)

Switch sidebar to **🧰 Input telemetry**.

> "Every input feature has a documented source. Wind comes from a
> **different** NOAA station (Kings Point, 23 km away) because The
> Battery doesn't have an anemometer. We're explicit about this —
> mixing data sources is honest, not hidden. Each feature shows its
> latest valid timestamp, missingness count, and coverage percentage."

Point to: the telemetry table, especially the Source column showing
two different station IDs.

### Part 5 — Replay mode (2 min)

Switch sidebar to **⏪ Replay a historical forecast**.

> "This is the strongest live demo of the actual workflow. I pick a
> historical timestamp — let's choose 2023-09-29 18:00 UTC, during
> Tropical Storm Ophelia. The system reconstructs the input window
> using **only data knowable at that moment** — no future information
> leaks in. Then it runs the LSTM and produces a forecast for 3 hours
> later."

Click "▶ Run replay forecast".

> "The forecast is +0.258 m of surge residual, with an 80% interval
> of [+0.233, +0.283] and a 95% interval of [+0.228, +0.289]. The
> status is 'inputs stale' because we're using historical data, not
> live data. The behind-the-scenes trace shows the exact window shape
> and model version."

> "Now the truth. The observed surge 3 hours later, at 21:00 UTC,
> was... let me check the box to reveal it."

Check the "Reveal observed surge value" checkbox.

> "0.270 m. The forecast was 0.258 m. **Error: 1.2 centimetres.** The
> forecast was within the 80% prediction interval."

### Part 6 — Explainability (1 min)

Switch sidebar to **🧠 Explainability**.

> "The ablation panel sets each feature to its training mean, one at
> a time, and re-runs the forecast. This shows which features the
> model relies on. It is **not** a causal explanation — correlation
> is not causation — but it gives a reviewer a starting point for
> asking whether the model's dependencies make physical sense."

Point to: the ablation table sorted by |Δ|.

### Closing (15 seconds)

> "Every number you've seen — from raw row count to model parameters
> to peak-event MAE — is traceable through the pipeline trace, the
> model card, the evaluation report, and the data dictionary. A
> reviewer can verify each step. That is what this project delivers:
> not just a polished dashboard, but an end-to-end evidence trail."

---

## Backup Q&A

**Q: Why an LSTM rather than a transformer?**
A: The scope explicitly recommends starting with one Deep LSTM and
earning complexity on held-out data. Transformers are out of scope
for the MVP.

**Q: How did you prevent leakage?**
A: Three mechanisms: (1) chronological splits (no random shuffling
across time), (2) scalers fit on training data only, (3) the replay
forecast only uses data at or before `t_input_end`.

**Q: Why use a different station for wind?**
A: The Battery has no anemometer. Kings Point (8516945) is the nearest
NOAA met station, ~23 km away. We document this honestly in the
telemetry panel and the model card.

**Q: What would you do differently in production?**
A: Conformal prediction for calibrated intervals; longer training
history (5+ years); colocated wind sensors; multi-horizon heads; and
proper tropical cyclone event hold-outs.

**Q: Can this be deployed operationally?**
A: **No.** This is an educational prototype. Real deployment requires
validated local data, domain-expert review, rigorous extreme-event
evaluation, failover infrastructure, and integration with official
warning authorities.
