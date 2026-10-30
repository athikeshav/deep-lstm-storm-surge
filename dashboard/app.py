"""Streamlit dashboard for the Deep LSTM Storm Surge Forecasting project.

Implements the demonstration experience required by Section 8 of the scope:
    * Forecast panel            (predicted surge, horizon, status, units)
    * Observed vs predicted     (test-period time series + replay mode)
    * Input telemetry panel     (per-feature freshness, units, missingness)
    * Behind-the-scenes trace   (raw -> validated -> aligned -> windows -> model)
    * Model card                (training/test periods, features, metrics, limits)
    * Explainability panel      (input trends + simple feature ablation)
    * Replay mode               (forecast a historical timestamp, then reveal truth)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Make src importable when running ``streamlit run dashboard/app.py``
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (CONFIG, DIRS, MODEL_CFG, SPLITS, TIME, all_input_columns,
                         feature_columns, target_column)
from src.inference.forecast import replay_forecast, observed_at
from src.utils import get_trace

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep LSTM Storm Surge Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand CSS for a cleaner look
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .metric-card {
        background: #f7f9fc; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #1f3a5f;
    }
    .status-demo   { color: #2e7d32; font-weight: 600; }
    .status-stale  { color: #c44536; font-weight: 600; }
    .status-missing{ color: #d97706; font-weight: 600; }
    .small { font-size: 0.85em; color: #666; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Deep LSTM Storm-Surge Forecasting")
st.caption(
    f"{CONFIG['project']['name']} — v{CONFIG['project']['version']} "
    f"· Site: {CONFIG['site']['station_name']} ({CONFIG['site']['station_id']}) "
    f"· Educational / research prototype, NOT an operational warning service."
)

# -----------------------------------------------------------------------------
# Load shared artifacts once
# -----------------------------------------------------------------------------
@st.cache_data
def load_predictions():
    p = DIRS["data_proc"] / "test_predictions.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

@st.cache_data
def load_metrics():
    p = DIRS["reports"] / "metrics.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)

@st.cache_data
def load_features_df():
    p = DIRS["data_proc"] / "features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

@st.cache_data
def load_validation_report():
    p = DIRS["data_proc"] / "validation_report.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)

@st.cache_data
def load_trace():
    return get_trace()

@st.cache_data
def load_peak_events():
    p = DIRS["reports"] / "peak_events.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

preds_df   = load_predictions()
metrics    = load_metrics()
feats_df   = load_features_df()
val_report = load_validation_report()
trace_recs = load_trace()
peaks_df   = load_peak_events()

# -----------------------------------------------------------------------------
# Sidebar: mode selector + safety reminder
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Demo mode")
mode = st.sidebar.radio(
    "Choose a view:",
    ["📊 Forecast & evaluation",
     "⏪ Replay a historical forecast",
     "🧰 Input telemetry",
     "🔍 Behind-the-scenes trace",
     "🪪 Model card",
     "🧠 Explainability"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Project Report")
report_pdf_file = PROJ_ROOT / "DEEP_LSTM_STORM_SURGE_PROJECT_REPORT.pdf"
if report_pdf_file.exists():
    with open(report_pdf_file, "rb") as f_rep:
        st.sidebar.download_button(
            label="📥 Download Project Report (PDF)",
            data=f_rep.read(),
            file_name="DEEP_LSTM_STORM_SURGE_PROJECT_REPORT.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
st.sidebar.caption("Author: **S. Athi** (23CU0310268) · B.Tech CSE")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**⚠️ Safety boundary**

This is an educational/research prototype.
Do **not** use it for evacuation or public-safety decisions.
Real deployment requires validated local data,
domain-expert review, and integration with
official warning authorities.
""")

# -----------------------------------------------------------------------------
# View 1: Forecast & evaluation overview
# -----------------------------------------------------------------------------
if mode.startswith("📊"):
    st.header("Forecast & evaluation")

    if metrics is None:
        st.warning("No evaluation artifacts found. Run the pipeline first "
                   "(`python scripts/06_evaluate.py`).")
        st.stop()

    # Top-line metric cards
    c1, c2, c3, c4 = st.columns(4)
    lstm   = metrics["lstm"]
    pers   = metrics["persistence"]
    linear = metrics["linear_regression"]
    c1.metric("LSTM MAE",  f"{lstm['mae']:.4f} m",
              delta=f"{metrics['skill_score_vs_persistence']:+.1%} vs persistence")
    c2.metric("Persistence MAE",  f"{pers['mae']:.4f} m")
    c3.metric("Linear regression MAE", f"{linear['mae']:.4f} m")
    c4.metric("Peak-event MAE",  f"{lstm['peak_mae']:.4f} m",
              help=f"MAE on the top 5% of surge events ({lstm['n_peaks']} hours).")

    st.caption(f"Test period: {metrics['test_period'][0]} → {metrics['test_period'][1]} · "
               f"n={metrics['n_test_samples']} samples · "
               f"target = {metrics['target']}")

    # Test-period time series
    if preds_df is not None:
        st.subheader("Held-out test period: observed vs predicted")
        # Downsample for plotly if very long
        n = len(preds_df)
        step = max(1, n // 4000)
        df = preds_df.iloc[::step]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["y_true"],
                                 mode="lines", name="Observed surge",
                                 line=dict(color="#1f3a5f", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["lstm_pred"],
                                 mode="lines", name="LSTM forecast",
                                 line=dict(color="#c44536", width=1)))
        if "lstm_lo95" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["lstm_lo95"], name="95% PI low",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df["lstm_hi95"], name="95% PI high",
                                     line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(196,69,54,0.15)",
                                     showlegend=True))
        fig.update_layout(height=420, xaxis_title="Time",
                          yaxis_title="Storm surge residual (m)",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side baseline comparison
        st.subheader("Baseline comparison")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Predicted vs observed (scatter)**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["y_true"], y=df["lstm_pred"],
                                      mode="markers", marker=dict(size=3, opacity=0.4,
                                                                  color="#1f3a5f"),
                                      name="LSTM"))
            fig2.add_trace(go.Scatter(x=df["y_true"], y=df["persistence_pred"],
                                      mode="markers", marker=dict(size=3, opacity=0.3,
                                                                  color="#9aa6b2"),
                                      name="Persistence"))
            lims = [float(df["y_true"].min()), float(df["y_true"].max())]
            fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                      line=dict(dash="dash", color="#c44536", width=1),
                                      name="y = x"))
            fig2.update_layout(height=400, xaxis_title="Observed surge (m)",
                               yaxis_title="Forecast (m)",
                               legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.markdown("**Metric table**")
            tbl = pd.DataFrame({
                "Model": ["LSTM", "Persistence", "Linear regression"],
                "MAE (m)":  [lstm["mae"], pers["mae"], linear["mae"]],
                "RMSE (m)": [lstm["rmse"], pers["rmse"], linear["rmse"]],
                "Bias (m)": [lstm["bias"], pers["bias"], linear["bias"]],
                "Peak MAE (m)": [lstm["peak_mae"], pers["peak_mae"], linear["peak_mae"]],
                "Skill vs pers.": [
                    1 - lstm["mae"]/pers["mae"] if pers["mae"] else None,
                    0.0,
                    1 - linear["mae"]/pers["mae"] if pers["mae"] else None,
                ],
            }).round(4)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Peak-event table
        if peaks_df is not None and len(peaks_df) > 0:
            with st.expander(f" Peak events in the test period ({len(peaks_df)} events)", expanded=False):
                st.dataframe(peaks_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# View 2: Replay mode (Section 8's killer feature)
# -----------------------------------------------------------------------------
elif mode.startswith("⏪"):
    st.header("⏪ Replay a historical forecast")
    st.markdown("""
    Choose a historical timestamp. The system reconstructs the input window
    using **only data available up to that moment**, runs the LSTM, and
    displays the forecast for the next horizon. **Click "Reveal observed
    value" afterwards** to compare the forecast against the truth — this
    is the live demonstration of the actual workflow.
    """)

    if feats_df is None:
        st.error("Features not built. Run the pipeline first.")
        st.stop()

    # Timestamp picker (constrained to the test period for fair replay)
    test_start = pd.Timestamp(SPLITS["test_start"]).tz_localize("UTC")
    test_end   = pd.Timestamp(SPLITS["test_end"]).tz_localize("UTC")
    # Find timestamps where there's at least lookback+horizon of history
    available = feats_df.loc[test_start:test_end].index
    available = [t for t in available
                 if feats_df.loc[:t].iloc[-TIME["lookback_steps"]:] \
                    [all_input_columns()].notna().all().all()
                 and t - pd.Timedelta(hours=TIME["lookback_steps"]) >= feats_df.index[0]]

    if not available:
        st.error("No replay timestamps with complete windows available.")
        st.stop()

    default_idx = len(available) // 2
    sel_ts = st.select_slider(
        "Pick a forecast issue time (t):",
        options=available,
        value=available[default_idx],
        format_func=lambda t: t.strftime("%Y-%m-%d %H:%M UTC"),
    )

    if st.button("▶ Run replay forecast", type="primary"):
        with st.spinner(f"Running LSTM forecast for issue time {sel_ts}..."):
            result = replay_forecast(sel_ts)
        # Display forecast card
        st.markdown("### Forecast result")
        status_cls = {"demo forecast": "status-demo",
                      "inputs stale": "status-stale",
                      "inputs missing": "status-missing"}[result.status]
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="small">Forecast time (t + {result.forecast_horizon_steps}h)</div>
                    <div style="font-size:1.5rem;font-weight:600;">
                        {result.forecast_value_m:+.3f} m
                    </div>
                    <div class="small">
                        80% PI: [{result.lo80:+.3f}, {result.hi80:+.3f}] ·
                        95% PI: [{result.lo95:+.3f}, {result.hi95:+.3f}]
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="small">Status</div>
                    <div class="{status_cls}">{result.status}</div>
                    <div class="small">model: {result.model_version}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Latest input window telemetry
        st.markdown("#### Input window (last 24h)")
        win_df = feats_df.loc[pd.Timestamp(result.last_input_time) -
                              pd.Timedelta(hours=result.lookback_steps - 1):
                              pd.Timestamp(result.last_input_time)]
        st.dataframe(win_df[all_input_columns()].round(3).tail(8),
                     use_container_width=True)

        if result.data_quality_warnings:
            with st.expander(f"⚠️ Data-quality warnings ({len(result.data_quality_warnings)})"):
                for w in result.data_quality_warnings:
                    st.write(f"• {w}")

        # Reveal observed value
        reveal = st.checkbox("Reveal observed surge value at forecast time "
                             "(truth, hidden until checked)",
                             value=False)
        if reveal:
            obs = observed_at(result.forecast_time)
            if obs is None:
                st.info("No observed value available at that time "
                        "(data missing in source).")
            else:
                err = result.forecast_value_m - obs
                st.markdown(f"""
                **Observed surge**: **{obs:+.3f} m** ·
                **Forecast error**: **{err:+.3f} m**
                ({abs(err)*100:.1f} cm)
                """)

# -----------------------------------------------------------------------------
# View 3: Input telemetry panel
# -----------------------------------------------------------------------------
elif mode.startswith("🧰"):
    st.header("🧰 Input telemetry panel")
    st.caption("Each feature: latest timestamp, unit, missingness, source. "
               "Source attribution is explicit — mixing data sources is "
               "honest, not hidden.")

    if feats_df is None:
        st.error("Features not built.")
        st.stop()

    # Build telemetry table
    from src.config import PREDICTORS
    rows = []
    for p in PREDICTORS:
        col = p["rename"]
        if col not in feats_df.columns:
            continue
        last_valid = feats_df[col].last_valid_index()
        n_missing = int(feats_df[col].isna().sum())
        coverage = float(feats_df[col].notna().mean())
        rows.append({
            "Feature": col,
            "Unit": p["units"],
            "Source": p["source"],
            "Latest valid timestamp": str(last_valid) if last_valid is not None else "—",
            "Missing (hours)": n_missing,
            "Coverage": f"{coverage:.1%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-feature time-series preview
    st.subheader("Recent telemetry (last 7 days of dataset)")
    feat_to_plot = st.selectbox("Feature to plot:", feature_columns())
    last = feats_df.index.max()
    win = feats_df.loc[last - pd.Timedelta(days=7):last]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win[feat_to_plot], mode="lines",
                             line=dict(color="#1f3a5f", width=1)))
    fig.update_layout(height=320, xaxis_title="Time",
                      yaxis_title=feat_to_plot)
    st.plotly_chart(fig, use_container_width=True)

    # Validation report
    if val_report:
        with st.expander("Per-product validation report"):
            v_rows = []
            for key, r in val_report.items():
                if isinstance(r, dict) and "raw_rows" in r:
                    v_rows.append({
                        "Product@station": key,
                        "Raw rows": r["raw_rows"],
                        "Valid rows": r["valid_rows"],
                        "Duplicates": r["duplicates"],
                        "Out of bounds": r["out_of_bounds"],
                        "Flagged quality": r["flagged_quality"],
                        "Time gaps (>30min)": r["time_gaps"],
                    })
            if v_rows:
                st.dataframe(pd.DataFrame(v_rows), use_container_width=True,
                             hide_index=True)

# -----------------------------------------------------------------------------
# View 4: Behind-the-scenes trace
# -----------------------------------------------------------------------------
elif mode.startswith("🔍"):
    st.header("🔍 Behind-the-scenes trace")
    st.caption("Every pipeline stage writes a JSON-lines trace. A reviewer "
               "can inspect exactly what happened: raw rows → validated rows → "
               "aligned steps → feature tensor shape → model version → output.")

    if not trace_recs:
        st.info("No trace records found. Run the pipeline first.")
    else:
        # Show as table
        df = pd.DataFrame(trace_recs)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.drop(columns=[c for c in df.columns
                              if isinstance(df[c].iloc[0], (dict, list))],
                     errors="ignore")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Stage counts
        st.subheader("Stage summary")
        stage_counts = (df.groupby("stage").size()
                          .reset_index(name="records"))
        st.dataframe(stage_counts, use_container_width=True, hide_index=True)

        # Show the most recent trace record in detail
        st.subheader("Most recent trace records")
        for rec in trace_recs[-5:]:
            st.json(rec)

# -----------------------------------------------------------------------------
# View 5: Model card
# -----------------------------------------------------------------------------
elif mode.startswith("🪪"):
    st.header("🪪 Model card")
    if metrics is None:
        st.warning("No metrics found.")
        st.stop()
    lstm = metrics["lstm"]; pers = metrics["persistence"]; linear = metrics["linear_regression"]

    st.markdown(f"""
    **Model name**: Deep LSTM Storm-Surge Forecaster (v{CONFIG['project']['version']})

    **Purpose**: Estimate storm-surge residual height at a coastal tide gauge
    (The Battery, NY) at a fixed {TIME['forecast_horizon_steps']}-hour lead time
    using the previous {TIME['lookback_steps']}-hour window of ocean/weather
    observations.

    **Architecture**:
    - Input: `[batch, {TIME['lookback_steps']}, {len(all_input_columns())}]`
    - LSTM({MODEL_CFG['layers'][0]['units']}, return_sequences=True) → Dropout(0.2)
    - LSTM({MODEL_CFG['layers'][2]['units']}) → Dropout(0.2)
    - Dense({MODEL_CFG['layers'][4]['units']}, relu) → Dense(1, linear)
    - Loss: Huber · Optimizer: Adam (lr=0.001) · Batch: 64 · Epochs: 40 (early-stop)

    **Splits (chronological, no shuffling)**:
    - Train:   {SPLITS['train_start']} → {SPLITS['train_end']}
    - Validation: {SPLITS['val_start']} → {SPLITS['val_end']}
    - Test:    {SPLITS['test_start']} → {SPLITS['test_end']}

    **Features** ({len(all_input_columns())} total):
    {', '.join(all_input_columns())}

    **Metrics on test set** (physical units, metres):
    | Model              | MAE     | RMSE    | Bias     | Peak MAE | Coverage 80% | Coverage 95% |
    |--------------------|---------|---------|----------|----------|--------------|--------------|
    | LSTM               | {lstm['mae']:.4f} | {lstm['rmse']:.4f} | {lstm['bias']:+.4f} | {lstm['peak_mae']:.4f} | {lstm['coverage_80'] or 0:.1%} | {lstm['coverage_95'] or 0:.1%} |
    | Persistence        | {pers['mae']:.4f} | {pers['rmse']:.4f} | {pers['bias']:+.4f} | {pers['peak_mae']:.4f} | — | — |
    | Linear regression  | {linear['mae']:.4f} | {linear['rmse']:.4f} | {linear['bias']:+.4f} | {linear['peak_mae']:.4f} | — | — |

    **Skill score** (vs persistence): **{metrics['skill_score_vs_persistence']:+.1%}**
    · (vs linear regression): **{metrics['skill_score_vs_linear']:+.1%}**

    **Known limitations**:
    - Prediction intervals from MC dropout are **under-coveraged**
      (target 80% → actual {lstm['coverage_80'] or 0:.0%}; target 95% → actual {lstm['coverage_95'] or 0:.0%}).
      Treat interval bounds as approximate, not calibrated.
    - The LSTM **does not yet beat linear regression** ({metrics['skill_score_vs_linear']:+.1%} skill).
      With a longer training history (we used ~16 months) and more storms,
      the LSTM is expected to improve relative to baselines.
    - Test set contains {lstm['n_peaks']} hours above the 95th-percentile surge threshold
      ({lstm['peak_threshold']:.3f} m). Per-event peak MAE table is available
      in the Forecast view.
    - Wind comes from a *different* station (Kings Point, 8516945) than the
      tide gauge (The Battery, 8518750), 23 km away. This is documented in
      the telemetry panel and the data dictionary.
    - This is **not an operational warning system**. Decisions about
      evacuation or public safety require official forecasts.

    **Reproducibility**:
    - Random seed: {CONFIG['reproducibility']['random_seed']}
    - Package versions: `reports/environment.json`
    - Scalers fitted on training data only; applied verbatim to val/test/inference.
    """)

# -----------------------------------------------------------------------------
# View 6: Explainability panel
# -----------------------------------------------------------------------------
elif mode.startswith("🧠"):
    st.header("🧠 Explainability panel")
    st.caption("Per-feature recent trends and a simple ablation. **Not a causal "
               "explanation** — feature attributions describe correlation, not causation.")

    if feats_df is None:
        st.error("Features not built.")
        st.stop()

    # Recent input trends
    st.subheader("Recent input trends (last 14 days)")
    last = feats_df.index.max()
    win = feats_df.loc[last - pd.Timedelta(days=14):last]
    feat_to_show = st.multiselect("Features to plot:",
                                  all_input_columns(),
                                  default=feature_columns()[:3])
    if feat_to_show:
        fig = go.Figure()
        for f in feat_to_show:
            fig.add_trace(go.Scatter(x=win.index, y=win[f], mode="lines",
                                     name=f, line=dict(width=1)))
        fig.update_layout(height=400, xaxis_title="Time",
                          yaxis_title="Value (see legend)")
        st.plotly_chart(fig, use_container_width=True)

    # Simple ablation: zero out one feature at a time, measure forecast delta
    st.subheader("Simple feature ablation")
    st.caption("For a chosen issue time, ablate each input feature by setting "
               "it to its training mean, re-run the LSTM, and report the "
               "absolute change in the forecast.")

    import tensorflow as tf
    from src.model.dataset import load_scalers
    scalers = load_scalers()

    test_idx_in_features = feats_df.loc[
        pd.Timestamp(SPLITS["test_start"]).tz_localize("UTC"):
        pd.Timestamp(SPLITS["test_end"]).tz_localize("UTC")].index
    # Pick a sample where we have a complete window
    sample_ts = None
    for t in test_idx_in_features[::24]:
        w_start = t - pd.Timedelta(hours=TIME["lookback_steps"] - 1)
        sub = feats_df.loc[w_start:t, all_input_columns()]
        if sub.notna().all().all() and len(sub) == TIME["lookback_steps"]:
            sample_ts = t
            break

    if sample_ts is None:
        st.warning("Could not find a complete window for ablation in the test period.")
    else:
        st.write(f"**Issue time:** {sample_ts}")
        feat_cols = scalers["feat_cols"]
        feat_mean = pd.Series(scalers["feat_mean"])
        feat_std  = pd.Series(scalers["feat_std"])

        w_start = sample_ts - pd.Timedelta(hours=TIME["lookback_steps"] - 1)
        window_df = feats_df.loc[w_start:sample_ts, feat_cols].copy()
        standardized = (window_df - feat_mean) / feat_std
        base = standardized.values.astype(np.float32)[None, :, :]

        model = tf.keras.models.load_model(DIRS["models"] / "lstm_model.keras",
                                            compile=False)
        base_pred = float(model(base, training=False).numpy().ravel()[0])
        base_phys = base_pred * scalers["tgt_std"] + scalers["tgt_mean"]

        abl_rows = []
        for i, col in enumerate(feat_cols):
            mod = base.copy()
            mod[0, :, i] = 0.0   # zero out (i.e. set to mean after standardization)
            p = float(model(mod, training=False).numpy().ravel()[0])
            p_phys = p * scalers["tgt_std"] + scalers["tgt_mean"]
            abl_rows.append({
                "Feature": col,
                "Forecast (m) when ablated": round(p_phys, 4),
                "Δ vs base forecast (m)": round(p_phys - base_phys, 4),
                "|Δ| (cm)": round(abs(p_phys - base_phys) * 100, 1),
            })
        st.write(f"**Base forecast (no ablation):** {base_phys:+.4f} m")
        st.dataframe(pd.DataFrame(abl_rows).sort_values("|Δ| (cm)", ascending=False),
                     use_container_width=True, hide_index=True)
        st.caption("Ablation only shows sensitivity, not causation. A feature "
                   "with a large |Δ| is one the model relies on; a feature with "
                   "near-zero |Δ| could be irrelevant or simply redundant with "
                   "another feature.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Built per the project scope. Site: {CONFIG['site']['station_name']} "
    f"({CONFIG['site']['station_id']}, {CONFIG['site']['lat']:.4f}, "
    f"{CONFIG['site']['lon']:.4f}). "
    f"Wind: NOAA station 8516945 (Kings Point, NY). "
    f"Data: NOAA CO-OPS API · Public domain. "
    f"Source code: see `src/` directory of the project zip."
)
