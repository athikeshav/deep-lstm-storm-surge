"""Stage 7 - Evaluation.

Runs the trained LSTM on the held-out test period, runs both baselines,
computes metrics (ordinary + peak events + interval coverage), and
produces the plots and tables required by Section 7 of the scope:
time-series plot, predicted-vs-observed scatter, residual distribution,
and baseline comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Font setup so plots render cleanly even on systems missing default fonts
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.config import DIRS, UNCERTAINTY, feature_columns, target_column
from src.evaluate.metrics import (MetricBundle, compute_metrics,
                                   baseline_skill_score, event_wise_table)
from src.model.baseline import linear_predict, persistence_predict_from_features
from src.model.dataset import load_scalers
from src.utils import get_logger, trace

log = get_logger(__name__)


def load_artifacts():
    """Load trained model, scalers, and windowed test set."""
    wd_path = DIRS["models"] / "windows.npz"
    model_path = DIRS["models"] / "lstm_model.keras"
    if not wd_path.exists() or not model_path.exists():
        raise FileNotFoundError("Model/windows not found - run training first")
    npz = np.load(wd_path, allow_pickle=True)
    X_test = npz["X_test"]; y_test = npz["y_test"]
    test_idx = pd.to_datetime(npz["test_idx"])
    scalers = load_scalers()
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, scalers, X_test, y_test, test_idx


def predict_with_uncertainty(model, X: np.ndarray, scalers: dict) -> dict:
    """Monte-Carlo dropout: run inference N times with dropout enabled,
    then derive mean and prediction intervals from the ensemble spread."""
    n = UNCERTAINTY["num_samples"]
    # Enable dropout at inference by calling the model with training=True
    preds = np.stack([model(X, training=True).numpy().ravel() for _ in range(n)], axis=0)
    # Convert back to physical units
    mu = preds.mean(axis=0)
    tgt_mean = scalers["tgt_mean"]; tgt_std = scalers["tgt_std"]
    mu_phys  = mu * tgt_std + tgt_mean
    lo80 = np.percentile(preds, 10, axis=0) * tgt_std + tgt_mean
    hi80 = np.percentile(preds, 90, axis=0) * tgt_std + tgt_mean
    lo95 = np.percentile(preds,  2.5, axis=0) * tgt_std + tgt_mean
    hi95 = np.percentile(preds, 97.5, axis=0) * tgt_std + tgt_mean
    return {"mean_scaled": mu, "mean_phys": mu_phys,
            "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95}


def evaluate() -> dict:
    model, scalers, X_test, y_test, test_idx = load_artifacts()
    tgt_mean = scalers["tgt_mean"]; tgt_std = scalers["tgt_std"]

    # ----- LSTM point forecast -----
    y_pred_lstm_scaled = model.predict(X_test, verbose=0).ravel()

    # ----- LSTM + uncertainty -----
    unc = predict_with_uncertainty(model, X_test, scalers)
    y_pred_lstm_scaled = unc["mean_scaled"]
    lo80_s = (unc["lo80"] - tgt_mean) / tgt_std
    hi80_s = (unc["hi80"] - tgt_mean) / tgt_std
    lo95_s = (unc["lo95"] - tgt_mean) / tgt_std
    hi95_s = (unc["hi95"] - tgt_mean) / tgt_std

    # ----- Baselines -----
    # Persistence: the last scaled surge value in each window.
    # We need to know which feature column corresponds to surge. The
    # surge residual is engineered as the TARGET, but we add surge_lag1_m
    # as a feature; persistence = surge at t (which equals surge_lag1_m at t+1).
    # Cleaner approach: we'll use the *predicted_tide*-removed water level
    # at the last window step. Since water_level_m and predicted_tide_m are
    # both features, we reconstruct surge_at_t = wl_lag - tide_lag from the
    # last window step.
    feat_cols = scalers["feat_cols"]
    wl_idx  = feat_cols.index("water_level_m")
    tide_idx = feat_cols.index("predicted_tide_m")
    last_wl_scaled   = X_test[:, -1, wl_idx]
    last_tide_scaled = X_test[:, -1, tide_idx]
    persistence_scaled = (last_wl_scaled * scalers["feat_std"]["water_level_m"]
                           + scalers["feat_mean"]["water_level_m"]) \
                          - (last_tide_scaled * scalers["feat_std"]["predicted_tide_m"]
                              + scalers["feat_mean"]["predicted_tide_m"])
    # Re-standardize persistence into target space using TARGET scaler
    persistence_scaled = (persistence_scaled - tgt_mean) / tgt_std

    y_pred_lin_scaled = linear_predict(X_test)

    # ----- Metrics -----
    m_lstm = compute_metrics(y_test, y_pred_lstm_scaled, tgt_mean, tgt_std,
                             y_lo_80=lo80_s, y_hi_80=hi80_s,
                             y_lo_95=lo95_s, y_hi_95=hi95_s)
    m_pers = compute_metrics(y_test, persistence_scaled, tgt_mean, tgt_std)
    m_lin = compute_metrics(y_test, y_pred_lin_scaled, tgt_mean, tgt_std)

    skill_vs_pers = baseline_skill_score(m_lstm.mae, m_pers.mae)
    skill_vs_lin = baseline_skill_score(m_lstm.mae, m_lin.mae)

    # ----- Plots -----
    out_dir = DIRS["reports"]; out_dir.mkdir(parents=True, exist_ok=True)
    _plot_timeseries(test_idx, y_test, y_pred_lstm_scaled,
                     lo80_s, hi80_s, lo95_s, hi95_s,
                     tgt_mean, tgt_std, out_dir / "timeseries.png")
    _plot_scatter(y_test, y_pred_lstm_scaled, tgt_mean, tgt_std,
                  out_dir / "scatter.png")
    _plot_residuals(y_test, y_pred_lstm_scaled, tgt_mean, tgt_std,
                    out_dir / "residuals.png")

    # ----- Event-wise table -----
    y_true_phys = y_test * tgt_std + tgt_mean
    y_pred_phys = y_pred_lstm_scaled * tgt_std + tgt_mean
    events = event_wise_table(test_idx, y_true_phys, y_pred_phys,
                              peak_percentile=95.0)
    events_path = out_dir / "peak_events.csv"
    events.to_csv(events_path, index=False)

    # ----- Save predictions parquet for the dashboard -----
    pred_df = pd.DataFrame({
        "t": test_idx,
        "y_true": y_true_phys,
        "lstm_pred": y_pred_phys,
        "lstm_lo80": unc["lo80"], "lstm_hi80": unc["hi80"],
        "lstm_lo95": unc["lo95"], "lstm_hi95": unc["hi95"],
        "persistence_pred": persistence_scaled * tgt_std + tgt_mean,
        "linear_pred": y_pred_lin_scaled * tgt_std + tgt_mean,
    }).set_index("t")
    pred_path = DIRS["data_proc"] / "test_predictions.parquet"
    pred_df.to_parquet(pred_path)

    # ----- Final report -----
    report = {
        "target": target_column(),
        "n_test_samples": int(len(y_test)),
        "test_period": [str(test_idx.min()), str(test_idx.max())],
        "lstm":   _metric_dict(m_lstm),
        "persistence": _metric_dict(m_pers),
        "linear_regression": _metric_dict(m_lin),
        "skill_score_vs_persistence": skill_vs_pers,
        "skill_score_vs_linear": skill_vs_lin,
        "n_peak_events": int(len(events)),
        "peak_threshold_metres": float(m_lstm.peak_threshold),
        "plots": {
            "timeseries": str(out_dir / "timeseries.png"),
            "scatter":    str(out_dir / "scatter.png"),
            "residuals":  str(out_dir / "residuals.png"),
        },
        "predictions_path": str(pred_path),
        "peak_events_path": str(events_path),
    }
    report_path = out_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    trace("evaluate", lstm_mae=m_lstm.mae, persistence_mae=m_pers.mae,
          linear_mae=m_lin.mae, skill_vs_pers=skill_vs_pers,
          skill_vs_lin=skill_vs_lin, n_peaks=m_lstm.n_peaks,
          cov80=m_lstm.coverage_80, cov95=m_lstm.coverage_95,
          report_path=str(report_path))
    log.info(f"Evaluation: LSTM MAE={m_lstm.mae:.4f}  Persistence MAE={m_pers.mae:.4f}  "
             f"Linear MAE={m_lin.mae:.4f}  skill_vs_pers={skill_vs_pers:+.1%}")
    return report


def _metric_dict(m: MetricBundle) -> dict:
    return {"mae": m.mae, "rmse": m.rmse, "bias": m.bias,
            "peak_mae": m.peak_mae, "n_samples": m.n_samples,
            "n_peaks": m.n_peaks, "peak_threshold": m.peak_threshold,
            "coverage_80": m.coverage_80, "coverage_95": m.coverage_95}


def _phys(y_scaled, mu, std): return y_scaled * std + mu


def _plot_timeseries(idx, y_true_s, y_pred_s, lo80_s, hi80_s, lo95_s, hi95_s,
                     mu, std, path: Path) -> None:
    y_true = _phys(y_true_s, mu, std)
    y_pred = _phys(y_pred_s, mu, std)
    lo95 = _phys(lo95_s, mu, std); hi95 = _phys(hi95_s, mu, std)
    lo80 = _phys(lo80_s, mu, std); hi80 = _phys(hi80_s, mu, std)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(idx, y_true, color="#1f3a5f", lw=1.0, label="Observed surge")
    ax.plot(idx, y_pred, color="#c44536", lw=1.0, label="LSTM forecast")
    ax.fill_between(idx, lo95, hi95, color="#c44536", alpha=0.12, label="95% PI")
    ax.fill_between(idx, lo80, hi80, color="#c44536", alpha=0.20, label="80% PI")
    ax.set_title("Held-out test period: observed vs LSTM-forecast storm surge (The Battery, NY)")
    ax.set_xlabel("Time"); ax.set_ylabel("Storm surge residual (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scatter(y_true_s, y_pred_s, mu, std, path: Path) -> None:
    y_true = _phys(y_true_s, mu, std)
    y_pred = _phys(y_pred_s, mu, std)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#1f3a5f")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#c44536", lw=1.0, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed surge (m)"); ax.set_ylabel("LSTM forecast (m)")
    ax.set_title("Predicted vs observed (test set)")
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_residuals(y_true_s, y_pred_s, mu, std, path: Path) -> None:
    y_true = _phys(y_true_s, mu, std)
    y_pred = _phys(y_pred_s, mu, std)
    err = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(err, bins=60, color="#1f3a5f", alpha=0.85)
    axes[0].axvline(0, color="#c44536", lw=1.5)
    axes[0].set_title(f"Residual distribution  (bias={err.mean():+.4f} m)")
    axes[0].set_xlabel("Forecast - observed (m)"); axes[0].set_ylabel("count")
    axes[1].scatter(y_true, err, s=4, alpha=0.3, color="#1f3a5f")
    axes[1].axhline(0, color="#c44536", lw=1.0)
    axes[1].set_xlabel("Observed surge (m)")
    axes[1].set_ylabel("Residual (m)")
    axes[1].set_title("Residuals vs observed")
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
