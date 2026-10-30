"""Stage 8 + 9 - Inference and presentation.

Given a historical timestamp, ``replay_forecast`` reconstructs the input
window from data knowable at that time only, runs the LSTM, attaches
prediction intervals, and returns a structured result plus a data-quality
status string ("demo forecast" / "inputs stale" / "inputs missing").

This is the engine behind the dashboard's "replay mode".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import DIRS, MODEL_CFG, TIME, UNCERTAINTY, all_input_columns, target_column
from src.model.dataset import load_scalers
from src.utils import get_logger, trace

log = get_logger(__name__)


@dataclass
class ForecastResult:
    forecast_time: str            # the time we're forecasting (t + horizon)
    last_input_time: str          # the last observation in the window (t)
    forecast_value_m: float       # predicted surge at t+H, in metres
    lo80: float
    hi80: float
    lo95: float
    hi95: float
    status: str                   # "demo forecast" | "inputs stale" | "inputs missing"
    model_version: str
    lookback_steps: int
    forecast_horizon_steps: int
    feature_cols: list
    n_features_missing: int
    data_quality_warnings: list
    pipeline_trace: list          # the behind-the-scenes trace


def _load_model_and_scalers():
    model_path = DIRS["models"] / "lstm_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run training first.")
    model = tf.keras.models.load_model(model_path, compile=False)
    scalers = load_scalers()
    return model, scalers


def _input_window_for_timestamp(df: pd.DataFrame, t: pd.Timestamp,
                                feat_cols: list, feat_mean, feat_std) -> tuple[Optional[np.ndarray], list]:
    """Build the standardized input window ending exactly at ``t``."""
    warnings_list = []
    if not isinstance(t, pd.Timestamp):
        t = pd.Timestamp(t)
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    end = t
    start = end - pd.Timedelta(hours=TIME["lookback_steps"] - 1)
    window_df = df.loc[start:end].copy()
    if len(window_df) != TIME["lookback_steps"]:
        return None, [f"only {len(window_df)} of {TIME['lookback_steps']} expected "
                      f"lookback rows available before {end}"]
    # check feature presence
    needed = [c for c in feat_cols if c not in window_df.columns]
    if needed:
        return None, [f"missing feature columns: {needed}"]

    n_missing = int(window_df[feat_cols].isna().any(axis=1).sum())
    if n_missing > 0:
        warnings_list.append(f"{n_missing}/{TIME['lookback_steps']} lookback rows have missing "
                             "feature values; forward-filled from earlier observations.")
        window_df[feat_cols] = window_df[feat_cols].ffill().bfill()

    standardized = (window_df[feat_cols] - feat_mean) / feat_std
    return standardized.values.astype(np.float32)[None, :, :], warnings_list


def replay_forecast(t_input_end: str | pd.Timestamp) -> ForecastResult:
    """Run a forecast using ONLY data knowable at or before ``t_input_end``."""
    model, scalers = _load_model_and_scalers()
    feat_cols = scalers["feat_cols"]
    feat_mean = pd.Series(scalers["feat_mean"])
    feat_std  = pd.Series(scalers["feat_std"])
    tgt_mean  = scalers["tgt_mean"]; tgt_std = scalers["tgt_std"]

    feats_path = DIRS["data_proc"] / "features.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(f"Run the pipeline up to features first; expected {feats_path}")
    df = pd.read_parquet(feats_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if not isinstance(t_input_end, pd.Timestamp):
        t_end = pd.Timestamp(t_input_end)
    else:
        t_end = t_input_end
    if t_end.tz is None:
        t_end = t_end.tz_localize("UTC")
    else:
        t_end = t_end.tz_convert("UTC")
    t_forecast = t_end + pd.Timedelta(hours=TIME["forecast_horizon_steps"])

    window, warnings_list = _input_window_for_timestamp(df, t_end, feat_cols, feat_mean, feat_std)
    if window is None:
        return ForecastResult(
            forecast_time=str(t_forecast), last_input_time=str(t_end),
            forecast_value_m=float("nan"),
            lo80=float("nan"), hi80=float("nan"),
            lo95=float("nan"), hi95=float("nan"),
            status="inputs missing", model_version=_model_version(),
            lookback_steps=TIME["lookback_steps"],
            forecast_horizon_steps=TIME["forecast_horizon_steps"],
            feature_cols=feat_cols, n_features_missing=len(warnings_list),
            data_quality_warnings=warnings_list, pipeline_trace=[],
        )

    # Run inference with MC dropout for uncertainty
    n = UNCERTAINTY["num_samples"]
    preds = np.stack([model(window, training=True).numpy().ravel()[0] for _ in range(n)])
    mu_phys = float(preds.mean()) * tgt_std + tgt_mean
    lo80 = float(np.percentile(preds, 10)) * tgt_std + tgt_mean
    hi80 = float(np.percentile(preds, 90)) * tgt_std + tgt_mean
    lo95 = float(np.percentile(preds, 2.5)) * tgt_std + tgt_mean
    hi95 = float(np.percentile(preds, 97.5)) * tgt_std + tgt_mean

    # Determine status: stale if last observation is >2h older than the
    # newest in the dataset; "demo forecast" otherwise.
    newest_in_data = df.index.max()
    age_hours = (newest_in_data - t_end).total_seconds() / 3600.0
    if age_hours > 2:
        status = "inputs stale"
        warnings_list.append(f"most recent input is {age_hours:.1f}h older than the "
                             "newest observation in the dataset")
    else:
        status = "demo forecast"

    # Build behind-the-scenes trace
    pipeline_trace = _build_trace(df, t_end, window, n_features_missing=len(warnings_list))

    trace("inference_replay", input_end=str(t_end), forecast_time=str(t_forecast),
          forecast_m=round(mu_phys, 4), status=status,
          lo80=round(lo80, 4), hi80=round(hi80, 4))

    return ForecastResult(
        forecast_time=str(t_forecast),
        last_input_time=str(t_end),
        forecast_value_m=mu_phys,
        lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
        status=status,
        model_version=_model_version(),
        lookback_steps=TIME["lookback_steps"],
        forecast_horizon_steps=TIME["forecast_horizon_steps"],
        feature_cols=feat_cols,
        n_features_missing=len(warnings_list),
        data_quality_warnings=warnings_list,
        pipeline_trace=pipeline_trace,
    )


def _model_version() -> str:
    """Read model version from src/config.py CONFIG.project.version."""
    from src.config import CONFIG
    return CONFIG["project"]["version"]


def _build_trace(df: pd.DataFrame, t_end: pd.Timestamp,
                window: np.ndarray, n_features_missing: int) -> list:
    raw_rows = int(len(df))
    # validated rows: drop NaN targets
    valid_rows = int(df[target_column()].notna().sum())
    # aligned rows: rows where all features known
    aligned_rows = int(df[all_input_columns()].notna().all(axis=1).sum())
    return [
        {"step": "ingestion",       "raw_rows": raw_rows},
        {"step": "validation",      "valid_rows": valid_rows},
        {"step": "time_alignment",  "aligned_rows": aligned_rows},
        {"step": "feature_engineering", "feature_tensor_shape": list(window.shape)},
        {"step": "inference",       "model_version": _model_version(),
                                    "input_end": str(t_end),
                                    "n_features_missing": n_features_missing},
    ]


def observed_at(t: str | pd.Timestamp) -> Optional[float]:
    """Return the actual observed surge at time ``t`` (for replay comparison)."""
    feats_path = DIRS["data_proc"] / "features.parquet"
    df = pd.read_parquet(feats_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    t = pd.Timestamp(t, tz="UTC")
    if t in df.index:
        return float(df.loc[t, target_column()])
    return None


if __name__ == "__main__":
    import sys
    ts = sys.argv[1] if len(sys.argv) > 1 else "2023-09-29 18:00"
    r = replay_forecast(ts)
    print(json.dumps(asdict(r), indent=2, default=str))
