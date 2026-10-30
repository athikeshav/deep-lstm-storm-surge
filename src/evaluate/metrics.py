"""Evaluation metrics for storm-surge forecasting.

Reports metrics in original physical units (metres), evaluates ordinary
and peak-event periods separately, and computes prediction-interval
coverage when uncertainty estimates are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricBundle:
    mae: float
    rmse: float
    bias: float
    peak_mae: float               # MAE on the top p% surge events
    n_samples: int
    n_peaks: int
    peak_threshold: float
    coverage_80: Optional[float]   # if intervals given
    coverage_95: Optional[float]


def _to_phys(y_scaled: np.ndarray, tgt_mean: float, tgt_std: float) -> np.ndarray:
    return y_scaled * tgt_std + tgt_mean


def compute_metrics(y_true_scaled: np.ndarray,
                    y_pred_scaled: np.ndarray,
                    tgt_mean: float,
                    tgt_std: float,
                    peak_percentile: float = 95.0,
                    y_lo_80: Optional[np.ndarray] = None,
                    y_hi_80: Optional[np.ndarray] = None,
                    y_lo_95: Optional[np.ndarray] = None,
                    y_hi_95: Optional[np.ndarray] = None) -> MetricBundle:
    """Compute MAE/RMSE/bias/peak-event MAE/coverage in physical units."""
    y_true = _to_phys(y_true_scaled, tgt_mean, tgt_std)
    y_pred = _to_phys(y_pred_scaled, tgt_mean, tgt_std)

    err = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # Peak-event error: top p% of true surge values
    thresh = float(np.percentile(y_true, peak_percentile))
    peak_mask = y_true >= thresh
    n_peaks = int(peak_mask.sum())
    peak_mae = float(np.mean(np.abs(err[peak_mask]))) if n_peaks > 0 else float("nan")

    cov80 = cov95 = None
    if y_lo_80 is not None and y_hi_80 is not None:
        lo = _to_phys(y_lo_80, tgt_mean, tgt_std)
        hi = _to_phys(y_hi_80, tgt_mean, tgt_std)
        cov80 = float(((y_true >= lo) & (y_true <= hi)).mean())
    if y_lo_95 is not None and y_hi_95 is not None:
        lo = _to_phys(y_lo_95, tgt_mean, tgt_std)
        hi = _to_phys(y_hi_95, tgt_mean, tgt_std)
        cov95 = float(((y_true >= lo) & (y_true <= hi)).mean())

    return MetricBundle(
        mae=mae, rmse=rmse, bias=bias, peak_mae=peak_mae,
        n_samples=len(y_true), n_peaks=n_peaks, peak_threshold=thresh,
        coverage_80=cov80, coverage_95=cov95,
    )


def baseline_skill_score(model_mae: float, baseline_mae: float) -> float:
    """Skill score: 1 - model_mae / baseline_mae.
    Positive => model beats baseline. Negative => worse than baseline.
    """
    if baseline_mae == 0:
        return float("nan")
    return float(1.0 - model_mae / baseline_mae)


def event_wise_table(timestamps: pd.DatetimeIndex,
                     y_true: np.ndarray, y_pred: np.ndarray,
                     peak_percentile: float = 95.0) -> pd.DataFrame:
    """Build a table of the top surge events with per-event MAE."""
    df = pd.DataFrame({"t": timestamps, "y_true": y_true, "y_pred": y_pred}).set_index("t")
    df["err"] = df["y_pred"] - df["y_true"]
    df["abs_err"] = df["err"].abs()
    thresh = float(np.percentile(df["y_true"], peak_percentile))
    peaks = df[df["y_true"] >= thresh].copy()
    if peaks.empty:
        return peaks
    # Group consecutive peak hours into events (gap > 6h => new event)
    peaks["event_id"] = (peaks.index.to_series().diff().dt.total_seconds() > 6 * 3600).cumsum()
    summary = peaks.groupby("event_id").agg(
        start=("y_true", "idxmin" if False else lambda s: s.index.min()),
        n_hours=("y_true", "size"),
        peak_true=("y_true", "max"),
        peak_pred=("y_pred", "max"),
        mae=("abs_err", "mean"),
    ).reset_index(drop=True)
    return summary
