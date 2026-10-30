"""Smoke tests for the major pipeline stages.

These tests verify that the pipeline components run end-to-end on a tiny
synthetic dataset without requiring network access. They are intended to
catch regressions in the public API of each module, not to assert
specific numerical values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable when run as a script
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def test_config_loads():
    from src.config import CONFIG, SITE, TIME, all_input_columns, target_column
    assert "site" in CONFIG
    assert SITE["station_id"] == "8518750"
    assert TIME["lookback_steps"] == 24
    assert TIME["forecast_horizon_steps"] == 3
    assert target_column() == "storm_surge_residual"
    assert "water_level_m" in all_input_columns()
    assert "wind_u_ms" in all_input_columns()  # derived feature


# -----------------------------------------------------------------------------
# Synthetic ingestion (offline)
# -----------------------------------------------------------------------------
def test_synthetic_ingest(tmp_path, monkeypatch):
    """fetch_synthetic should produce the expected file schema."""
    from src.config import DIRS
    from src.data.ingest import fetch_synthetic
    # Use a short window for speed
    result = fetch_synthetic("2023-01-01", "2023-02-28")
    assert result["n_ok"] > 0
    # Manifest should be written
    manifest_path = DIRS["data_raw"] / "manifest.json"
    assert manifest_path.exists()
    # At least one water_level CSV should exist
    files = list(DIRS["data_raw"].glob("synthetic_water_level.csv"))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert "t" in df.columns
    assert "v" in df.columns


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def test_validate_normalize_columns():
    """_normalize_columns should map NOAA headers to our tidy schema."""
    from src.data.ingest import _normalize_columns
    # Simulate a NOAA water_level CSV header
    df = pd.DataFrame({
        "Date Time": ["2023-01-01 00:00"],
        " Water Level": [1.234],
        " Sigma": [0.05],
        " O or I (for verified)": [0],
        " F": [0],
        " R": [0],
        " L": [0],
        " Quality ": ["v"],
    })
    out = _normalize_columns(df, "water_level")
    assert list(out.columns) == ["t", "v", "s", "f", "q"]
    assert out["v"].iloc[0] == 1.234


# -----------------------------------------------------------------------------
# Window creation
# -----------------------------------------------------------------------------
def test_build_windows_shapes():
    """build_windows should produce (n, lookback, n_features) tensors."""
    from src.model.dataset import build_windows
    wd = build_windows()
    assert wd.X_train.ndim == 3
    assert wd.X_train.shape[1] == 24
    assert wd.X_train.shape[2] == len(wd.feature_cols)
    assert wd.X_test.shape[1] == 24
    assert len(wd.y_train) == len(wd.X_train)
    assert len(wd.test_idx) == len(wd.X_test)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def test_metrics_compute():
    """compute_metrics should return finite values on a synthetic case."""
    from src.evaluate.metrics import compute_metrics, baseline_skill_score
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.normal(0, 0.2, n)
    y_pred = y_true + rng.normal(0, 0.05, n)   # near-perfect forecast
    m = compute_metrics(y_true, y_pred, tgt_mean=0.0, tgt_std=0.2)
    assert m.n_samples == n
    assert 0 <= m.mae < 0.1
    assert m.coverage_80 is None   # no intervals provided
    skill = baseline_skill_score(model_mae=0.05, baseline_mae=0.10)
    assert skill == 0.5


def test_event_wise_table():
    """event_wise_table should cluster peak hours into events."""
    from src.evaluate.metrics import event_wise_table
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-09-01", periods=500, freq="h")
    y_true = rng.normal(0, 0.2, 500)
    y_pred = y_true + rng.normal(0, 0.05, 500)
    # Inject two distinct peaks
    y_true[100] = 1.5; y_true[200] = 1.5; y_true[201] = 1.4; y_true[300] = 1.3
    tbl = event_wise_table(idx, y_true, y_pred, peak_percentile=95.0)
    assert len(tbl) >= 1
    assert "mae" in tbl.columns


# -----------------------------------------------------------------------------
# Replay forecast
# -----------------------------------------------------------------------------
def test_replay_forecast_returns_struct():
    """replay_forecast should return a ForecastResult with required fields."""
    from src.inference.forecast import replay_forecast
    # Pick a timestamp we know is in the test set
    r = replay_forecast("2023-09-29 18:00")
    assert r.forecast_value_m is not None
    assert r.status in ("demo forecast", "inputs stale", "inputs missing")
    assert len(r.pipeline_trace) >= 4   # ingest, validate, align, features, inference
    assert r.lookback_steps == 24
    assert r.forecast_horizon_steps == 3


# -----------------------------------------------------------------------------
# Logging trace
# -----------------------------------------------------------------------------
def test_trace_records_written():
    """get_trace should return the records written by trace()."""
    from src.utils import get_trace, trace
    n_before = len(get_trace())
    trace("test_stage", some_field="value", another=42)
    n_after = len(get_trace())
    assert n_after == n_before + 1


if __name__ == "__main__":
    # Allow running tests without pytest for environments without it
    import inspect
    mod = sys.modules[__name__]
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            print(f"  {name} ... ", end="")
            try:
                sig = inspect.signature(fn)
                if "tmp_path" in sig.parameters or "monkeypatch" in sig.parameters:
                    # Skip fixtures in non-pytest mode
                    print("skipped (needs pytest fixtures)")
                else:
                    fn()
                    print("OK")
            except Exception as e:
                print(f"FAIL: {e}")
