"""Baselines that the LSTM must beat.

1. Persistence:    y_hat(t+H) = y(t)              (forecast = latest surge)
2. Linear regression: y_hat(t+H) = w . flatten(window) + b  (fit on train only)

A complex model that does not beat persistence is not a successful result.
"""
from __future__ import annotations

import joblib
import numpy as np

from src.config import DIRS, TIME
from src.utils import get_logger, trace

log = get_logger(__name__)


def persistence_predict(scaled_window_y: np.ndarray) -> np.ndarray:
    """Return the last observed (scaled) surge in each window as the forecast."""
    # In our windowing, the LAST step's value of the window is the most recent
    # surge observation. We don't actually need a model for this.
    return scaled_window_y


def fit_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """Fit a Ridge regression on flattened windows. Persisted as baseline.joblib."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n, L, F = X_train.shape
    flat = X_train.reshape(n, L * F)
    scaler = StandardScaler().fit(flat)
    flat_s = scaler.transform(flat)
    model = Ridge(alpha=1.0).fit(flat_s, y_train)

    joblib.dump({"scaler": scaler, "model": model, "L": L, "F": F,
                 "lookback": TIME["lookback_steps"],
                 "horizon": TIME["forecast_horizon_steps"]},
                DIRS["models"] / "baseline_linear.joblib")
    trace("baseline_fit", baseline="linear_ridge", n_train=n, L=L, F=F,
          path=str(DIRS["models"] / "baseline_linear.joblib"))
    log.info(f"Linear baseline trained: n_train={n}, L={L}, F={F}")
    return model


def linear_predict(X: np.ndarray) -> np.ndarray:
    art = joblib.load(DIRS["models"] / "baseline_linear.joblib")
    n = X.shape[0]
    flat = X.reshape(n, art["L"] * art["F"])
    flat_s = art["scaler"].transform(flat)
    return art["model"].predict(flat_s)


def persistence_predict_from_features(X: np.ndarray, scalers: dict,
                                      target_col_idx: int) -> np.ndarray:
    """Use the LAST timestep of the window for the persistence baseline.

    We need the scaled surge value at the last lookback step. The features
    matrix includes the surge column among its features, so we can read it
    directly from the window tensor (after standardization).
    """
    return X[:, -1, target_col_idx]
