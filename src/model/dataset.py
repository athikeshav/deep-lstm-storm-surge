"""Stage 5 - Window creation.

Turns the tidy hourly table into supervised-learning tensors:
    X : shape (n_samples, lookback_steps, n_features)
    y : shape (n_samples,)            # surge at t + forecast_horizon

Chronological split is enforced; no random shuffling across time. The
target and feature scalers are fitted on the *training* period only and
persisted as joblib artifacts so inference can apply the exact same
transformations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DIRS, SPLITS, TIME, all_input_columns, target_column
from src.utils import get_logger, trace

log = get_logger(__name__)


@dataclass
class WindowedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    train_idx: pd.DatetimeIndex
    val_idx:   pd.DatetimeIndex
    test_idx:  pd.DatetimeIndex
    feature_cols: list[str]
    target_col: str
    lookback: int
    horizon: int


def build_windows() -> WindowedData:
    in_path = DIRS["data_proc"] / "features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Run feature engineering first; expected {in_path}")
    df = pd.read_parquet(in_path)

    feat_cols = all_input_columns()
    tgt_col   = target_column()
    lookback  = TIME["lookback_steps"]
    horizon   = TIME["forecast_horizon_steps"]

    # Keep only complete rows (every feature and target known)
    needed = feat_cols + [tgt_col]
    df_clean = df[needed].dropna()

    # Chronological splits
    train_mask = (df_clean.index >= pd.Timestamp(SPLITS["train_start"]).tz_localize("UTC")) & \
                 (df_clean.index <= pd.Timestamp(SPLITS["train_end"]).tz_localize("UTC"))
    val_mask   = (df_clean.index >= pd.Timestamp(SPLITS["val_start"]).tz_localize("UTC")) & \
                 (df_clean.index <= pd.Timestamp(SPLITS["val_end"]).tz_localize("UTC"))
    test_mask  = (df_clean.index >= pd.Timestamp(SPLITS["test_start"]).tz_localize("UTC")) & \
                 (df_clean.index <= pd.Timestamp(SPLITS["test_end"]).tz_localize("UTC"))

    train_df = df_clean[train_mask]
    val_df   = df_clean[val_mask]
    test_df  = df_clean[test_mask]

    if len(train_df) < lookback + horizon:
        raise RuntimeError(f"Not enough training rows ({len(train_df)}) for "
                           f"lookback={lookback} + horizon={horizon}")

    # Fit scalers on TRAIN ONLY
    feat_mean = train_df[feat_cols].mean()
    feat_std  = train_df[feat_cols].std().replace(0, 1.0)
    tgt_mean  = float(train_df[tgt_col].mean())
    tgt_std   = float(train_df[tgt_col].std()) or 1.0

    scaler_path = DIRS["models"] / "scalers.joblib"
    joblib.dump({
        "feat_cols": feat_cols, "tgt_col": tgt_col,
        "feat_mean": feat_mean.to_dict(),
        "feat_std": feat_std.to_dict(),
        "tgt_mean": float(tgt_mean), "tgt_std": float(tgt_std),
        "lookback": lookback, "horizon": horizon,
    }, scaler_path)

    def _to_tensor(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        feats = (d[feat_cols] - feat_mean) / feat_std
        tgt   = (d[tgt_col] - tgt_mean) / tgt_std
        feats_arr = feats.values.astype(np.float32)
        tgt_arr   = tgt.values.astype(np.float32)
        X, y, ts = [], [], []
        for i in range(len(d) - lookback - horizon + 1):
            X.append(feats_arr[i:i + lookback])
            y.append(tgt_arr[i + lookback + horizon - 1])
            ts.append(d.index[i + lookback + horizon - 1])
        return np.array(X), np.array(y), pd.DatetimeIndex(ts)

    X_train, y_train, train_idx = _to_tensor(train_df)
    X_val,   y_val,   val_idx   = _to_tensor(val_df)
    X_test,  y_test,  test_idx  = _to_tensor(test_df)

    wd = WindowedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        feature_cols=feat_cols, target_col=tgt_col,
        lookback=lookback, horizon=horizon,
    )

    # Persist tensors (small enough) for the dashboard / evaluation
    np.savez(DIRS["models"] / "windows.npz",
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             train_idx=train_idx.values.astype("datetime64[s]"),
             val_idx=val_idx.values.astype("datetime64[s]"),
             test_idx=test_idx.values.astype("datetime64[s]"))

    trace("windows", train=X_train.shape, val=X_val.shape, test=X_test.shape,
          feat_cols=feat_cols, tgt_col=tgt_col, scaler=str(scaler_path))
    log.info(f"Windows built: train={X_train.shape} val={X_val.shape} "
             f"test={X_test.shape}  (lookback={lookback} horizon={horizon})")
    return wd


def load_scalers() -> dict:
    return joblib.load(DIRS["models"] / "scalers.joblib")


if __name__ == "__main__":
    build_windows()
