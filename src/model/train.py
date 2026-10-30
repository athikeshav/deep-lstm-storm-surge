"""Stage 6 - Training.

Trains the Deep LSTM on the chronological training period, monitors
validation loss, uses early stopping with best-weight restoration, and
saves the model + training history as inspectable artifacts.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from src.config import DIRS, MODEL_CFG, REPRO
from src.model.dataset import build_windows, load_scalers
from src.model.lstm import build_model
from src.model.baseline import fit_linear_regression
from src.utils import get_logger, save_environment, seed_everything, trace

log = get_logger(__name__)


def train() -> dict:
    seed_everything(REPRO["random_seed"])
    wd = build_windows()
    scalers = load_scalers()

    # ----- baseline: linear ridge on flattened windows -----
    fit_linear_regression(wd.X_train, wd.y_train)

    # ----- LSTM -----
    n_features = wd.X_train.shape[2]
    lookback   = wd.X_train.shape[1]
    model = build_model(n_features=n_features, lookback=lookback)

    tc = MODEL_CFG["training"]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=tc["early_stopping_patience"],
            restore_best_weights=tc["restore_best_weights"], verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.CSVLogger(DIRS["logs"] / "training_log.csv", append=False),
    ]

    t0 = time.time()
    history = model.fit(
        wd.X_train, wd.y_train,
        validation_data=(wd.X_val, wd.y_val),
        epochs=tc["epochs"], batch_size=tc["batch_size"],
        callbacks=callbacks, verbose=2,
        shuffle=False,                       # NEVER shuffle time series
    )
    train_secs = time.time() - t0

    # Save model + training history + environment snapshot
    model_path = DIRS["models"] / "lstm_model.keras"
    model.save(model_path)
    hist_path = DIRS["models"] / "training_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  f, indent=2)
    env_path = DIRS["reports"] / "environment.json"
    env = save_environment(env_path)

    final_epoch = {k: float(v[-1]) for k, v in history.history.items()}
    trace("train_complete", model_path=str(model_path),
          history_path=str(hist_path), env_path=str(env_path),
          train_seconds=round(train_secs, 1),
          epochs_run=len(history.history["loss"]),
          final_val_loss=final_epoch.get("val_loss"),
          final_val_mae=final_epoch.get("val_mae"))
    log.info(f"Training complete in {train_secs:.1f}s. "
             f"val_loss={final_epoch.get('val_loss', float('nan')):.4f} "
             f"val_mae={final_epoch.get('val_mae', float('nan')):.4f}")
    return {
        "model_path": str(model_path),
        "history_path": str(hist_path),
        "train_seconds": round(train_secs, 1),
        "epochs_run": len(history.history["loss"]),
        "final": final_epoch,
    }


if __name__ == "__main__":
    train()
