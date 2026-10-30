"""Deep LSTM model definition.

Architecture follows the scope's recommended first attempt:
    Input [batch, lookback, features]
        -> LSTM(64, return_sequences=True) -> Dropout(0.2)
        -> LSTM(32, return_sequences=False) -> Dropout(0.2)
        -> Dense(16, relu)
        -> Dense(1, linear)

These are starting values; they must earn their complexity on held-out data.
"""
from __future__ import annotations

import os
# Quieten TF oneDNN logs for cleaner stdout
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import MODEL_CFG
from src.utils import get_logger, trace

log = get_logger(__name__)


def build_model(n_features: int, lookback: int) -> tf.keras.Model:
    cfg = MODEL_CFG
    inp = layers.Input(shape=(lookback, n_features), name="input_window")
    x = inp
    for layer_cfg in cfg["layers"]:
        t = layer_cfg["type"]
        if t == "lstm":
            x = layers.LSTM(layer_cfg["units"],
                            return_sequences=layer_cfg["return_sequences"],
                            name=layer_cfg.get("name", f"lstm_{t}"))(x)
        elif t == "dropout":
            x = layers.Dropout(layer_cfg["rate"])(x)
        elif t == "dense":
            x = layers.Dense(layer_cfg["units"],
                             activation=layer_cfg.get("activation", "linear"))(x)
        else:
            raise ValueError(f"Unknown layer type: {t}")
    out = x   # final Dense(1) is the linear output

    model = models.Model(inp, out, name="deep_lstm_storm_surge")
    tc = cfg["training"]
    opt = tf.keras.optimizers.Adam(learning_rate=tc["learning_rate"])
    loss = tc["loss"]
    if loss == "huber":
        loss_obj = tf.keras.losses.Huber()
    else:
        loss_obj = loss
    model.compile(optimizer=opt, loss=loss_obj,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    log.info(f"Built model with {model.count_params():,} params; "
             f"input_shape=({lookback},{n_features})")
    trace("model_build", n_params=int(model.count_params()),
          n_features=n_features, lookback=lookback, loss=tc["loss"])
    return model


if __name__ == "__main__":
    m = build_model(n_features=15, lookback=24)
    m.summary()
