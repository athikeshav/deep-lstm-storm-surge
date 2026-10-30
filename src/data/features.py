"""Stage 4 - Feature engineering.

Computes the storm-surge residual target and physically meaningful
derived features. Crucially, every feature here is *knowable at forecast
time*: no future information is used. Wind components, 3-hour pressure
change, lag-1 surge, and cyclical time encodings are all standard
additions that give the LSTM a better-conditioned input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DIRS, TIME, derived_feature_columns, feature_columns, target_column
from src.utils import get_logger, trace

log = get_logger(__name__)


def build_features() -> pd.DataFrame:
    in_path = DIRS["data_proc"] / "aligned.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Run alignment first; expected {in_path}")
    df = pd.read_parquet(in_path)

    # ----- target: storm-surge residual = observed WL - predicted tide -----
    df[target_column()] = df["water_level_m"] - df["predicted_tide_m"]

    # ----- derived features (only ones knowable at forecast time) -----
    # Wind vector components (more useful to a NN than raw direction)
    wdir_rad = np.deg2rad(df["wind_direction_deg"])
    df["wind_u_ms"] = df["wind_speed_ms"] * np.cos(wdir_rad)   # east-west
    df["wind_v_ms"] = df["wind_speed_ms"] * np.sin(wdir_rad)   # north-south

    # 3-hour pressure change (a proxy for storm approach rate)
    df["pressure_change_3h_hpa"] = df["air_pressure_hpa"].diff(periods=3)

    # Lag-1 features
    df["water_level_lag1_m"] = df["water_level_m"].shift(1)
    df["surge_lag1_m"]       = df[target_column()].shift(1)

    # Cyclical time encodings (give the LSTM the position in the day/year)
    hour = df.index.hour
    doy  = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    # Per-column missingness for derived features (driven by source missingness)
    for c in derived_feature_columns():
        if c in df.columns:
            df[f"{c}_was_missing"] = df[c].isna()

    out_path = DIRS["data_proc"] / "features.parquet"
    df.to_parquet(out_path)

    summary = {c: int(df[c].isna().sum()) for c in feature_columns() + [target_column()]}
    trace("features", rows=len(df),
          target=target_column(),
          n_features=len(derived_feature_columns()) + len(feature_columns()),
          missingness=summary, out_path=str(out_path))
    log.info(f"Features built -> {out_path.name}  "
             f"target={target_column()}  rows={len(df)}  "
             f"surge_min={df[target_column()].min():.3f}  "
             f"surge_max={df[target_column()].max():.3f}")
    return df


if __name__ == "__main__":
    build_features()
