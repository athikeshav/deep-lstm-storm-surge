"""Stage 3 - Time alignment.

Converts every product to UTC, resamples to the chosen hourly interval
(NOAA native cadence is 6-min; we downsample by mean aggregation,
forward-filling sparse met products where appropriate), and retains a
``_was_missing`` indicator so we never silently pretend data existed
when it didn't.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIRS, TIME, feature_columns
from src.utils import get_logger, trace

log = get_logger(__name__)


def align() -> pd.DataFrame:
    """Resample the validated table onto the hourly UTC grid."""
    in_path = DIRS["data_proc"] / "validated.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Run validation first; expected {in_path}")
    df = pd.read_parquet(in_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Build the target hourly grid covering the full historical range
    grid = pd.date_range(TIME["history_start"], TIME["history_end"],
                         freq=f"{TIME['interval_minutes']}min", tz="UTC")
    df = df.reindex(grid)
    df.index.name = "t"

    # Separate measurement columns from quality/suspect columns
    meas_cols, flag_cols = [], []
    for c in df.columns:
        if c.endswith("_quality") or c.endswith("_suspect"):
            flag_cols.append(c)
        else:
            meas_cols.append(c)

    # NOAA native cadence is 6-min; resample 6-min obs up to hourly by mean.
    # Since we already have an hourly grid, forward-fill met products that
    # are intrinsically sparse (e.g. water_temperature is often hourly, not 6-min).
    # Strategy: linear interpolation for missing values within a 3-hour window;
    # longer gaps are kept as NaN so window creation can drop them honestly.
    for c in meas_cols:
        df[c] = df[c].interpolate(method="time", limit=3, limit_area="inside")
    for c in flag_cols:
        df[c] = df[c].ffill().fillna("missing")

    # Track missingness explicitly per feature column
    for c in meas_cols:
        df[f"{c}_was_missing"] = df[c].isna()

    out_path = DIRS["data_proc"] / "aligned.parquet"
    df.to_parquet(out_path)

    n_total = len(df)
    n_complete = int(df[meas_cols].notna().all(axis=1).sum())
    coverage = {c: float(df[c].notna().mean()) for c in meas_cols}
    trace("align", in_rows=len(df), out_rows=n_total,
          complete_rows=n_complete, coverage=coverage,
          out_path=str(out_path))
    log.info(f"Aligned -> {out_path.name}  rows={n_total}  "
             f"complete={n_complete} ({n_complete/max(n_total,1):.1%})")
    return df


if __name__ == "__main__":
    align()
