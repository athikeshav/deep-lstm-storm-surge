"""Shared utility functions for time-series handling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame, col: str = "t") -> pd.DataFrame:
    """Convert a column named ``col`` to datetime and set as index."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=[col]).set_index(col).sort_index()
    return df


def regular_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="h", tz="UTC")


def safe_reindex(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex onto a regular hourly grid, tracking inserted NaNs explicitly."""
    out = df.reindex(idx)
    out["_was_missing"] = out.iloc[:, 0].isna() if len(out.columns) else False
    return out


def hours_since(ts: pd.Timestamp, ref: pd.Timestamp) -> float:
    return (ts - ref).total_seconds() / 3600.0
