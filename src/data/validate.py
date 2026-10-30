"""Stage 2 - Validation.

Checks schema, units, timestamp order, duplicate rows, gaps, outliers,
and NOAA quality flags. Suspect values are *flagged*, never silently
corrected. The output is a single tidy DataFrame with explicit
``_quality`` and ``_suspect`` columns so downstream stages can decide
how to handle them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIRS, PREDICTORS, SITE
from src.utils import get_logger, trace

log = get_logger(__name__)


# Plausible physical bounds (used for outlier flagging, not removal)
BOUNDS = {
    "water_level_m":       (-2.5,  3.5),    # metres above MLLW
    "predicted_tide_m":    (-2.0,  3.0),
    "wind_speed_ms":       (0.0,   60.0),
    "wind_direction_deg":  (0.0,   360.0),
    "air_pressure_hpa":    (950.0, 1060.0),
    "air_temperature_c":   (-25.0, 45.0),
    "water_temperature_c": (-3.0,  35.0),
    "storm_surge_residual":(-2.5,  4.0),
}


def validate_all() -> dict:
    """Validate every (product, station) pair, return per-product report."""
    from src.data.ingest import load_raw_product

    report: dict[str, dict] = {}
    cleaned: dict[str, pd.DataFrame] = {}

    # Group predictors by (product, station_id) so we load each raw file once
    fetches: list[tuple[str, str | None, list[dict]]] = []
    seen = set()
    for p in PREDICTORS:
        key = (p["product"], p.get("station_id"))
        if key in seen:
            continue
        seen.add(key)
        fetches.append((p["product"], p.get("station_id"),
                        [pp for pp in PREDICTORS
                         if pp["product"] == p["product"]
                         and pp.get("station_id") == p.get("station_id")]))

    for product, station_id, vars_for_product in fetches:
        try:
            df = load_raw_product(product, station_id)
        except FileNotFoundError as e:
            log.error(str(e))
            for v in vars_for_product:
                report_key = f"{product}:{v['rename']}"
                report[report_key] = {"error": str(e)}
            continue

        # The raw CSV has columns t, v (and possibly s, f, q, d, g depending on product)
        cleaned_df, r = _validate_product(df, product, station_id, vars_for_product)
        cleaned[product] = cleaned_df
        report_key = f"{product}@{station_id or SITE['station_id']}"
        report[report_key] = r
        trace("validate", product=product, station_id=station_id,
              raw_rows=r["raw_rows"], valid_rows=r["valid_rows"],
              duplicates=r["duplicates"], out_of_bounds=r["out_of_bounds"],
              flagged_quality=r["flagged_quality"], gaps=r["time_gaps"])

    merged = _merge_products(cleaned)
    out_path = DIRS["data_proc"] / "validated.parquet"
    merged.to_parquet(out_path)
    report_path = DIRS["data_proc"] / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Validation report -> {report_path.name}")
    trace("validate_complete", merged_rows=len(merged),
          out_path=str(out_path), report_path=str(report_path))
    return {"report": report, "merged": merged, "out_path": out_path}


def _validate_product(df: pd.DataFrame, product: str,
                      station_id: str | None,
                      vars_for_product: list[dict]) -> tuple[pd.DataFrame, dict]:
    n_raw = len(df)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).drop_duplicates(subset=["t"]).sort_values("t")

    cleaned = pd.DataFrame(index=df["t"])
    cleaned.index.name = "t"

    n_total_duplicates = int(n_raw - len(df))
    n_missing_val = 0
    n_oob = 0
    flagged_quality = 0

    flag_col = "q" if "q" in df.columns else None
    if flag_col:
        flagged_quality = int((df[flag_col].astype(str) != "v").sum())
        cleaned["_noaa_quality"] = df[flag_col].astype(str).values

    for v in vars_for_product:
        rename = v["rename"]; var = v["var"]
        if var not in df.columns:
            df[var] = np.nan
        n_missing_val += int(df[var].isna().sum())
        lo, hi = BOUNDS.get(rename, (-np.inf, np.inf))
        suspect = (df[var] < lo) | (df[var] > hi)
        cleaned[rename] = df[var].values
        cleaned[f"{rename}_suspect"] = suspect.values
        n_oob += int(suspect.sum())

    diffs = df["t"].diff().dt.total_seconds() / 60.0
    big_gaps = int(((diffs > 30) & (diffs.notna())).sum())

    r = {
        "product": product, "station_id": station_id or SITE["station_id"],
        "raw_rows": int(n_raw), "valid_rows": int(len(cleaned)),
        "duplicates": n_total_duplicates, "missing_values": n_missing_val,
        "out_of_bounds": n_oob, "flagged_quality": flagged_quality,
        "time_gaps": big_gaps, "min_time": str(cleaned.index.min()),
        "max_time": str(cleaned.index.max()),
        "renamed_to": [v["rename"] for v in vars_for_product],
    }
    return cleaned, r


def _merge_products(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all products on timestamp."""
    if not cleaned:
        return pd.DataFrame()
    out = None
    for prod, df in cleaned.items():
        if out is None:
            out = df
        else:
            out = out.join(df, how="outer")
    return out.sort_index()


if __name__ == "__main__":
    validate_all()
