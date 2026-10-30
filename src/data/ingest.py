"""Stage 1 - Ingestion.

Fetches historical ocean/weather time series for the configured coastal
station(s) from the NOAA CO-OPS Data API:
    https://api.tidesandcurrents.noaa.gov/api/prod/

NOAA returns one CSV per (station, product, year). The verbose header
names NOAA uses (e.g. "Date Time", "Water Level", "Prediction") are
normalized to a tidy (t, v, s, f, q) schema on save so downstream
stages don't have to special-case them. A ``manifest.json`` records
source name, station, retrieval time, URL, and row count for every
artifact (Section 4 metadata + Section 5.1 ingestion).

If the network is unreachable, ``fetch_synthetic`` produces a physically
plausible surrogate series with the same tidy schema so the rest of the
pipeline can still be exercised for demonstration purposes.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import DIRS, PREDICTORS, SITE, TIME
from src.utils import get_logger, trace

log = get_logger(__name__)

API_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
APPLICATION = "storm_surge_research_project"
MAX_RETRIES = 4
RETRY_BACKOFF = 5  # seconds

# NOAA API limits per request (see config/config.yaml)
# Empirically: ALL products limited to 31 days per request.
# Only `predictions` accepts multi-year windows.
CHUNK_DAYS = {
    "water_level":      30,
    "predictions":      365,
    "wind":             30,
    "air_pressure":     30,
    "air_temperature":  30,
    "water_temperature":30,
}

# Normalised schema: NOAA's verbose CSV headers -> our tidy names.
# Every fetched CSV is renamed to these columns on save.
PRODUCT_COLUMNS = {
    # product: list of (noaa_header_substring, our_name)
    "water_level":     [("Date Time", "t"), ("Water Level", "v"),
                         ("Sigma", "s"), ("F", "f"), ("Quality", "q")],
    "predictions":     [("Date Time", "t"), ("Prediction", "v")],
    "wind":            [("Date Time", "t"), ("Speed", "s"),
                         ("Direction", "d"), ("Gust", "g")],
    "air_pressure":     [("Date Time", "t"), ("Pressure", "v")],
    "air_temperature":  [("Date Time", "t"), ("Air Temperature", "v")],
    "water_temperature":[("Date Time", "t"), ("Water Temperature", "v")],
}


def _normalize_columns(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Rename NOAA's verbose headers to our tidy schema for the given product."""
    rename_map: dict[str, str] = {}
    for header_substr, our_name in PRODUCT_COLUMNS.get(product, []):
        # Find columns whose name CONTAINS the substring (NOAA adds trailing spaces)
        for col in df.columns:
            if header_substr.lower() in col.lower() and col not in rename_map.values():
                rename_map[col] = our_name
                break
    df = df.rename(columns=rename_map)
    # Ensure the standard columns exist (fill missing with NaN)
    keep = [our_name for _, our_name in PRODUCT_COLUMNS.get(product, [])]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    return df[keep].copy()


# ---------------------------------------------------------------------------
# NOAA CO-OPS fetch
# ---------------------------------------------------------------------------
def _build_url(product: str, begin: str, end: str,
               station_id: str | None = None) -> str:
    begin_fmt = pd.Timestamp(begin).strftime("%Y%m%d %H:%M")
    end_fmt   = pd.Timestamp(end).strftime("%Y%m%d %H:%M")
    sid = station_id or SITE["station_id"]
    params = {
        "begin_date":   begin_fmt,
        "end_date":     end_fmt,
        "station":      sid,
        "product":      product,
        "units":        SITE["units"],
        "time_zone":    SITE["timezone"],
        "format":       "csv",
        "application":  APPLICATION,
    }
    # Datum only applies to water-level / tide products
    if product in ("water_level", "predictions"):
        params["datum"] = SITE["datum"]
    return API_BASE + "?" + "&".join(f"{k}={v.replace(' ', '%20')}" for k, v in params.items())


def _fetch_product(product: str, begin: str, end: str,
                   out_path: Path, station_id: str | None = None) -> pd.DataFrame | None:
    """Download one product for one date window and persist a normalised CSV."""
    url = _build_url(product, begin, end, station_id)
    sid = station_id or SITE["station_id"]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 400:
                # 400 is a permanent error (date range/format problem); don't retry.
                log.warning(f"NOAA HTTP 400 for product={product} station={sid}: {r.text[:120]}")
                return None
            if r.status_code != 200:
                log.warning(f"NOAA HTTP {r.status_code} for product={product} station={sid} "
                            f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_BACKOFF); continue
            text = r.text
            if text.lstrip().startswith("{"):
                err = json.loads(text)
                msg = err.get("error", err)
                # If it's a date-range error, don't retry
                if "Range Limit" in str(msg) or "Wrong Date" in str(msg):
                    log.warning(f"NOAA range-limit error product={product} station={sid}: {msg}")
                    return None
                log.warning(f"NOAA error product={product} station={sid}: {msg}")
                time.sleep(RETRY_BACKOFF); continue
            df = pd.read_csv(io.StringIO(text))
            if df.empty:
                log.warning(f"NOAA returned no rows for product={product} station={sid}")
                time.sleep(RETRY_BACKOFF); continue
            df = _normalize_columns(df, product)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            log.info(f"NOAA fetch OK: station={sid} product={product:20s} rows={len(df):>7} "
                     f"-> {out_path.name}")
            return df
        except requests.RequestException as e:
            log.warning(f"NOAA network error product={product} station={sid}: {e}")
            time.sleep(RETRY_BACKOFF)
    return None


def fetch_noaa(begin: str | None = None, end: str | None = None) -> dict:
    """Fetch every configured (product, station) pair from NOAA."""
    begin = begin or TIME["history_start"]
    end   = end   or TIME["history_end"]
    manifest: list[dict] = []

    # Build a list of unique (product, station_id) pairs to fetch
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

    for product, station_id, _vars in fetches:
        chunk_days = CHUNK_DAYS.get(product, 30)
        chunks = list(_chunk_dates(begin, end, chunk_days))
        for cb, ce in chunks:
            safe_begin = cb.strftime("%Y%m%d")
            safe_end   = ce.strftime("%Y%m%d")
            station_tag = station_id or SITE["station_id"]
            fname = f"{product}_{station_tag}_{safe_begin}_{safe_end}.csv"
            out_path = DIRS["data_raw"] / fname
            # Skip if already fetched successfully (resume support)
            if out_path.exists() and out_path.stat().st_size > 100:
                manifest.append({
                    "product": product, "station_id": station_tag,
                    "begin": str(cb), "end": str(ce), "status": "ok",
                    "rows": sum(1 for _ in open(out_path)) - 1,
                    "path": str(out_path.relative_to(DIRS["data_raw"].parent.parent)),
                    "url": _build_url(product, cb, ce, station_id),
                    "retrieved_at": "cached",
                    "source": "NOAA CO-OPS Data API (cached)",
                    "license": "Public domain (NOAA)",
                    "station_name": SITE["station_name"] if station_id is None else "Kings Point, NY",
                    "datum": SITE["datum"] if product in ("water_level", "predictions") else None,
                    "units": SITE["units"], "timezone": SITE["timezone"],
                })
                continue
            df = _fetch_product(product, cb.strftime("%Y-%m-%d"),
                                ce.strftime("%Y-%m-%d"), out_path, station_id)
            if df is None:
                manifest.append({"product": product, "station_id": station_tag,
                                 "begin": str(cb), "end": str(ce),
                                 "status": "failed", "rows": 0, "path": None,
                                 "url": _build_url(product, cb, ce, station_id)})
                continue
            manifest.append({
                "product": product, "station_id": station_tag,
                "begin": str(cb), "end": str(ce),
                "status": "ok", "rows": int(len(df)),
                "path": str(out_path.relative_to(DIRS["data_raw"].parent.parent)),
                "url": _build_url(product, cb, ce, station_id),
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "NOAA CO-OPS Data API",
                "license": "Public domain (NOAA)",
                "station_name": SITE["station_name"] if station_id is None else "Kings Point, NY",
                "datum": SITE["datum"] if product in ("water_level", "predictions") else None,
                "units": SITE["units"], "timezone": SITE["timezone"],
            })

    manifest_path = DIRS["data_raw"] / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"site": SITE, "begin": begin, "end": end,
                   "products": manifest}, f, indent=2, default=str)
    n_ok  = sum(1 for m in manifest if m["status"] == "ok")
    n_bad = sum(1 for m in manifest if m["status"] == "failed")
    trace("ingest_noaa", manifest_path=str(manifest_path),
          fetches_ok=n_ok, fetches_failed=n_bad, total=len(manifest))
    log.info(f"Ingestion complete: ok={n_ok} failed={n_bad} "
             f"manifest -> {manifest_path.name}")
    return {"manifest": manifest, "n_ok": n_ok, "n_failed": n_bad}


def _chunk_dates(begin: str, end: str, days: int) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cur = pd.Timestamp(begin)
    stop = pd.Timestamp(end)
    while cur <= stop:
        nxt = min(cur + pd.Timedelta(days=days - 1), stop)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Synthetic fallback (offline demonstration)
# ---------------------------------------------------------------------------
def fetch_synthetic(begin: str | None = None, end: str | None = None) -> dict:
    """Generate a physically plausible surrogate series.

    Used when the NOAA API is unreachable. The generator produces an
    astronomical tide (M2 + S2 + K1 + O1 constituents), a low-frequency
    weather-driven surge component with occasional storm spikes, plus
    correlated wind/pressure series. Schema matches the real NOAA CSV.
    """
    import numpy as np
    begin = begin or TIME["history_start"]
    end   = end   or TIME["history_end"]
    idx = pd.date_range(begin, end, freq="6min", tz="UTC")
    t_hours = (idx - idx[0]).total_seconds().values / 3600.0

    rng = np.random.default_rng(42)
    tide = (0.85 * np.sin(2 * np.pi * t_hours / 12.42)
            + 0.25 * np.sin(2 * np.pi * t_hours / 12.00)
            + 0.12 * np.sin(2 * np.pi * t_hours / 23.93)
            + 0.10 * np.sin(2 * np.pi * t_hours / 25.82))
    base = 0.05 * np.sin(2 * np.pi * t_hours / (24 * 14))
    noise = rng.normal(0, 0.03, size=len(idx))
    surge = np.zeros(len(idx))
    n_years = (t_hours[-1] / (24 * 365))
    n_storms = int(8 * n_years)
    for _ in range(n_storms):
        peak_t = rng.integers(48, len(idx) - 48)
        amp = rng.uniform(0.4, 1.2)
        dur = rng.integers(12, 36)
        w = np.exp(-((np.arange(len(idx)) - peak_t) ** 2) / (2 * dur ** 2))
        surge += amp * w
    residual = base + noise + surge
    water_level = tide + residual
    pressure = 1013.0 - 0.6 * surge + rng.normal(0, 2, size=len(idx))
    wind_speed = np.abs(rng.normal(5, 2, size=len(idx))) + 4 * np.sqrt(np.abs(surge) + 0.1)
    wind_dir = (180 + 60 * np.sin(2 * np.pi * t_hours / 36)
                + rng.normal(0, 30, size=len(idx))) % 360
    air_temp = 15 + 12 * np.sin(2 * np.pi * (t_hours / 24 - 80) / 365) + rng.normal(0, 1.5, len(idx))
    water_temp = 13 + 8 * np.sin(2 * np.pi * (t_hours / 24 - 100) / 365) + rng.normal(0, 0.6, len(idx))

    mask_keep = rng.random(len(idx)) > 0.03
    idx = idx[mask_keep]

    def _save(product: str, value: np.ndarray, value_col: str = "v") -> Path:
        out = DIRS["data_raw"] / f"synthetic_{product}.csv"
        df = pd.DataFrame({"t": idx.strftime("%Y-%m-%d %H:%M"), value_col: value[mask_keep]})
        if product == "water_level":
            df["s"] = 0.0; df["f"] = "0,0,0,0"; df["q"] = "p"
        df.to_csv(out, index=False)
        return out

    paths = {
        "water_level":       _save("water_level",     water_level,  "v"),
        "predictions":       _save("predictions",     tide,         "v"),
        "wind":              _save_wind(idx, mask_keep, wind_speed, wind_dir),
        "air_pressure":      _save("air_pressure",    pressure,    "v"),
        "air_temperature":   _save("air_temperature", air_temp,    "v"),
        "water_temperature": _save("water_temperature", water_temp, "v"),
    }
    manifest = []
    for product, p in paths.items():
        manifest.append({"product": product, "station_id": SITE["station_id"],
                         "begin": str(idx[0]), "end": str(idx[-1]),
                         "status": "ok", "rows": int(mask_keep.sum()),
                         "path": str(p.relative_to(DIRS["data_raw"].parent.parent)),
                         "url": None,
                         "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "source": "synthetic_generator_v1",
                         "license": "CC0 (project-generated)",
                         "station_name": SITE["station_name"],
                         "datum": SITE["datum"], "units": SITE["units"],
                         "timezone": SITE["timezone"]})
    manifest_path = DIRS["data_raw"] / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"site": SITE, "begin": begin, "end": end,
                   "products": manifest, "source": "synthetic"}, f, indent=2, default=str)
    trace("ingest_synthetic", manifest_path=str(manifest_path),
          rows=int(mask_keep.sum()), n_products=len(manifest))
    log.info(f"Synthetic ingestion complete: rows={int(mask_keep.sum())}")
    return {"manifest": manifest, "n_ok": len(manifest), "n_failed": 0}


def _save_wind(idx, mask_keep, wind_speed, wind_dir) -> Path:
    out = DIRS["data_raw"] / "synthetic_wind.csv"
    df = pd.DataFrame({
        "t": idx.strftime("%Y-%m-%d %H:%M"),
        "s": wind_speed[mask_keep], "d": wind_dir[mask_keep],
        "g": np.nan,
    })
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Loader used by the rest of the pipeline
# ---------------------------------------------------------------------------
def load_raw_product(product: str, station_id: str | None = None) -> pd.DataFrame:
    """Load every chunked CSV for a (product, station) and concatenate."""
    station_tag = station_id or SITE["station_id"]
    # First try real NOAA chunks: <product>_<station>_<dates>.csv
    pattern = f"{product}_{station_tag}_*.csv"
    files = sorted(DIRS["data_raw"].glob(pattern))
    if not files:
        # try synthetic naming
        files = sorted(DIRS["data_raw"].glob(f"synthetic_{product}.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files matching {pattern} or "
                                f"synthetic_{product}.csv in {DIRS['data_raw']}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out


if __name__ == "__main__":
    fetch_noaa()
