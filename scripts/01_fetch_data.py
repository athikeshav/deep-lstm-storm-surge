"""01 - Fetch raw data from NOAA CO-OPS API (or synthetic fallback).

Usage:
    python scripts/01_fetch_data.py            # try NOAA first
    python scripts/01_fetch_data.py --synthetic  # force synthetic
    python scripts/01_fetch_data.py --begin 2019-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import fetch_noaa, fetch_synthetic
from src.utils import get_logger, reset_trace

log = get_logger("01_fetch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="skip NOAA and use the offline synthetic generator")
    ap.add_argument("--begin", default=None)
    ap.add_argument("--end",   default=None)
    args = ap.parse_args()

    reset_trace()

    if args.synthetic:
        fetch_synthetic(args.begin, args.end)
        return 0

    try:
        result = fetch_noaa(args.begin, args.end)
        if result["n_ok"] == 0:
            log.warning("NOAA fetch returned 0 products; falling back to synthetic.")
            fetch_synthetic(args.begin, args.end)
    except Exception as e:
        log.error(f"NOAA fetch failed: {e}; falling back to synthetic.")
        fetch_synthetic(args.begin, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
