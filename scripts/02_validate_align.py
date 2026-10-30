"""02 - Validate raw products (schema, gaps, outliers, quality flags) and
        merge into one tidy hourly table."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate import validate_all
from src.utils import get_logger

log = get_logger("02_validate")


if __name__ == "__main__":
    report = validate_all()
    log.info(f"Validated {len(report['report'])} products. "
             f"merged_rows={len(report['merged'])}")
