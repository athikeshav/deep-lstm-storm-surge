"""03 - Time alignment (hourly grid) + feature engineering."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.align import align
from src.data.features import build_features
from src.utils import get_logger

log = get_logger("03_align_features")


if __name__ == "__main__":
    align()
    build_features()
