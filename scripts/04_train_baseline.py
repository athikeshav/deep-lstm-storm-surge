"""04 - Train baselines (persistence + linear ridge)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.dataset import build_windows
from src.model.baseline import fit_linear_regression
from src.utils import get_logger

log = get_logger("04_baseline")


if __name__ == "__main__":
    wd = build_windows()
    fit_linear_regression(wd.X_train, wd.y_train)
    log.info("Baselines trained.")
