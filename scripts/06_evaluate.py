"""06 - Evaluate on the held-out test period and produce plots + metrics."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate.evaluate import evaluate
from src.utils import get_logger

log = get_logger("06_evaluate")


if __name__ == "__main__":
    report = evaluate()
    log.info(f"Evaluation done. report -> {report['plots']}")
