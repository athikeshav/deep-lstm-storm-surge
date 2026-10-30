"""05 - Train the Deep LSTM + linear baseline together."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.train import train
from src.utils import get_logger

log = get_logger("05_train_lstm")


if __name__ == "__main__":
    out = train()
    log.info(f"Done. Model saved at {out['model_path']}")
