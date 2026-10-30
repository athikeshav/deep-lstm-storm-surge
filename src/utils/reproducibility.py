"""Reproducibility helpers: seed everything, capture environment versions."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tensorflow import is heavy; defer until actually needed
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def capture_environment() -> dict:
    """Snapshot the package versions of every library we depend on."""
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn",
                "tensorflow", "matplotlib", "seaborn",
                "plotly", "fastapi", "streamlit", "yaml", "requests"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not_installed"
    import platform
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    return versions


def save_environment(path: Path) -> dict:
    env = capture_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, default=str)
    return env
