"""Central configuration loader.

Resolves paths relative to the project root, loads YAML config, and exposes
a singleton ``CONFIG`` for the rest of the package. No magic strings should
live outside this file and ``config/config.yaml``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../deep-lstm-storm-surge/

DIRS = {
    "config":     PROJECT_ROOT / "config",
    "data_raw":   PROJECT_ROOT / "data" / "raw",
    "data_proc":  PROJECT_ROOT / "data" / "processed",
    "data_ext":   PROJECT_ROOT / "data" / "external",
    "models":     PROJECT_ROOT / "models",
    "reports":    PROJECT_ROOT / "reports",
    "dashboard":  PROJECT_ROOT / "dashboard",
    "scripts":    PROJECT_ROOT / "scripts",
    "logs":       PROJECT_ROOT / "logs",
}

for _name, _p in DIRS.items():
    _p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG: dict[str, Any] = _load_yaml(DIRS["config"] / "config.yaml")


# Convenience accessors ------------------------------------------------------
SITE          = CONFIG["site"]
TARGET        = CONFIG["target"]
PREDICTORS    = CONFIG["predictors"]
TIME          = CONFIG["time"]
SPLITS        = CONFIG["splits"]
MODEL_CFG     = CONFIG["model"]
BASELINES_CFG = CONFIG["baselines"]
UNCERTAINTY   = CONFIG["uncertainty"]
REPRO         = CONFIG["reproducibility"]


def feature_columns() -> list[str]:
    """Return the ordered list of model input feature column names."""
    return [p["rename"] for p in PREDICTORS]


def target_column() -> str:
    return TARGET["variable"]


def derived_feature_columns() -> list[str]:
    """Engineered features produced by features.py (kept in sync manually)."""
    return [
        "wind_u_ms", "wind_v_ms", "pressure_change_3h_hpa",
        "water_level_lag1_m", "surge_lag1_m",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]


def all_input_columns() -> list[str]:
    """Every column fed to the LSTM (raw + engineered), in deterministic order."""
    cols = feature_columns() + derived_feature_columns()
    # de-duplicate while preserving order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def path(*parts: str) -> Path:
    """Join parts onto PROJECT_ROOT and ensure parent dirs exist."""
    p = PROJECT_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
