"""Structured pipeline logger and artifact store.

Every pipeline stage writes a JSON-lines trace record so a reviewer can
inspect what happened: raw rows -> validated rows -> aligned steps ->
window tensor shape -> model version -> output. This is the "behind-the-scenes
trace" required by Section 5 of the scope document.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.config import DIRS, REPRO

LOG_FILE = DIRS["logs"] / "pipeline.log"
TRACE_FILE = DIRS["logs"] / "pipeline_trace.jsonl"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to both stderr and a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(REPRO["log_level"])
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Pipeline trace (JSON-lines)
# ---------------------------------------------------------------------------
_trace_records: list[dict[str, Any]] = []


def trace(stage: str, **fields: Any) -> None:
    """Append a trace record for the named pipeline stage."""
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "stage": stage,
        **fields,
    }
    _trace_records.append(rec)
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    # also echo to the regular logger
    get_logger("trace").info(f"{stage}: {fields}")


def get_trace() -> list[dict[str, Any]]:
    """Read the full trace from disk."""
    if not TRACE_FILE.exists():
        return []
    out = []
    with TRACE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reset_trace() -> None:
    """Wipe the trace file at the start of a fresh pipeline run."""
    if TRACE_FILE.exists():
        TRACE_FILE.unlink()
    _trace_records.clear()
