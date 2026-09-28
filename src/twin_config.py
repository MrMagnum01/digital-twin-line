"""Loads config.yaml, the single frozen numeric source of truth.

Every module (generator, ingest, features, evaluator) reads its frozen
numbers through load_config(); none of them hardcodes a duplicate.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    """Return the parsed config. Callers must treat it as read-only."""
    return _load(str(Path(path) if path is not None else CONFIG_PATH))
