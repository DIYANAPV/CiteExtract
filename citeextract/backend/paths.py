
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


def data_dir() -> Path:
    override = os.environ.get("CITEEXTRACT_DATA_DIR")
    if override:
        return Path(override)
    return _REPO_ROOT / "data"


def config_path() -> Path:
    return _REPO_ROOT / "config" / "config.yaml"


def env_path() -> Path:
    return _REPO_ROOT / ".env"
