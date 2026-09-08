"""Load immutable public method settings from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_method_config(config_root: str | Path, method: str) -> dict[str, Any]:
    path = Path(config_root) / f"{method}.toml"
    if not path.is_file():
        raise ValueError(f"unknown method {method!r}: missing {path}")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config.get("method") != method:
        raise ValueError(f"{path}: method must be {method!r}")
    return config
