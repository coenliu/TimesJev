"""YAML config loading: `inherit: <path>` starts from another config, and a string ending in `.yaml` under `model` or
`train` is loaded in place. Paths are relative to the file that names them."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INLINE_KEYS = ("model", "train")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path) if not path.exists() else path.resolve()
    cfg = yaml.safe_load(path.read_text()) or {}

    if "inherit" in cfg:
        parent = load_config(path.parent / cfg.pop("inherit"))
        cfg = deep_merge(parent, cfg)

    for key in INLINE_KEYS:
        ref = cfg.get(key)
        if isinstance(ref, str) and ref.endswith(".yaml"):
            cfg[key] = load_config(PROJECT_ROOT / ref)
    return cfg
