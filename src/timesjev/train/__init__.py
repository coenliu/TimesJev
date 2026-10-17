"""Training methods. Each one takes a train config and returns the output checkpoint path."""

from __future__ import annotations

from importlib import import_module

METHODS = {
    "sft": "timesjev.train.sft:train_sft",
}


def run_training(cfg: dict):
    method = cfg["train"]["method"]
    if method not in METHODS:
        raise ValueError(f"unknown training method: {method}")
    module, fn = METHODS[method].split(":")
    return getattr(import_module(module), fn)(cfg)
