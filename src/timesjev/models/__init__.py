"""Model registry. Backends import their heavy dependencies lazily."""

from __future__ import annotations

from importlib import import_module

from timesjev.models.base import DecisionModel

BACKENDS = {
    "decider": "timesjev.models.decider_model:DeciderModel",
    "decider_ts": "timesjev.models.decider_ts:DeciderTSModel",
}


def build_model(cfg: dict) -> DecisionModel:
    backend = cfg["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    module, cls = BACKENDS[backend].split(":")
    return getattr(import_module(module), cls)(cfg)
