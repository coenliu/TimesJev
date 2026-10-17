"""The interface every model backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from timesjev.schema import Prediction, Sample


class DecisionModel(ABC):
    """Maps Samples to Predictions. One instance per run."""

    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def predict(self, samples: list[Sample]) -> list[Prediction]:
        """Return one Prediction per sample, in order; a failed or unparseable answer gives answer=None so that SR
        counts it."""
