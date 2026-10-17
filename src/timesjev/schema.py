"""Shared record types. Data loaders produce Samples, models produce Predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Sample:
    id: str                         # unique across sources, e.g. "tsqa/anomaly/123"
    task: str                       # "tsqa_<converter>"
    split: str                      # "train" / "val"
    context: str                    # background and data, without the question sentence or options
    question: str                   # the question sentence; used as the System 1 instruction
    options: dict[str, str]         # letter -> option text
    answer: str                     # option letter
    series: list[list[float]] = field(default_factory=list)  # numeric arrays in the context, in order
    domain: str = ""
    source: str = "tsqa"
    raw_problem: str = ""           # the original question text, unmodified


@dataclass
class Prediction:
    id: str
    task: str
    split: str
    answer: str | None                      # None when no valid answer was produced
    probs: dict[str, float] | None = None   # per-option probabilities, if the model gives them
    latency_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.answer is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
