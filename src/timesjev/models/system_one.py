"""Shared request and response handling for System 1 backends: each choice sample becomes one Choice question whose
instruction is the question sentence, whose state is the context and whose criteria are the options."""

from __future__ import annotations

from timesjev.data.state import build_state
from timesjev.schema import Prediction, Sample

QUESTION_KEY = "answer"


def build_request(sample: Sample, state_kind: str = "raw") -> tuple[object, dict]:
    """Return (state, questions) for one choice sample."""
    question = {
        "type": "choice",
        "instructions": sample.question,
        "criteria": dict(sample.options),
    }
    return build_state(sample, state_kind), {QUESTION_KEY: question}


def parse_answer(sample: Sample, response: dict, latency_s: float | None = None) -> Prediction:
    """Turn a System 1 response into a Prediction."""
    answer = (response or {}).get("answers", {}).get(QUESTION_KEY) or {}
    choice = answer.get("choice")
    return Prediction(
        id=sample.id,
        task=sample.task,
        split=sample.split,
        answer=choice if choice in sample.options else None,
        probs=answer.get("probabilities"),
        latency_s=latency_s,
        extra={"confidence": answer.get("confidence")},
    )
