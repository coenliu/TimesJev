"""Accuracy and calibration metrics: SR counts valid answers, ACC uses valid answers only, and acc_all counts invalid
answers as wrong."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from timesjev.schema import Prediction, Sample


# Probabilities are floored at 0.005 for the NLL, so one rounded-to-zero answer does not dominate the mean.
NLL_FLOOR = 0.005


def _true_prob(pred: Prediction, answer: str) -> float:
    return float(pred.probs.get(answer, 0.0))


def ece(confidences: list[float], correct: list[bool], n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    if not confidences:
        return float("nan")
    conf = np.asarray(confidences)
    hit = np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total / len(conf))


def choice_metrics(samples: list[Sample], preds: list[Prediction]) -> dict:
    valid = [(s, p) for s, p in zip(samples, preds) if p.valid]
    n = len(samples)
    correct = [p.answer == s.answer for s, p in valid]
    out = {
        "n": n,
        "sr": len(valid) / n if n else float("nan"),
        "acc": float(np.mean(correct)) if valid else float("nan"),
        "acc_all": sum(correct) / n if n else float("nan"),
    }

    scored = [(s, p) for s, p in valid if p.probs]
    if scored:
        confs = [max(p.probs.values()) for _, p in scored]
        hits = [max(p.probs, key=p.probs.get) == s.answer for s, p in scored]
        briers = [
            sum((p.probs.get(k, 0.0) - (k == s.answer)) ** 2 for k in s.options)
            for s, p in scored
        ]
        nlls = [-math.log(max(_true_prob(p, s.answer), NLL_FLOOR)) for s, p in scored]
        out.update({
            "ece": ece(confs, hits),
            "brier": float(np.mean(briers)),
            "nll": float(np.mean(nlls)),
        })
    return out


def latency_metrics(preds: list[Prediction]) -> dict:
    lat = [p.latency_s for p in preds if p.latency_s is not None]
    if not lat:
        return {}
    return {"latency_mean_s": float(np.mean(lat)), "latency_p50_s": float(np.median(lat))}


def compute_metrics(samples: list[Sample], preds: list[Prediction]) -> dict[str, dict[str, dict]]:
    """Return metrics[split][task] for every (split, task) present."""
    groups: dict[tuple[str, str], list[tuple[Sample, Prediction]]] = defaultdict(list)
    for s, p in zip(samples, preds):
        assert s.id == p.id, f"prediction order mismatch: {s.id} vs {p.id}"
        groups[(s.split, s.task)].append((s, p))

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (split, task), pairs in sorted(groups.items()):
        ss, ps = [s for s, _ in pairs], [p for _, p in pairs]
        out[split][task] = {**choice_metrics(ss, ps), **latency_metrics(ps)}
    return dict(out)
