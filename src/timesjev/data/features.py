"""Named, human-readable features of a numeric series, following the input statistics of Time-LLM's Prompt-as-Prefix
(Jin et al., 2024)."""

from __future__ import annotations

import numpy as np


def top_lags(x: np.ndarray, k: int = 5) -> list[int]:
    """The k lags (1 .. len // 2) with the highest autocorrelation, computed with the FFT as in Time-LLM."""
    x = np.asarray(x, float) - np.nanmean(x)
    x = np.nan_to_num(x)
    n = len(x)
    if n < 3:
        return []
    f = np.fft.rfft(x, n=2 * n)                                   # zero-padded: linear, not circular, correlation
    corr = np.fft.irfft(f * np.conj(f))[: n // 2 + 1]
    order = np.argsort(-corr[1:], kind="stable") + 1
    return [int(v) for v in order[:k]]


def describe_series(values: list[float]) -> dict:
    """Named features of one series: min, max, median, trend (sum of first differences > 0) and top-5 lags."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"length": 0}
    return {"length": int(len(x)), "min": float(x.min()), "max": float(x.max()), "median": float(np.median(x)),
            "trend": "upward" if np.diff(x).sum() > 0 else "downward", "top_lags": top_lags(x)}


def _num(v: float) -> str:
    return f"{v:.4g}"


def dominant_period(x) -> int | None:
    """Period (in samples) of the strongest frequency of the mean-removed series, from the FFT; None if too short."""
    x = np.nan_to_num(np.asarray(x, float) - np.nanmean(x))
    n = len(x)
    if n < 4:
        return None
    power = np.abs(np.fft.rfft(x))[1:]
    k = int(np.argmax(power)) + 1
    return int(round(n / k))


def cycle_extremes(x, period: int, k: int = 3) -> tuple[list[int], list[int]] | None:
    """Positions (0-based) of the k highest and k lowest points of the mean profile over full cycles of `period`
    points (the last full cycles), or None if there are fewer than 2 cycles."""
    x = np.asarray([v for v in x if v == v], float)
    if not period or period < 3 or len(x) < 2 * period:
        return None
    prof = x[len(x) - (len(x) // period) * period:].reshape(-1, period).mean(0)
    order = np.argsort(prof, kind="stable")
    return [int(i) for i in order[::-1][:k]], [int(i) for i in order[:k]]
