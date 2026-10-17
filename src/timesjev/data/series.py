"""Find numeric arrays in problem text."""

from __future__ import annotations

import math
import re

_NUM = r"-?\d[\d.eE+\-]*"
_MISSING = r"'X'|\"X\"|X"   # TSQA imputation marks missing values as 'X'
_ITEM = rf"(?:{_NUM}|{_MISSING})"
ARRAY = re.compile(rf"\[\s*{_ITEM}(?:\s*,\s*{_ITEM})*\s*\]")


def _to_float(token: str) -> float:
    token = token.strip().strip("'\"")
    return math.nan if token == "X" else float(token)


def extract_series(text: str) -> list[list[float]]:
    """Return every numeric array in the text, in order. Missing values ('X') become NaN."""
    series = []
    for match in ARRAY.finditer(text):
        try:
            series.append([_to_float(v) for v in match.group(0)[1:-1].split(",")])
        except ValueError:
            continue
    return series


# Activity recognition (TSQA classification) stores 10 timestamps of X, Y, Z acceleration as one interleaved
# array [x1, y1, z1, x2, ...]; every other array is a single channel.
CHANNEL_NAMES = {"tsqa_classification": ("X", "Y", "Z")}


def series_channels(task: str, values: list[float]) -> int:
    """Number of interleaved channels of one array of a sample of this task."""
    n = len(CHANNEL_NAMES.get(task, ("",)))
    return n if n > 1 and len(values) % n == 0 else 1
