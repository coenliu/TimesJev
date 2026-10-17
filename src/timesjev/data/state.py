"""Build the System 1 `state` of a sample, the text the model reads; the config's `state` key picks the input format."""

from __future__ import annotations

from timesjev.data.features import _num
from timesjev.data.series import ARRAY
from timesjev.schema import Sample

VALUE_PLACEHOLDER = "<|vision_pad|>"


def raw_state(sample: Sample) -> str:
    """The context as-is: background plus the numeric arrays inline."""
    return sample.context


def no_series_state(sample: Sample) -> str:
    """The context with every numeric array removed."""
    return ARRAY.sub("[series omitted]", sample.context)


def selv_noarr_state(sample: Sample, key: str) -> str:
    """The context without the raw series plus the selected features (data/pool.py, selection <key>) in one line
    "Input statistics: <readable name> <value><placeholder>, ...", a value token after every value."""
    from timesjev.data.pool import readable, selected
    vals, names = selected(sample, key)
    if not vals:
        return no_series_state(sample)
    text = "Input statistics:\n" + ", ".join(f"{readable(n)} {_num(v)}{VALUE_PLACEHOLDER}" for n, v in zip(names, vals))
    return f"{no_series_state(sample)}\n\n{text}"


def build_state(sample: Sample, kind: str = "raw"):
    if kind.startswith("selv_noarr:"):                      # selected features, selection key after the colon
        return selv_noarr_state(sample, kind.split(":", 1)[1])
    if kind == "raw":
        return raw_state(sample)
    raise ValueError(f"unknown state kind: {kind}")
