"""A pool of generic named statistics of a question's series: distribution, temporal, catch24 and stats3 statistics per
series or channel, and cross-channel correlations for interleaved series. A selection (configs/selections/<key>.json) is
rendered as "Input statistics:" text with a value token after every number."""

from __future__ import annotations

import functools
import json

import numpy as np

from timesjev.config import PROJECT_ROOT
from timesjev.data.features import cycle_extremes, describe_series, dominant_period
from timesjev.data.series import series_channels

SELECTIONS = PROJECT_ROOT / "configs" / "selections"

DN = ["mean", "std", "min", "max", "median", "q10", "q25", "q75", "q90", "IQR", "range", "skew", "kurtosis", "max robust z"]
TN = ["acf lag1", "acf lag2", "acf lag3", "acf lag4", "acf lag5", "mean abs step", "std abs step", "max abs step", "slope",
      "argmax position", "argmin position", "dominant period", "zero-crossing rate", "longest monotone run"]
CN = ["mode_5", "mode_10", "acf_timescale", "acf_first_min", "ami2", "trev", "high_fluctuation", "stretch_high",
      "transition_matrix", "periodicity", "embedding_dist", "ami_timescale", "whiten_timescale", "outlier_timing_pos",
      "outlier_timing_neg", "centroid_freq", "stretch_decreasing", "entropy_pairs", "rs_range", "dfa", "low_freq_power",
      "forecast_error", "c24 mean", "c24 SD"]
SN = ["length", "stats3 period", "stats3 min", "stats3 max", "stats3 median", "trend sign", "top lag 1", "top lag 2",
      "top lag 3", "top lag 4", "top lag 5", "peak pos 1", "peak pos 2", "peak pos 3", "low pos 1", "low pos 2", "low pos 3"]

# Readable names for the text (no digits, so a number in the text is always a value).
READABLE = {
    "q10": "tenth percentile", "q25": "lower quartile", "q75": "upper quartile", "q90": "ninetieth percentile",
    "IQR": "interquartile range", "max robust z": "max robust z-score",
    "acf lag1": "autocorrelation at lag one", "acf lag2": "autocorrelation at lag two",
    "acf lag3": "autocorrelation at lag three", "acf lag4": "autocorrelation at lag four",
    "acf lag5": "autocorrelation at lag five", "slope": "trend slope", "argmax position": "position of the maximum",
    "argmin position": "position of the minimum", "dominant period": "dominant period share",
    "mode_5": "histogram mode with five bins", "mode_10": "histogram mode with ten bins",
    "acf_timescale": "autocorrelation timescale", "acf_first_min": "first minimum of the autocorrelation",
    "ami2": "auto mutual information at lag two", "trev": "time-reversal asymmetry",
    "high_fluctuation": "share of large successive changes", "stretch_high": "longest stretch above the mean",
    "transition_matrix": "transition matrix covariance", "periodicity": "periodicity measure",
    "embedding_dist": "embedding distance", "ami_timescale": "auto mutual information timescale",
    "whiten_timescale": "whitened timescale", "outlier_timing_pos": "timing of positive outliers",
    "outlier_timing_neg": "timing of negative outliers", "centroid_freq": "spectral centroid frequency",
    "stretch_decreasing": "longest decreasing stretch", "entropy_pairs": "pattern entropy",
    "rs_range": "rescaled range fluctuation", "dfa": "detrended fluctuation", "low_freq_power": "low-frequency power",
    "forecast_error": "local forecast error", "c24 mean": "mean", "c24 SD": "standard deviation",
    "stats3 period": "dominant period", "stats3 min": "min", "stats3 max": "max", "stats3 median": "median",
    "top lag 1": "strongest lag", "top lag 2": "second strongest lag", "top lag 3": "third strongest lag",
    "top lag 4": "fourth strongest lag", "top lag 5": "fifth strongest lag",
    "peak pos 1": "highest position in the cycle", "peak pos 2": "second highest position in the cycle",
    "peak pos 3": "third highest position in the cycle", "low pos 1": "lowest position in the cycle",
    "low pos 2": "second lowest position in the cycle", "low pos 3": "third lowest position in the cycle",
    "length": "number of points",
}
_LAG_WORDS = {-4: "minus four", -3: "minus three", -2: "minus two", -1: "minus one", 0: "zero", 1: "one", 2: "two",
              3: "three", 4: "four"}


def readable(name: str) -> str:
    """Readable text name of a pool name ("Y-Z xcorr lag3" -> "Y-Z cross-correlation at lag three")."""
    if " xcorr lag" in name:
        pair, k = name.split(" xcorr lag")
        return f"{pair} cross-correlation at lag {_LAG_WORDS[int(k)]}"
    if name.endswith(" best lag"):
        return name[: -len(" best lag")] + " best cross-correlation lag"
    if name.endswith(" best corr"):
        return name[: -len(" best corr")] + " best cross-correlation"
    head, _, base = name.partition(" ")
    if head in ("X", "Y", "Z") and base:
        return f"{head} {READABLE.get(base, base)}"
    return READABLE.get(name, name)


def _dist(x):
    from scipy.stats import kurtosis, skew
    q = np.quantile(x, [0.1, 0.25, 0.75, 0.9])
    med = np.median(x)
    rz = (np.abs(x - med) / (np.median(np.abs(x - med)) + 1e-6)).max()
    return [x.mean(), x.std(), x.min(), x.max(), med, *q, q[2] - q[1], x.max() - x.min(),
            float(np.nan_to_num(skew(x))), float(np.nan_to_num(kurtosis(x))), min(rz, 1e3)]


def _temp(x):
    n = len(x)
    z = (x - x.mean()) / (x.std() + 1e-9)
    acf = [float(np.mean(z[:-k] * z[k:])) if n > k else 0.0 for k in range(1, 6)]
    d = np.abs(np.diff(x)) if n > 1 else np.zeros(1)
    slope = float(np.polyfit(np.arange(n), z, 1)[0]) * n if n > 1 else 0.0
    spec = np.abs(np.fft.rfft(z))[1:] if n > 3 else np.zeros(1)
    per = n / (int(np.argmax(spec)) + 1) / n if spec.any() else 0.0
    zc = float(np.mean(np.diff(np.sign(z)) != 0)) if n > 1 else 0.0
    s = np.sign(np.diff(x)) if n > 1 else np.zeros(1)
    run, best = 1, 1
    for k in range(1, len(s)):
        run = run + 1 if s[k] == s[k - 1] and s[k] != 0 else 1
        best = max(best, run)
    return [*acf, d.mean(), d.std(), d.max(), slope, np.argmax(x) / n, np.argmin(x) / n, per, zc, best / n]


def _c24(x):
    import pycatch22
    v = pycatch22.catch22_all(list(map(float, x)), catch24=True)["values"]
    return [0.0 if not np.isfinite(t) else t for t in v]


def _s3(x):
    d = describe_series(list(x))
    p = dominant_period(x)
    ext = cycle_extremes(list(x), p) if p else None
    lags = (d["top_lags"] + [-1] * 5)[:5]
    hi, lo = (ext if ext else ([-1] * 3, [-1] * 3))
    return [d["length"], p or -1, d["min"], d["max"], d["median"], 1 if d["trend"] == "upward" else -1, *lags,
            *(list(hi) + [-1] * 3)[:3], *(list(lo) + [-1] * 3)[:3]]


def _rel(cs):
    out = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            n = min(len(cs[i]), len(cs[j]))
            a, b = cs[i][:n], cs[j][:n]
            za, zb = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
            xc = [float(np.mean(za[:n - k] * zb[k:])) if k >= 0 else float(np.mean(za[-k:] * zb[:n + k]))
                  for k in range(-4, 5)] if n > 5 else [0.0] * 9
            out += xc + [float(np.argmax(xc) - 4), float(max(xc))]
    return out


def pool_row(sample) -> tuple[list[float], list[str]]:
    """(values, names) of the pool for the first array of a question (per channel for interleaved arrays)."""
    cs = []
    for v in sample.series[:1]:
        v = np.nan_to_num(np.asarray(v, float))
        c = series_channels(sample.task, list(v))
        cs += [v[j::c] for j in range(c)]
    vals, names = [], []
    multi = len(cs) > 1
    for ci, x in enumerate(cs):
        tag = f"{'XYZ'[ci]} " if multi else ""
        if len(x) < 5:                                   # catch22 (C code) crashes on empty or very short series
            x = np.resize(x, 5) if len(x) else np.zeros(5)
        for fn, nm in ((_dist, DN), (_temp, TN), (_c24, CN), (_s3, SN)):
            vals += [float(v) for v in fn(x)]
            names += [tag + n for n in nm]
    if multi:
        vals += _rel(cs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                p = f"{'XYZ'[i]}-{'XYZ'[j]}"
                names += [f"{p} xcorr lag{k}" for k in range(-4, 5)] + [f"{p} best lag", f"{p} best corr"]
    vals = list(np.nan_to_num(np.asarray(vals, float), posinf=1e6, neginf=-1e6))
    return vals, names


@functools.lru_cache(maxsize=64)
def selection(key: str) -> tuple[str, ...]:
    """The pool names of a selection (configs/selections/<key>.json: {"names": [...], ...})."""
    return tuple(json.loads((SELECTIONS / f"{key}.json").read_text())["names"])


def selected(sample, key: str) -> tuple[list[float], list[str]]:
    """(values, names) of the selected features, in selection order."""
    if not sample.series or not len(sample.series[0]):       # a question without a series has no statistics
        return [], []
    vals, names = pool_row(sample)
    sel = selection(key)
    pos = {n: i for i, n in enumerate(names)}
    return [float(vals[pos[n]]) for n in sel], list(sel)
