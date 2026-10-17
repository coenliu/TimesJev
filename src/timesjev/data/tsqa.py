"""TSQA (Time-MQA) loading: anomaly detection, classification and open-ended QA rows become Choice Samples, and binary
tasks become 2-option choices. Rows a converter cannot parse are skipped and counted; the forecasting and imputation
files are not used."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from timesjev.config import PROJECT_ROOT
from timesjev.data.series import ARRAY, extract_series
from timesjev.schema import Sample

TSQA_DIR = PROJECT_ROOT / "data" / "TSQA"

FILES = {
    "anomaly_detection": ["Anomaly_Detection/anomaly_detection.csv"],
    "classification": ["Classification/classification.csv"],
    "open_ended_qa": ["Open_Ended_QA/open_ended_QA.csv"],
}


# ---------------------------------------------------------------- raw loading

_QA_PATTERN = re.compile(r'^\s*"question":\s*"(.*)",\s*"answer":\s*"(.*)"\s*$', re.S)


def normalize_task(name: str) -> str:
    """Open-ended QA labels mix spaces and underscores ("trend analysis" vs "trend_analysis")."""
    return str(name).strip().lower().replace(" ", "_")


def parse_qa(cell: str) -> dict[str, str] | None:
    """Parse one QA_list cell: non-strict JSON first, then a regex. None if neither works."""
    if not isinstance(cell, str):     # empty cell (pandas >= 3 keeps NaN through astype(str))
        return None
    try:
        return json.loads("{" + cell + "}", strict=False)
    except json.JSONDecodeError:
        match = _QA_PATTERN.match(cell)
        return {"question": match.group(1), "answer": match.group(2)} if match else None


def load_tsqa_raw(files: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    """One row per unique QA pair. Columns: row, file, domain, task, format, question, answer."""
    frames = []
    for key in files or list(FILES):
        for rel in FILES[key]:
            df = pd.read_csv(TSQA_DIR / rel, nrows=limit)
            qa = df["QA_list"].astype(str).map(parse_qa)
            if qa.isna().any():
                print(f"[tsqa] {rel}: dropped {int(qa.isna().sum())} unparseable rows")
            keep = qa.notna()
            df, qa = df[keep], qa[keep]
            frames.append(pd.DataFrame({
                "row": [f"{rel.split('/')[-1].removesuffix('.csv')}:{i}" for i in df.index],
                "file": key,
                "domain": df["application_domain"].values,
                "task": df["task_type"].map(normalize_task).values,
                "format": (df["question_format"].fillna("").str.lower().values
                           if "question_format" in df else ""),
                "question": qa.map(lambda d: d.get("question", "")).values,
                "answer": qa.map(lambda d: d.get("answer", "")).values,
            }))
    raw = pd.concat(frames, ignore_index=True)
    dupes = raw.duplicated(["question", "answer"])
    if dupes.any():
        print(f"[tsqa] dropped {int(dupes.sum())} duplicate QA pairs")
    return raw[~dupes].reset_index(drop=True)


# ----------------------------------------------------------------- converters

def _letters(labels: list[str]) -> dict[str, str]:
    return {chr(ord("A") + i): label for i, label in enumerate(labels)}


def _sample(row, converter: str, context: str, question: str,
            options: dict[str, str], answer: str) -> Sample:
    return Sample(
        id=f"tsqa/{converter}/{row.row}",
        task=f"tsqa_{converter}",
        split="train",
        context=context.strip(),
        question=question.strip(),
        options=options,
        answer=answer,
        series=extract_series(context),
        domain=row.domain,
        source="tsqa",
        raw_problem=row.question,
    )


_CLS_OPTIONS = re.compile(r"corresponds to (.+?)\.? given information above")
_CLS_ANSWER = re.compile(r"(?:activity|answer) is (.+?)\.?$")


def convert_classification(raw: pd.DataFrame, skipped: Counter) -> Iterator[Sample]:
    """Activity recognition (6-way) from 10 timestamps of X, Y, Z acceleration; the freeze-of-gait rows are skipped, as
    each holds a single timestamp."""
    for row in raw[raw["file"] == "classification"].itertuples():
        opts, ans = _CLS_OPTIONS.search(row.question), _CLS_ANSWER.search(row.answer.strip())
        if not (opts and ans):
            skipped["unparseable"] += 1
            continue
        labels = [x.strip(" '\".") for x in opts.group(1).split(" or ")]
        if set(labels) == {"No freeze", "Freeze"}:
            skipped["freeze_of_gait_not_a_series"] += 1
            continue
        answer = ans.group(1).strip(" '\"")
        if answer not in labels:
            skipped["unparseable"] += 1
            continue
        options = _letters(labels)
        yield _sample(row, "classification", row.question[:opts.start()].rsplit("Please", 1)[0],
                      "Which option does this data correspond to?", options,
                      next(k for k, v in options.items() if v == answer))


def convert_anomaly(raw: pd.DataFrame, skipped: Counter) -> Iterator[Sample]:
    """Binary anomaly detection as a 2-option choice."""
    options = {"A": "Normal: the series contains only normal points",
               "B": "Anomaly: the series contains anomaly points"}
    for row in raw[raw["file"] == "anomaly_detection"].itertuples():
        if "Normal Point" in row.answer:
            answer = "A"
        elif "Anomaly Point" in row.answer:
            answer = "B"
        else:
            skipped["unparseable"] += 1
            continue
        yield _sample(row, "anomaly", row.question.split("Please determine")[0],
                      "Are there anomalies in this time series?", options, answer)


# ------------------------------------------------------------ label checks
# MCQ and T/F labels are often arithmetically wrong; where the answer can be computed from the series, rows with a
# provably wrong label are dropped.

_FORECAST_TOPIC = re.compile(r"\b(forecast\w*|predict\w*|future|next (?:value|point|data point|number)s?)\b", re.I)
_NOT_SIMPLE_STAT = re.compile(r"\b(range|first|last|difference|change|moving|rolling|period|"
                              r"standard|deviation|variance|median|between|rate|absolute|window)\b", re.I)
_STAT_QUESTION = {
    "mean": re.compile(r"\b(mean|average)\b", re.I),
    "max": re.compile(r"\b(maximum|highest|peak|largest)\b", re.I),
    "min": re.compile(r"\b(minimum|lowest|smallest)\b", re.I),
}
_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")


def _single_series(text: str):
    series = extract_series(text)
    if len(series) != 1:
        return None
    x = np.asarray(series[0], dtype=float)
    x = x[~np.isnan(x)]
    return x if len(x) >= 3 else None


def mcq_label_is_correct(stem: str, options: dict[str, str], answer: str) -> bool | None:
    """Check a mean/max/min MCQ against the series. None if the question is not checkable."""
    x = _single_series(stem)
    values = {k: _NUMBER.match(v) for k, v in options.items()}
    kinds = [k for k, p in _STAT_QUESTION.items() if p.search(stem)]
    if x is None or not all(values.values()) or len(kinds) != 1 or _NOT_SIMPLE_STAT.search(stem):
        return None
    truth = {"mean": x.mean(), "max": x.max(), "min": x.min()}[kinds[0]]
    gaps = {k: abs(float(m.group(1)) - truth) for k, m in values.items()}
    return bool(gaps[answer] <= min(gaps.values()) + 1e-9)


_TF_MEAN = re.compile(r"\b(mean|average)\b[^.\[\]]*?\b(greater|higher|above|more|exceeds|less|lower|below)\b"
                      r"(?: than)?\s*(-?\d+(?:\.\d+)?)", re.I)


def tf_label_is_correct(statement: str, is_true: bool) -> bool | None:
    """Check a "the mean is greater/less than N" statement. None if not checkable."""
    # Replace arrays first so the pattern can span "the mean of [..] is greater than N".
    x, m = _single_series(statement), _TF_MEAN.search(ARRAY.sub("SERIES", statement))
    if x is None or not m or _NOT_SIMPLE_STAT.search(statement):
        return None
    above = m.group(2).lower() in ("greater", "higher", "above", "more", "exceeds")
    truth = x.mean() > float(m.group(3)) if above else x.mean() < float(m.group(3))
    return bool(truth == is_true)


def _has_data(stem: str, options: dict[str, str]) -> bool:
    """True if the stem holds a series or some option lists several numbers."""
    return bool(extract_series(stem)) or any(
        len(re.findall(r"-?\d+(?:\.\d+)?", v)) >= 3 for v in options.values())


# MCQ markers: "A) x", "(a) x", or uppercase "A. x" / "A: x" after whitespace or punctuation.
_MCQ_MARK = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[:;?]))(?:\(?([A-Fa-f])\)|([A-F])[.:](?=\s))\s*")
_MCQ_ANSWER = [
    re.compile(r"^\W*(?:the\s+)?(?:correct\s+)?(?:answer|option)?\s*(?:is)?:?\s*\(?([A-Fa-f])\)", re.I),
    re.compile(r"(?:answer is|option)\s*:?\s*\(?([A-Fa-f])(?:[).:\s]|$)", re.I),
    re.compile(r"^\s*([A-F])[.:]\s"),
]


def parse_mcq_options(question: str) -> tuple[str, dict[str, str]] | None:
    """Return (stem, options) using the last run of consecutive letters starting at A."""
    marks = list(_MCQ_MARK.finditer(question))
    letter = lambda m: (m.group(1) or m.group(2)).upper()  # noqa: E731
    starts = [i for i, m in enumerate(marks) if letter(m) == "A"]
    if not starts:
        return None
    run = [marks[starts[-1]]]
    for m in marks[starts[-1] + 1:]:
        if letter(m) != chr(ord(letter(run[-1])) + 1):
            break
        run.append(m)
    if len(run) < 2:
        return None
    options = {}
    for i, m in enumerate(run):
        end = run[i + 1].start() if i + 1 < len(run) else len(question)
        options[letter(m)] = question[m.end():end].strip().rstrip(",;").rstrip("?.").strip()
    # Drop a trailing "Options:" or a short "Choose ...:" lead-in, but never a stem that
    # itself starts with "Choose" and holds the data.
    stem = re.sub(r"\s*(Options?:|(?<=[.?!])\s*Choose[^:\[\]]*:)\s*$", "", question[:run[0].start()]).strip()
    return stem, options


def parse_mcq_answer(answer: str, options: dict[str, str]) -> str | None:
    for pattern in _MCQ_ANSWER:
        m = pattern.search(answer)
        if m and m.group(1).upper() in options:
            return m.group(1).upper()
    # Fall back to an answer that starts with exactly one option's text ("Yes (a). ...").
    hits = [k for k, v in options.items() if v and answer.strip().lower().startswith(v.lower())]
    return hits[0] if len(hits) == 1 else None


def convert_mcq(raw: pd.DataFrame, skipped: Counter) -> Iterator[Sample]:
    """Open-ended QA rows with question_format multiple_choice."""
    for row in raw[raw["format"] == "multiple_choice"].itertuples():
        parsed = parse_mcq_options(row.question)
        answer = parse_mcq_answer(row.answer, parsed[1]) if parsed else None
        if not answer:
            skipped["unparseable"] += 1
            continue
        stem, options = parsed
        if "forecast" in row.task or _FORECAST_TOPIC.search(stem):
            skipped["forecasting_unverifiable"] += 1
        elif not _has_data(stem, options):
            skipped["no_data"] += 1
        elif mcq_label_is_correct(stem, options, answer) is False:
            skipped["wrong_label"] += 1
        else:
            yield _sample(row, "mcq", stem, "Which option correctly answers the question in the state?",
                          options, answer)


_TF_ANSWER = re.compile(r"^\W*(?:(?:correct\s+)?answer:?\s*)?(true|false|yes|no)\b", re.I)


def convert_true_false(raw: pd.DataFrame, skipped: Counter) -> Iterator[Sample]:
    """Open-ended QA rows with question_format true/false, as a 2-option choice."""
    options = {"A": "True", "B": "False"}
    for row in raw[raw["format"] == "true/false"].itertuples():
        m = _TF_ANSWER.match(row.answer)
        if not m:
            skipped["unparseable"] += 1
            continue
        is_true = m.group(1).lower() in ("true", "yes")
        statement = re.sub(r"\s*True or False\??\s*$", "", row.question, flags=re.I)
        if "forecast" in row.task or _FORECAST_TOPIC.search(statement):
            skipped["forecasting_unverifiable"] += 1
        elif not extract_series(statement):
            skipped["no_data"] += 1
        elif tf_label_is_correct(statement, is_true) is False:
            skipped["wrong_label"] += 1
        else:
            yield _sample(row, "true_false", statement, "Is the statement in the state true?",
                          options, "A" if is_true else "B")


CONVERTERS: dict[str, Callable[[pd.DataFrame, Counter], Iterator[Sample]]] = {
    "classification": convert_classification,
    "anomaly": convert_anomaly,
    "mcq": convert_mcq,
    "true_false": convert_true_false,
}


# ------------------------------------------------------------------ public API

def _hash01(text: str) -> float:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(sample_id: str, val_fraction: float) -> str:
    """Stable train/val split from a hash of the sample id."""
    return "val" if _hash01(sample_id) < val_fraction else "train"


def shuffle_options(sample: Sample, seed: int = 0) -> Sample:
    """Reorder options with a per-sample seed and relabel them A, B, C, ..., since TSQA lists options in a fixed
    order."""
    items = list(sample.options.values())
    correct = sample.options[sample.answer]
    order = sorted(range(len(items)), key=lambda i: _hash01(f"{seed}:{sample.id}:{i}"))
    sample.options = {chr(ord("A") + j): items[i] for j, i in enumerate(order)}
    sample.answer = next(k for k, v in sample.options.items() if v == correct)
    return sample


def load_tsqa(
    converters: list[str],
    splits: list[str] = ("train",),
    val_fraction: float = 0.02,
    shuffle: bool = True,
    seed: int = 0,
    limit: int | None = None,
    domains: list[str] | None = None,
    max_per_domain: int | None = None,
    domain_offset: int = 0,
    offset_scope: str = "domain",
) -> list[Sample]:
    """Load TSQA as Choice Samples. `limit`, `domains`, `max_per_domain` and `domain_offset` keep seeded, disjoint
    slices per domain, or over all samples with offset_scope="all"."""
    raw = load_tsqa_raw(limit=limit)
    samples: list[Sample] = []
    for name in converters:
        if name not in CONVERTERS:
            raise ValueError(f"unknown TSQA converter: {name}")
        skipped: Counter = Counter()
        converted = list(CONVERTERS[name](raw, skipped))
        for s in converted:
            s.split = assign_split(s.id, val_fraction)
            if shuffle:
                shuffle_options(s, seed)
        samples.extend(s for s in converted if s.split in splits and (not domains or s.domain in domains))
        print(f"[tsqa] {name}: {len(converted)} samples; skipped {dict(skipped)}")
    if max_per_domain or domain_offset:
        by_domain: dict[str, list[Sample]] = {}
        for s in samples:
            by_domain.setdefault(s.domain if offset_scope == "domain" else "all", []).append(s)
        samples = []
        for domain in sorted(by_domain):
            group = sorted(by_domain[domain], key=lambda s: _hash01(f"{seed}:{s.id}"))
            samples.extend(group[domain_offset:domain_offset + max_per_domain if max_per_domain else None])
    return samples
