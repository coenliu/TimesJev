"""Write report.md for one run: setup, results and answer distribution. A hand-written "## Notes" section is kept when
the report is rebuilt."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from timesjev.schema import Prediction, Sample

NOTES_PLACEHOLDER = "- (add observations here)"


def _pct(value) -> str:
    return "–" if value is None or value != value else f"{100 * value:.1f}"


def _num(value, digits: int = 3) -> str:
    return "–" if value is None or value != value else f"{value:.{digits}f}"


def _existing_notes(path: Path) -> str | None:
    if not path.exists():
        return None
    match = re.search(r"^## Notes\n(.*?)(?=^## |\Z)", path.read_text(), re.S | re.M)
    return match.group(1).strip() if match else None


def build_report(run_id: str, cfg: dict, samples: list[Sample], preds: list[Prediction],
                 metrics: dict, notes: str | None = None) -> str:
    model, data = cfg["model"], cfg["data"]
    lines = [
        f"# {run_id}",
        "",
        f"- Experiment: {cfg['name']}",
        f"- Model: {model.get('backend')}, {model.get('checkpoint')}",
        f"- Data: {data.get('source')}, splits {data.get('splits')}, state {data.get('state', 'raw')}",
        f"- Samples: {len(samples)}. Written {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Results",
        "",
        "| Split | Task | N | SR % | ACC % | ECE | Brier | NLL | Latency s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split, tasks in metrics.items():
        for task, m in sorted(tasks.items()):
            lines.append(
                f"| {split} | {task} | {m['n']} | {_pct(m.get('sr'))} | {_pct(m.get('acc'))} | "
                f"{_num(m.get('ece'))} | {_num(m.get('brier'))} | {_num(m.get('nll'))} | "
                f"{_num(m.get('latency_mean_s'))} |")

    # Answer distribution: a model that keeps picking one letter is guessing.
    lines += ["", "## Answer distribution", "",
              "| Split | Task | Gold letters | Predicted letters |", "|---|---|---|---|"]
    by_group: dict[tuple[str, str], tuple[Counter, Counter]] = {}
    for s, p in zip(samples, preds):
        gold, pred = by_group.setdefault((s.split, s.task), (Counter(), Counter()))
        gold[s.answer] += 1
        pred[p.answer or "invalid"] += 1
    fmt = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items()))  # noqa: E731
    for (split, task), (gold, pred) in sorted(by_group.items()):
        lines.append(f"| {split} | {task} | {fmt(gold)} | {fmt(pred)} |")

    invalid = sum(not p.valid for p in preds)
    lines += ["", f"{invalid} of {len(preds)} predictions are invalid.",
              "", "## Notes", "", notes or NOTES_PLACEHOLDER, "",
              "## Files", "",
              "- config.yaml: the full config of this run",
              "- predictions.jsonl: one prediction per sample, with probabilities",
              "- metrics.json: all metrics by split and task", ""]
    return "\n".join(lines)


def write_report(run_dir: Path, cfg: dict, samples: list[Sample], preds: list[Prediction],
                 metrics: dict) -> Path:
    path = run_dir / "report.md"
    path.write_text(build_report(run_dir.name, cfg, samples, preds, metrics, _existing_notes(path)))
    return path
