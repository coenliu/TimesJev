"""Run one evaluation: load data, predict, score and save runs/<name>_<timestamp>/ (config.yaml, predictions.jsonl,
metrics.json, report.md). One row per (split, task) is appended to results/experiments.csv."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from timesjev.config import PROJECT_ROOT
from timesjev.data import load_samples
from timesjev.eval.metrics import compute_metrics
from timesjev.eval.report import write_report
from timesjev.models import build_model

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_CSV = PROJECT_ROOT / "results" / "experiments.csv"
CSV_FIELDS = [
    "run_id", "exp", "date", "backend", "checkpoint", "state", "split", "task",
    "n", "sr", "acc", "acc_all", "ece", "brier", "nll", "latency_mean_s",
]


def run_eval(cfg: dict) -> Path:
    samples = load_samples(cfg["data"])
    model_cfg = {**cfg["model"], "state": cfg["data"].get("state", "raw")}
    model = build_model(model_cfg)
    preds = model.predict(samples)
    metrics = compute_metrics(samples, preds)

    run_id = f"{cfg['name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with open(run_dir / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict()) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(run_dir, cfg, samples, preds, metrics)
    append_results(run_id, cfg, metrics)
    return run_dir


def append_results(run_id: str, cfg: dict, metrics: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for split, tasks in metrics.items():
            for task, m in tasks.items():
                writer.writerow({
                    "run_id": run_id,
                    "exp": cfg["name"],
                    "date": f"{datetime.now():%Y-%m-%d}",
                    "backend": cfg["model"].get("backend"),
                    "checkpoint": cfg["model"].get("checkpoint") or cfg["model"].get("model_id"),
                    "state": cfg["data"].get("state", "raw"),
                    "split": split,
                    "task": task,
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                })
