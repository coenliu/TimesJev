"""Evaluate a trained checkpoint on the eval sets of an experiment config, as scripts/train.py does after training.
--eval keeps the eval sets with the given suffixes, and --ids_file / --ids_key keep the question ids of a subset
file."""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timesjev.config import load_config  # noqa: E402
from timesjev.eval.runner import run_eval  # noqa: E402

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True,
                        help="trained checkpoint, e.g. runs/<name>_train_<time>/checkpoint")
    parser.add_argument("--eval", nargs="*", help="eval suffixes to run (default: all eval sets of the config)")
    parser.add_argument("--ids_file", help="subset file with ids[<key>] lists, e.g. configs/subsets/tsqa_test500.json")
    parser.add_argument("--ids_key", help="key in the subset file, e.g. web")
    parser.add_argument("--name", help="run name prefix (default: the config name)")
    args = parser.parse_args()
    if bool(args.ids_file) != bool(args.ids_key):
        parser.error("--ids_file and --ids_key go together")

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    train_state = cfg["train"].get("state", "raw")
    evals = [ev for ev in cfg.get("eval", []) if not args.eval or ev["suffix"] in args.eval]
    if not evals:
        parser.error(f"no eval set matches {args.eval}; the config has {[ev['suffix'] for ev in cfg.get('eval', [])]}")
    for ev in evals:
        data = copy.deepcopy(ev["data"])
        if data.get("state", "@train") == "@train":
            data["state"] = train_state
        if args.ids_file:
            data["ids_file"], data["ids_key"] = args.ids_file, args.ids_key
        model = {**cfg["model"], "checkpoint": str(ckpt), "revision": None}
        if (ckpt / "ts_module.pt").exists():          # the value-token encoder (train/ts_input.py)
            model["backend"] = "decider_ts"
        eval_cfg = {**cfg, "name": f"{args.name or cfg['name']}-{ev['suffix']}", "data": data, "model": model}
        print(f"evaluated: {run_eval(eval_cfg)}")


if __name__ == "__main__":
    main()
