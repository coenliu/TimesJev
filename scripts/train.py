"""Train the model of an experiment config, then evaluate it on every eval set listed in the config. In an eval data
block, `state: "@train"` means the input format used in training."""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timesjev.config import load_config  # noqa: E402
from timesjev.eval.runner import run_eval  # noqa: E402
from timesjev.train import run_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = run_training(cfg)
    print(f"trained: {checkpoint}")

    train_state = cfg["train"].get("state", "raw")
    for ev in cfg.get("eval", []):
        data = copy.deepcopy(ev["data"])
        if data.get("state", "@train") == "@train":
            data["state"] = train_state
        model = {**cfg["model"], "checkpoint": str(checkpoint), "revision": None}
        if (Path(checkpoint) / "ts_module.pt").exists():          # the value-token encoder (train/ts_input.py)
            model["backend"] = "decider_ts"
        eval_cfg = {**cfg, "name": f"{cfg['name']}-{ev['suffix']}", "data": data, "model": model}
        print(f"evaluated: {run_eval(eval_cfg)}")


if __name__ == "__main__":
    main()
