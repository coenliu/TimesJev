# TimesJev

Code for TimesJev.

## Repository structure

```
configs/
  models/        backbones
  train/         training defaults
  experiments/
    indomain/    datasets and Decider-2B baselines
    ts_map/      TimesJev with Decider-2B
    decider4b/   TimesJev and baselines with Decider-4B
  selections/    selected features
  subsets/       500-question test subsets
src/timesjev/
  data/          data loading and model input
  models/        Decider backends
  train/         training and value-token encoder
  eval/          evaluation and metrics
scripts/
  download.sh                  download data and models
  select_stats_stg.py          feature selection
  select_stats_stg_mcq.py      feature selection for MCQA
  train.py                     training and evaluation
  evaluate.py                  inference
```

Dataset keys: `env`, `aiops`, `health`, `web` (anomaly detection in four domains), `cls` (human activity
classification), `mcq` (multiple-choice QA), `tf` (true/false QA).

## Environment

Tested with Python 3.13.5 on a single NVIDIA RTX 5090 (32 GB).

```bash
python -m venv .venv && source .venv/bin/activate
# install the PyTorch 2.9.1 build that matches your CUDA version first, e.g. for CUDA 12.8:
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Pinned versions (`pyproject.toml`): torch 2.9.1, transformers 5.17.0, peft 0.21.0, accelerate 1.12.0,
decider-ai 1.8.0, pycatch22 0.4.5, scikit-learn 1.6.1, scipy 1.15.3, numpy 2.1.3, pandas 2.2.3, huggingface_hub 1.33.0.

## Data and models

TSQA is gated on Hugging Face: accept its terms on https://huggingface.co/datasets/Time-MQA/TSQA, then

```bash
hf auth login
bash scripts/download.sh
export HF_HUB_OFFLINE=1   # optional, after the download
```

This puts the TSQA files under `data/TSQA/` and the backbones `Mapika/decider-2b` and `Mapika/decider-4b` in the
Hugging Face cache, at the revisions used in our experiments. The dev and test sets of each dataset are carved from
TSQA as defined in `configs/experiments/indomain/_<ds>.yaml`.

## Quick start (Web)

```bash
python scripts/train.py configs/experiments/ts_map/TM-web-frozen-sel-web-stg-32-s42.yaml
```

This uses the provided selection `configs/selections/web-stg-32.json`, trains the value-token encoder with
Decider-2B frozen, and then evaluates on the eval sets of the config (`dev-web` and `test-web`). Outputs:

- `runs/TM-web-frozen-sel-web-stg-32-s42_train_<time>/checkpoint/`: the frozen backbone together with the trained
  encoder (`ts_module.pt`, `ts_config.json`), the folder loaded for evaluation
- `runs/TM-web-frozen-sel-web-stg-32-s42-test-web_<time>/`: `predictions.jsonl` (one line per question, with the
  option probabilities), `metrics.json` (accuracy, ECE, Brier score, NLL, latency) and `report.md`

## Stage 1: feature selection

The selections used in our experiments are provided in `configs/selections/`, so this stage can be skipped. To
recompute them (this overwrites the provided files):

```bash
python scripts/select_stats_stg.py env aiops health web cls tf --budgets 32 \
    --lambdas 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 --write_selection
python scripts/select_stats_stg_mcq.py
```

For each dataset, a gate is trained per feature of the library (69 features; 240 for `cls`) jointly with a small
predictor; the largest λ that leaves at least 32 gates open is used, and the 32 features with the largest gate means
are kept. A GPU is used when available.

## Stage 2: training the encoder

```bash
python scripts/train.py <config>
```

| Dataset | Decider-2B | Decider-4B |
|---|---|---|
| env | `ts_map/TM-env-frozen-sel-env-stg-32-s42.yaml` | `decider4b/D4B-env-frozen-sel-env-stg-32-s42-ep6.yaml` |
| aiops | `ts_map/TM-aiops-frozen-sel-aiops-stg-32-s42.yaml` | `decider4b/D4B-aiops-frozen-sel-aiops-stg-32-s42-ep6.yaml` |
| health | `ts_map/TM-health-frozen-sel-health-stg-32-s42.yaml` | `decider4b/D4B-health-frozen-sel-health-stg-32-s42.yaml` |
| web | `ts_map/TM-web-frozen-sel-web-stg-32-s42.yaml` | `decider4b/D4B-web-frozen-sel-web-stg-32-s42-ep6.yaml` |
| cls | `ts_map/TM-cls-frozen-sel-cls-stg-32-s42.yaml` | `decider4b/D4B-cls-frozen-sel-cls-stg-32-s42-ep6.yaml` |
| mcq | `ts_map/TM-mcq-frozen-sel-mcq-stg-32-s42.yaml` | `decider4b/D4B-mcq-frozen-sel-mcq-stg-32-s42-ep6.yaml` |
| tf | `ts_map/TM-tf-frozen-sel-tf-stg-32-s42.yaml` | `decider4b/D4B-tf-frozen-sel-tf-stg-32-s42.yaml` |

All paths are under `configs/experiments/`. The backbone stays frozen; only the value-token encoder is trained.

Decider LoRA SFT baselines (same script): `indomain/{ID,AIOps,Health,Web,CLS,MCQ,TF}-sft-lr2e-4-ep3-s42.yaml` for
Decider-2B (ID is env), and `decider4b/D4B-<ds>-sft-s42.yaml` for Decider-4B.

## Inference and evaluation

`scripts/train.py` already evaluates after training. To run inference with a trained checkpoint:

```bash
python scripts/evaluate.py <config> --checkpoint runs/<name>_train_<time>/checkpoint [--eval <suffix> ...] \
    [--ids_file <subset.json> --ids_key <ds>]
```

`--eval` selects eval sets of the config by suffix (default: all), and `--ids_file` / `--ids_key` restrict them to the
question ids listed in a subset file. For example, the Web test set restricted to its 500-question subset:

```bash
python scripts/evaluate.py configs/experiments/ts_map/TM-web-frozen-sel-web-stg-32-s42.yaml \
    --checkpoint runs/TM-web-frozen-sel-web-stg-32-s42_train_<time>/checkpoint \
    --eval test-web --ids_file configs/subsets/tsqa_test500.json --ids_key web
```

Each evaluation writes `runs/<name>-<suffix>_<time>/` (as above) and appends its metrics to `results/experiments.csv`.

## Licenses

This code is released under the Apache License 2.0 (`LICENSE`). TSQA and the Decider models are also released
under Apache-2.0.
