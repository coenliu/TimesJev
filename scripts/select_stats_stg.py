"""Stage 1: select K features with stochastic gates (STG; Yamada et al., 2020), trained jointly with a small MLP on the
answer cross-entropy plus lambda times the expected number of open gates. For each budget K, the largest lambda that
leaves at least K gates open is used, and --write_selection saves its top-K features to
configs/selections/<task>-stg-<K>.json."""

import contextlib
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timesjev.config import PROJECT_ROOT, load_config  # noqa: E402
from timesjev.data import load_samples  # noqa: E402
from timesjev.data.pool import pool_row  # noqa: E402

BUDGETS = (4, 8, 16, 32)
LAMBDAS = (0.003, 0.01, 0.03, 0.1)
# Statement types of true/false questions, given to the predictor next to the gated features.
BUCKETS = [("seasonal", r"season|cycl|periodic"), ("volatility", r"volatil|fluctuat"), ("anomaly", r"anomal|outlier|spike"),
           ("break", r"structural|break|change point|shift"), ("up", r"upward|increas|rising|growth"),
           ("down", r"downward|decreas|declin|falling")]


def bucket(s):
    t = s.context.lower()
    for name, pat in BUCKETS:
        if re.search(pat, t):
            return name
    return "other"


def load(name):
    """A seeded sample of 4,000 training questions of a dataset."""
    cfg = load_config(str(PROJECT_ROOT / f"configs/experiments/ts_map/TM-{name}-frozen-sel-{name}-stg-32-s42.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_samples(cfg["data"])
    return random.Random(0).sample(pool, min(4000, len(pool)))


def stg(Xtr, ytr, Ctr, lam, seed=0, sigma=0.5, epochs=300, hidden=64):
    """Train gates + MLP; returns mu (gate means)."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(set(ytr))
    y = torch.tensor([classes.index(v) for v in ytr], device=dev)
    X = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    C = torch.tensor(Ctr, dtype=torch.float32, device=dev)
    d = X.shape[1]
    mu = torch.nn.Parameter(torch.full((d,), 0.5, device=dev))
    net = torch.nn.Sequential(torch.nn.Linear(d + C.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, len(classes))).to(dev)
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 1e-3}, {"params": [mu], "lr": 1e-2}])
    normal = torch.distributions.Normal(0.0, 1.0)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            z = torch.clamp(mu + sigma * torch.randn(d, device=dev), 0, 1)
            logits = net(torch.cat([X[b] * z, C[b]], 1))
            loss = torch.nn.functional.cross_entropy(logits, y[b]) + lam * normal.cdf(mu / sigma).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return mu.detach().cpu().numpy()


def main():
    global BUDGETS, LAMBDAS
    from sklearn.preprocessing import QuantileTransformer
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["env", "cls", "tf"])
    ap.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    ap.add_argument("--lambdas", nargs="*", type=float, default=list(LAMBDAS))
    ap.add_argument("--out", default="results/select_stats_stg.json")
    ap.add_argument("--write_selection", action="store_true", help="save configs/selections/<task>-stg-<B>.json")
    args = ap.parse_args()
    tasks, BUDGETS, LAMBDAS = args.tasks, tuple(args.budgets), tuple(args.lambdas)
    path = PROJECT_ROOT / args.out
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in tasks:
        train = load(name)
        lab = lambda s: s.options[s.answer]  # noqa: E731
        R = [pool_row(s) for s in train]
        names = R[0][1]
        Xtr = np.stack([r[0] for r in R])
        ytr = [lab(s) for s in train]
        B = [b for b, _ in BUCKETS] + ["other"]
        if name == "tf":
            ctr = np.eye(len(B))[[B.index(bucket(s)) for s in train]]
        else:
            ctr = np.zeros((len(train), 0))
        qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(Xtr)
        Ztr = qt.transform(Xtr)
        runs = {}
        for lam in LAMBDAS:
            mu = stg(Ztr, ytr, ctr, lam)
            runs[lam] = mu
            print(name, "lambda", lam, "open gates", int((mu > 0).sum()), flush=True)
        res = {"pool_size": len(names), "budgets": {}}
        for Bk in BUDGETS:
            # the sparsest run that still opens >= B gates (else the densest), then its top-B by mu
            cands = [lam for lam in LAMBDAS if (runs[lam] > 0).sum() >= Bk]
            lam = max(cands) if cands else min(LAMBDAS)
            sel = list(np.argsort(-runs[lam])[:Bk])
            res["budgets"][Bk] = {"lambda": lam, "open_gates": int((runs[lam] > 0).sum()),
                                  "stg_selected": [names[i] for i in sel]}
            if args.write_selection:
                (PROJECT_ROOT / "configs" / "selections" / f"{name}-stg-{Bk}.json").write_text(json.dumps(
                    {"names": [names[i] for i in sel], "method": f"stochastic gates (STG), budget {Bk}, lambda {lam}",
                     "dataset": f"TSQA {name}", "source": args.out}, indent=1))
            print(name, "B", Bk, "lambda", lam, flush=True)
        out[name] = res
        path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
