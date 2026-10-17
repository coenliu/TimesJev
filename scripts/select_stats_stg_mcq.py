"""Stage 1 for MCQA: select 32 features with stochastic gates, using as predictor a per-option scorer over the pool
features of the series and the TF-IDF of the question and option text. Writes configs/selections/mcq-stg-32.json."""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timesjev.config import PROJECT_ROOT, load_config  # noqa: E402
from timesjev.data import load_samples  # noqa: E402
from timesjev.data.pool import pool_row  # noqa: E402
from timesjev.data.state import no_series_state  # noqa: E402

TOKENS = r"-?\d+(?:\.\d+)?|\w+|[^\w\s]"          # signed decimals, words incl. single letters, single symbols
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
N_STAT = 69
LAMBDAS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
BUDGET, SIGMA = 32, 0.5


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def option_numbers(text: str) -> list[float]:
    """[has a number, symlog(first), symlog(last), count / 5] of the numbers in an option."""
    xs = [float(v) for v in NUMBER.findall(text)]
    return [1.0, symlog(xs[0]), symlog(xs[-1]), min(len(xs), 5) / 5] if xs else [0.0, 0.0, 0.0, 0.0]


def main():
    import torch
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import QuantileTransformer
    from timesjev.train.sft import select_samples

    cfg = load_config(str(PROJECT_ROOT / "configs/experiments/ts_map/TM-mcq-frozen-sel-mcq-stg-32-s42.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_samples(cfg["data"])
        dev = load_samples(next(e["data"] for e in cfg["eval"] if e["suffix"].startswith("dev")))
    train = select_samples(pool, 4000, 42)

    def stats(s):
        if not s.series or not len(s.series[0]):
            return np.zeros(N_STAT + 1, np.float32)
        return np.concatenate([np.asarray(pool_row(s)[0], np.float32), [1.0]])

    def texts(s):
        q = f"{no_series_state(s)} {s.question}"
        return [f"{q} || {o}" for o in s.options.values()]

    Rtr = np.stack([stats(s) for s in train])
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(Rtr[:, :N_STAT])
    has = Rtr[:, N_STAT:]                                 # [quantile (69) | symlog (69) | no-series flag]
    Ftr = np.hstack([qt.transform(Rtr[:, :N_STAT]) * has, symlog(Rtr[:, :N_STAT]), has]).astype(np.float32)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=TOKENS)
    vec.fit([t for s in train for t in texts(s)])
    svd = TruncatedSVD(64, random_state=0).fit(vec.transform([t for s in train for t in texts(s)]))

    def enc(s):
        e = svd.transform(vec.transform(texts(s)))
        return np.hstack([e, [option_numbers(o) for o in s.options.values()]]).astype(np.float32)

    Etr, Edv = [enc(s) for s in train], [enc(s) for s in dev]
    K = max(len(e) for e in Etr + Edv)                    # most options of any training or dev question (some have 5)
    X = np.zeros((len(Etr), K, Etr[0].shape[1]), np.float32); M = np.zeros((len(Etr), K), bool)
    for i, e in enumerate(Etr):
        X[i, :len(e)] = e; M[i, :len(e)] = True
    Xtr, Mtr, Ttr = torch.tensor(X), torch.tensor(M), torch.tensor(Ftr)
    Ytr = torch.tensor([list(s.options).index(s.answer) for s in train])

    def scorer(d):
        return torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                   torch.nn.Linear(256, 1))

    def scores(h, T, X, M):
        z = h(torch.cat([T[:, None, :].expand(-1, X.shape[1], -1), X], 2)).squeeze(2)
        return z.masked_fill(~M, float("-inf"))

    normal = torch.distributions.Normal(0.0, 1.0)

    def stg(lam, seed=0, epochs=60):
        torch.manual_seed(seed)
        mu = torch.nn.Parameter(torch.full((N_STAT,), 0.5))
        h = scorer(Ftr.shape[1] + Xtr.shape[2])
        opt = torch.optim.Adam([{"params": h.parameters(), "lr": 1e-3}, {"params": [mu], "lr": 1e-2}])
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            for b in torch.randperm(len(Ytr), generator=g).split(64):
                z = torch.clamp(mu + SIGMA * torch.randn(N_STAT), 0, 1)
                gate = torch.cat([z, z, torch.ones(1)])          # one gate per statistic, on both its columns
                sc = scores(h, Ttr[b] * gate, Xtr[b], Mtr[b])
                loss = torch.nn.functional.cross_entropy(sc, Ytr[b]) + lam * normal.cdf(mu / SIGMA).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return mu.detach().numpy()

    names = next(pool_row(x)[1] for x in train if x.series and len(x.series[0]))
    runs = {}
    for lam in LAMBDAS:
        runs[lam] = stg(lam)
        print("lambda", lam, "open gates", int((runs[lam] > 0).sum()), flush=True)
    # the sparsest run that still opens >= BUDGET gates (else the densest), then its top-BUDGET by mu
    cands = [lam for lam in LAMBDAS if (runs[lam] > 0).sum() >= BUDGET]
    lam = max(cands) if cands else min(LAMBDAS)
    sel = [names[int(i)] for i in np.argsort(-runs[lam])[:BUDGET]]
    (PROJECT_ROOT / "configs" / "selections" / "mcq-stg-32.json").write_text(json.dumps(
        {"names": sel, "method": f"stochastic gates (STG) with a per-option scorer (number-preserving TF-IDF text + "
         f"option numbers + statistics on two scales), budget {BUDGET}, lambda {lam}", "dataset": "TSQA mcq",
         "source": "scripts/select_stats_stg_mcq.py"}, indent=1))
    print("lambda", lam, "open", int((runs[lam] > 0).sum()), "selected", sel, flush=True)


if __name__ == "__main__":
    main()
