"""Value-token encoder of TimesJev: every number of the selected statistics text is followed by a placeholder token,
whose embedding is replaced by the output of a small MLP over that number and a one-hot of its statistic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from timesjev.schema import Sample

PLACEHOLDER = "<|vision_pad|>"
DEFAULTS = {"module": "value", "lr": 1e-3}


def config(ts_cfg: dict) -> dict:
    cfg = {**DEFAULTS, **(ts_cfg or {})}
    if cfg["module"] != "value":
        raise ValueError(f"unknown ts_input module: {cfg['module']}")
    return cfg


def value_row(v: float) -> np.ndarray:
    """Input of one value token: signed log1p, sign, order of magnitude / 5, and v / (1 + |v|)."""
    return np.asarray([np.sign(v) * np.log1p(abs(v)), np.sign(v), np.log10(abs(v) + 1e-6) / 5, v / (1 + abs(v))],
                      np.float32)


def value_rows(sample: Sample, cfg: dict) -> np.ndarray | None:
    """Input rows of a sample's value tokens, one per selected statistic; with value_types each row also gets a
    one-hot of its statistic."""
    from timesjev.data.pool import selected, selection
    block = cfg["value_block"]
    if not block.startswith("select:"):
        raise ValueError(f"value_block must name a selection (select:<key>), got {block}")
    key = block.split(":", 1)[1]
    vals, types = selected(sample, key)
    if not vals:
        return None
    rows = np.stack([value_row(v) for v in vals])
    if cfg.get("value_types"):
        names = selection(key)
        idx = [names.index(t) for t in types]
        rows = np.concatenate([rows, np.eye(len(names), dtype=np.float32)[idx]], 1)
    return rows


def prepare_item(sample: Sample, state_kind: str, tok, max_ctx: int, cfg: dict) -> dict:
    """build_item of the sample plus the rows of the placeholders that survive context truncation."""
    from timesjev.train.sft import build_item
    rows = value_rows(sample, cfg)
    it = build_item(sample, state_kind, tok, max_ctx)
    pid = tok.convert_tokens_to_ids(PLACEHOLDER)
    n = sum(1 for t in it["ids"] if t == pid)
    if rows is None or n == 0:
        it["ts_rows"] = None
    else:
        assert n <= len(rows), (sample.id, n, len(rows))
        it["ts_rows"] = rows[:n]
    return it


def in_dim_of(rows: list[np.ndarray | None]) -> int:
    dims = {r.shape[1] for r in rows if r is not None}
    if len(dims) != 1:
        raise ValueError(f"value tokens need one input size per run, got {sorted(dims)}")
    return dims.pop()


def collate_ts(items: list[dict], pad_id: int) -> dict:
    """decider's collate plus the value-token rows of all items, in batch and prompt order."""
    import torch
    from decider.model import collate
    b = collate(items, pad_id)
    rows = [it["ts_rows"] for it in items if it.get("ts_rows") is not None]
    if rows:
        b["ts_rows"] = torch.from_numpy(np.concatenate(rows))
    return b


def build_module(in_dim: int, embedding):
    """The encoder MLP; `embedding` is the model's input embedding layer, read only to scale the initial output."""
    import torch
    from torch import nn

    with torch.no_grad():                                                     # RMS of the word embeddings
        w = embedding.weight
        idx = torch.randperm(w.shape[0], generator=torch.Generator().manual_seed(0))[:20000].to(w.device)
        emb_rms = w[idx].float().pow(2).mean().sqrt().item()
    hidden = embedding.weight.shape[1]

    class ValueMLP(nn.Module):
        """A two-layer MLP from a value row to the hidden size, with output starting at the word-embedding RMS."""
        def __init__(self, width: int = 256):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, hidden))
            nn.init.normal_(self.net[2].weight, std=emb_rms / math.sqrt(width))
            nn.init.zeros_(self.net[2].bias)

        def forward(self, x):
            return self.net(x)

    return ValueMLP()


def ts_model(checkpoint: str, cfg: dict, in_dim: int, grad_ckpt: bool = True):
    """decider's DecisionModel with the encoder `ts`, whose outputs replace the placeholder embeddings."""
    import torch
    import torch.nn.functional as F
    from decider.model import DecisionModel, cap_logits
    from decider.prompt import MAX_OPTIONS

    class TSDecisionModel(DecisionModel):
        def __init__(self):
            super().__init__(checkpoint, grad_ckpt=grad_ckpt)
            self.placeholder_id = self.tok.convert_tokens_to_ids(PLACEHOLDER)
            # Own RNG stream: the global torch RNG (LoRA initialisation, dropout) stays as in a run without the encoder.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(cfg.get("init_seed", 0)))
                self.ts = build_module(in_dim, self.lm.get_input_embeddings())

        def forward(self, batch):
            ids = batch["input_ids"]
            emb = self.lm.get_input_embeddings()(ids)
            if batch.get("ts_rows") is not None:
                mask = ids == self.placeholder_id
                vals = self.ts(batch["ts_rows"].to(emb.device)).to(emb.dtype)
                assert int(mask.sum()) == len(vals), (int(mask.sum()), len(vals))
                emb = emb.masked_scatter(mask[..., None], vals)               # out of place
            out = self.lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"])
            hs = out.last_hidden_state[batch["slot_batch"], batch["slot_idx"]]
            logits = cap_logits(self, F.linear(hs, self.lm.lm_head.weight[self.letters]).float())
            ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
            return logits.masked_fill(ar >= batch["nopts"][:, None], float("-inf"))

    return TSDecisionModel()


def save(model, cfg: dict, in_dim: int, *dirs: Path) -> None:
    import torch
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.ts.state_dict(), d / "ts_module.pt")
        (d / "ts_config.json").write_text(json.dumps({**cfg, "in_dim": in_dim}, indent=2))


def load(checkpoint: str, grad_ckpt: bool = False):
    """(model with its encoder loaded, ts config) from a run's checkpoint folder."""
    import torch
    cfg = json.loads((Path(checkpoint) / "ts_config.json").read_text())
    model = ts_model(checkpoint, cfg, cfg["in_dim"], grad_ckpt=grad_ckpt)
    model.ts.load_state_dict(torch.load(Path(checkpoint) / "ts_module.pt", map_location="cpu"))
    return model, cfg
