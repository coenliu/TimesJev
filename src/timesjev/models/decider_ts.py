"""Decider with the value-token encoder (train/ts_input.py): batched evaluation of a checkpoint folder that holds
ts_module.pt and ts_config.json. Prompts are built as in training, and probabilities use Decider's choice
temperature."""

from __future__ import annotations

import random
import time

from timesjev.models.base import DecisionModel
from timesjev.schema import Prediction, Sample

MAX_CTX = 32768                     # decider.infer.Decider.system_one's max_state_tokens
# One sample per forward by default, as decider.infer.Decider evaluates (padding in a batch changes the bf16
# numerics slightly); set model.eval_max_tokens (e.g. 16384) to batch.


class DeciderTSModel(DecisionModel):
    name = "decider_ts"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from decider import temperature as TT
        from decider.prompt import load_decider_config
        from timesjev.train import ts_input

        ckpt = cfg["checkpoint"]
        self.model, self.ts_cfg = ts_input.load(ckpt)
        self.model = self.model.cuda().eval()
        (T, by_type), _ = TT.from_config(load_decider_config(ckpt))
        self.T = by_type.get("choice", T)
        self.state_kind = cfg.get("state", "raw")

    def predict(self, samples: list[Sample]) -> list[Prediction]:
        import torch
        from decider import temperature as TT
        from decider.train import batches_by_tokens
        from timesjev.train import ts_input

        tok = self.model.tok
        items = [ts_input.prepare_item(s, self.state_kind, tok, MAX_CTX, self.ts_cfg) for s in samples]
        preds: list[Prediction | None] = [None] * len(samples)
        with torch.no_grad():
            for bidx in batches_by_tokens(items, int(self.cfg.get("eval_max_tokens", 1)), random.Random(0)):
                start = time.perf_counter()
                b = ts_input.collate_ts([items[i] for i in bidx], tok.pad_token_id)
                b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
                probs = TT.scaled_softmax(self.model(b), self.T).cpu()
                latency = (time.perf_counter() - start) / len(bidx)
                for row, i in enumerate(bidx):
                    s = samples[i]
                    letters = list(s.options)
                    p = {k: round(float(probs[row, j]), 4) for j, k in enumerate(letters)}
                    best = max(p, key=p.get)
                    preds[i] = Prediction(id=s.id, task=s.task, split=s.split, answer=best, probs=p,
                                          latency_s=latency, extra={"confidence": p[best]})
        return preds
