"""Local Decider backend (decider-ai) for plain-text checkpoints, without the value-token encoder."""

from __future__ import annotations

import time

from timesjev.models.base import DecisionModel
from timesjev.models.system_one import build_request, parse_answer
from timesjev.schema import Prediction, Sample

# Config keys passed straight to decider.infer.Decider.
DECIDER_KWARGS = ("device", "dtype", "temperature", "use_graphs")


class DeciderModel(DecisionModel):
    name = "decider"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from decider.infer import Decider

        path = cfg["checkpoint"]
        if cfg.get("revision"):
            # Decider takes no revision argument; pin by loading a fixed Hub snapshot.
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=cfg["revision"])
        kwargs = {k: cfg[k] for k in DECIDER_KWARGS if cfg.get(k) is not None}
        self.model = Decider(path, **kwargs)
        self.state_kind = cfg.get("state", "raw")

    def predict(self, samples: list[Sample]) -> list[Prediction]:
        # One call per sample keeps the latency per sample exact.
        preds = []
        for i, sample in enumerate(samples):
            state, questions = build_request(sample, self.state_kind)
            start = time.perf_counter()
            response = self.model.system_one(state, questions)
            preds.append(parse_answer(sample, response, time.perf_counter() - start))
            if (i + 1) % 50 == 0:
                print(f"[decider] {i + 1}/{len(samples)}")
        return preds
