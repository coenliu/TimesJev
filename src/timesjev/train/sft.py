"""Supervised training of a System 1 model (Decider): LoRA fine-tuning, or with freeze_lm only the value-token encoder
of train/ts_input.py. Training items are built with Decider's own prompt builder, and the loss is the cross-entropy over
the option letters at the answer slot."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from timesjev.config import PROJECT_ROOT
from timesjev.data import load_samples
from timesjev.train import ts_input
from timesjev.models.system_one import build_request
from timesjev.schema import Sample

RUNS_DIR = PROJECT_ROOT / "runs"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",           # full-attention layers
                "in_proj_qkv", "in_proj_z", "out_proj",            # Gated DeltaNet layers
                "gate_proj", "up_proj", "down_proj"]               # MLP


def lora_config(lora: dict):
    """peft LoraConfig from the `lora` block; `target_modules` restricts LoRA to some projections."""
    from peft import LoraConfig
    return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
                      target_modules=lora.get("target_modules") or LORA_TARGETS)


def build_item(sample: Sample, state_kind: str, tok, max_ctx: int) -> dict:
    """Token ids and answer slot for one sample, rendered as decider renders a system_one call."""
    from decider.data.core import Example, Q
    from decider.infer import _NoShuffle
    from decider.prompt import MAX_OPTIONS, build
    from decider.systemone import render_question, render_state

    state, questions = build_request(sample, state_kind)
    rq = render_question(questions["answer"])
    gold = list(sample.options).index(sample.answer)          # render_question keeps the criteria order
    return build(Example(render_state(state), [Q(rq["question"], rq["options"], gold)], task=sample.task),
                 tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=max_ctx, layout="state_first")


def select_samples(samples: list[Sample], per_task_cap: int | None, seed: int) -> list[Sample]:
    by_task: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_task[s.task].append(s)
    rng = random.Random(seed)
    out = []
    for task in sorted(by_task):
        group = by_task[task]
        rng.shuffle(group)
        out.extend(group[:per_task_cap] if per_task_cap else group)
    return out


def balanced_random(samples: list[Sample], n: int, seed: int = 0, positive_prefix: str = "Anomaly") -> list[Sample]:
    """n random samples, half with a positive answer and half with a negative one."""
    rng = random.Random(seed)
    pos = [s for s in samples if s.options[s.answer].startswith(positive_prefix)]
    neg = [s for s in samples if not s.options[s.answer].startswith(positive_prefix)]
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = pos[:n // 2] + neg[:n - n // 2]
    rng.shuffle(chosen)
    return chosen


def train_sft(cfg: dict) -> Path:
    import torch
    import torch.nn.functional as F
    from decider.model import DecisionModel, collate
    from decider.train import batches_by_tokens
    from peft import get_peft_model

    tcfg = cfg["train"]
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    state_kind = tcfg.get("state", "raw")

    run_dir = RUNS_DIR / f"{cfg['name']}_train_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    logf = open(run_dir / "train.log", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    pool = load_samples(cfg["data"])
    ts_cfg = ts_input.config(tcfg["ts_input"]) if tcfg.get("ts_input") else None
    if ts_cfg:
        in_dim = ts_input.in_dim_of([ts_input.value_rows(s, ts_cfg) for s in pool])
        model = ts_input.ts_model(cfg["model"]["checkpoint"], ts_cfg, in_dim).cuda()
    else:
        model = DecisionModel(cfg["model"]["checkpoint"]).cuda()
    tok = model.tok
    lora = tcfg["lora"]
    # get_peft_model injects the LoRA layers into model.lm in place. Keep model.lm as the plain
    # causal LM, because decider reads the backbone as model.lm.model; peft_model is only for merging.
    peft_model = get_peft_model(model.lm, lora_config(lora))
    model.lm.enable_input_require_grads()                      # needed with gradient checkpointing
    if tcfg.get("freeze_lm"):                                  # only the encoder trains; LoRA stays at its
        for prm in model.lm.parameters():                      # zero start (B = 0), so the saved adapter is a no-op
            prm.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] {cfg['model']['checkpoint']} + LoRA r={lora['r']} on {lora.get('target_modules') or 'all targets'}: "
        f"{n_train / 1e6:.1f}M trainable params")
    if ts_cfg:
        log(f"[ts_input] {ts_cfg}, value input size {in_dim}, "
            f"{sum(p.numel() for p in model.ts.parameters()) / 1e6:.2f}M encoder params")

    sampling = tcfg.get("sampling", "per_task")
    if sampling == "balanced_random":
        samples = balanced_random(pool, tcfg["n_samples"], seed, tcfg.get("positive_prefix", "Anomaly"))
    else:
        samples = select_samples(pool, tcfg.get("per_task_cap"), seed)

    max_ctx = tcfg.get("max_ctx_tokens", 4096)
    batch_like = tcfg.get("batch_like")

    def make_item(s):
        return ts_input.prepare_item(s, state_kind, tok, max_ctx, ts_cfg) if ts_cfg else build_item(s, state_kind, tok, max_ctx)

    coll = ts_input.collate_ts if ts_cfg else collate
    items = []
    for s in samples:
        it = make_item(s)
        if batch_like:
            it["batch_len"] = len(build_item(s, batch_like, tok, max_ctx)["ids"])
        items.append(it)
    log(f"[data] sampling={sampling}: {len(items)} samples, state={state_kind}")

    def make_batches(batch_rng):
        if batch_like:                                             # the batches of the plain text run
            return batches_by_tokens([{"ids": _Len(it["batch_len"])} for it in items], max_tokens, batch_rng)
        return batches_by_tokens(items, max_tokens, batch_rng)

    opt_cfg = tcfg["optim"]
    params = [p for p in model.parameters() if p.requires_grad]
    if ts_cfg:
        # The encoder gets its own learning rate (same warmup and cosine schedule, scaled).
        ts_params = list(model.ts.parameters())
        ts_ids = {id(p) for p in ts_params}
        groups = [{"params": [p for p in params if id(p) not in ts_ids]},
                  {"params": ts_params, "lr_scale": float(ts_cfg["lr"]) / opt_cfg["lr"]}]
        opt = torch.optim.AdamW([g for g in groups if g["params"]],             # freeze_lm: no LoRA group
                                lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        opt = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    max_tokens = opt_cfg.get("max_tokens_per_batch", 16384)
    # grad_accum k: each step's batch runs as k micro-batches with gradient accumulation, weighted by their share of
    # the items.
    grad_accum = int(opt_cfg.get("grad_accum", 1))
    epochs = opt_cfg.get("epochs", 1)
    total = epochs * len(make_batches(random.Random(0)))
    warmup = max(1, int(total * opt_cfg.get("warmup_ratio", 0.05)))

    def lr_at(step: int) -> float:
        if step < warmup:
            return opt_cfg["lr"] * step / warmup
        return opt_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    # Per-epoch monitor: accuracy on one of the config's eval sets (by suffix), computed in-process
    # with dropout off and its own batch order, so the training path is unchanged.
    monitor_items, curve = [], []
    if tcfg.get("monitor"):
        ev = next(e for e in cfg.get("eval", []) if e["suffix"] == tcfg["monitor"])
        monitor_items = [make_item(s) for s in load_samples(ev["data"])]
        log(f"[monitor] {tcfg['monitor']}: {len(monitor_items)} items, accuracy after every epoch")

    def monitor_acc() -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for bidx in batches_by_tokens(monitor_items, max_tokens, random.Random(0)):
                b = coll([monitor_items[i] for i in bidx], tok.pad_token_id)
                b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
                correct += (model(b).argmax(-1) == b["golds"]).sum().item()
        model.train()
        return 100 * correct / len(monitor_items)

    def step_loss(batch_items):
        b = coll(batch_items, tok.pad_token_id)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        return F.cross_entropy(model(b), b["golds"])

    model.train()
    step, t0, hist, monitor_s = 0, time.time(), [], 0.0
    for epoch in range(epochs):
        for bidx in make_batches(rng):
            batch_items = [items[i] for i in bidx]
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g.get("lr_scale", 1.0)
            if grad_accum > 1 and len(batch_items) > 1:
                k, loss_v = min(grad_accum, len(batch_items)), 0.0
                for c in range(k):
                    part = batch_items[c * len(batch_items) // k:(c + 1) * len(batch_items) // k]
                    loss_p = step_loss(part)
                    w = len(part) / len(batch_items)
                    (loss_p * w).backward()
                    loss_v += w * loss_p.item()
            else:
                loss = step_loss(batch_items)
                loss.backward()
                loss_v = loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            hist.append({"step": step, "loss": loss_v})
            if step % 20 == 0 or step == total:
                recent = hist[-20:]
                log(f"[train] step {step}/{total} loss {sum(h['loss'] for h in recent) / len(recent):.4f} "
                    f"lr {lr_at(step):.2e} {(time.time() - t0) / 60:.1f}min "
                    f"mem {torch.cuda.max_memory_allocated() / 1e9:.0f}GB")
        if monitor_items:
            t_m = time.time()
            curve.append({"epoch": len(curve) + 1, "step": step, "acc": monitor_acc()})
            monitor_s += time.time() - t_m
            log(f"[monitor] epoch {len(curve)}/{epochs} step {step}: {tcfg['monitor']} accuracy {curve[-1]['acc']:.1f}")
    (run_dir / "train_hist.json").write_text(json.dumps(hist))
    if curve:
        (run_dir / "monitor.json").write_text(json.dumps({"set": tcfg["monitor"], "curve": curve}))
    train_min = (time.time() - t0 - monitor_s) / 60                 # training only, monitor time excluded
    ckpt = save_checkpoint(cfg, peft_model, tok, run_dir)
    if ts_cfg:
        ts_input.save(model, ts_cfg, in_dim, run_dir / "adapter", ckpt)
    log(f"[done] {step} steps in {train_min:.1f} min; saved {ckpt}")
    return ckpt


class _Len:
    """A stand-in for an item's ids with only a length (for batches_by_tokens)."""
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


def decider_config_path(checkpoint: str) -> str:
    """decider_config.json of a Hub model id or of a local checkpoint folder."""
    local = Path(checkpoint) / "decider_config.json"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(checkpoint, "decider_config.json")


def save_checkpoint(cfg: dict, peft_model, tok, run_dir: Path) -> Path:
    """Merge LoRA into the weights and save a folder Decider can load like the original model; the LoRA adapter alone is
    saved first as adapter/."""
    ckpt = run_dir / "checkpoint"
    peft_model.save_pretrained(run_dir / "adapter")
    peft_model.merge_and_unload().save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    shutil.copy(decider_config_path(cfg["model"]["checkpoint"]), ckpt / "decider_config.json")
    return ckpt
