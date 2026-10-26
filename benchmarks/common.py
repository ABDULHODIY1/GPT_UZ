"""Benchmark'lar uchun umumiy yordamchilar: modellarni qurish va bir xil sharoitda o'qitish."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from umid import UmidConfig, UmidLM
from umid.baseline import GPT, GPTConfig
from umid.data import cosine_lr

RESULTS = Path(__file__).parent / "results"


def build(kind, vocab_size, d_model, n_layers, n_heads, dropout=0.0, **kw):
    """kind: 'umid' (β ∈ (0,2)), 'umid-pos' (ablation: β ∈ (0,1)) yoki 'gpt' (Transformer)."""
    if kind == "gpt":
        return GPT(GPTConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout))
    cfg = UmidConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                     head_dim=d_model // n_heads, dropout=dropout, allow_neg_eigval=(kind == "umid"), **kw)
    return UmidLM(cfg)


def make_optimizer(model, lr, weight_decay=0.1):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def train(model, sample_batch, steps, lr, warmup=50, log_every=100, name="", loss_fn=None,
          callback=None, callback_every=0):
    """sample_batch(step) -> (x, y). callback(step) har callback_every qadamda chaqiriladi.
    Qaytaradi: (oxirgi loss'lar o'rtachasi, sekund/qadam)."""
    opt = make_optimizer(model, lr)
    model.train()
    t0, recent = time.time(), []
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = cosine_lr(step, steps, lr, warmup)
        x, y = sample_batch(step)
        if loss_fn is None:
            _, loss = model(x, targets=y)
        else:
            loss = loss_fn(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent = (recent + [loss.item()])[-50:]
        if log_every and (step + 1) % log_every == 0:
            print(f"  [{name}] qadam {step + 1}/{steps}  loss {sum(recent) / len(recent):.4f}  "
                  f"{(time.time() - t0) / (step + 1):.3f}s/qadam", flush=True)
        if callback is not None and callback_every and (step + 1) % callback_every == 0:
            callback(step + 1)
    return sum(recent) / len(recent), (time.time() - t0) / steps


def save_results(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Natijalar saqlandi: {path}")
