"""Umid arxitektura variantlarini o'zbekcha matnda solishtirish (bench_extrapolation protokoli).

    python -m benchmarks.bench_lm_variants --name fikr --variant "dict(n_reads=2)" --seed 0
    python -m benchmarks.bench_lm_variants --name v1_256 --variant "dict()" --seq 256 --steps 600

Har bir ishga tushirish bitta qator chiqaradi: parametrlar, eng yaxshi qadam va val loss (128 … 2048).
"""

import argparse
import ast
import copy
import json
import time

import torch

from umid import UmidConfig, UmidLM
from umid.data import default_threads, evaluate, prepare_tokens, random_batch, split_tokens

from .common import RESULTS, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="dict()", help="UmidConfig maydonlari, masalan \"dict(n_reads=2)\"")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data", default="chat-data/Data.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    args = ap.parse_args()
    default_threads()
    variant = ast.literal_eval(args.variant) if args.variant.startswith("{") else eval(args.variant, {"dict": dict})

    tok, tokens = prepare_tokens(args.data, args.tokenizer, verbose=False)
    tr, va = split_tokens(tokens, 0.1)
    torch.manual_seed(args.seed)
    cfg = dict(vocab_size=tok.vocab_size, d_model=256, n_layers=4, n_heads=4, head_dim=64, dropout=0.1)
    cfg.update(variant)
    model = UmidLM(UmidConfig(**cfg))
    g = torch.Generator().manual_seed(args.seed)
    best = {"val": float("inf"), "state": None, "step": 0}

    def checkpoint(step):
        v = evaluate(model, va, args.seq)
        if v < best["val"]:
            best.update(val=v, step=step, state=copy.deepcopy(model.state_dict()))

    t0 = time.time()
    train(model, lambda s: random_batch(tr, 16, args.seq, g), args.steps, args.lr, log_every=0,
          callback=checkpoint, callback_every=50)
    model.load_state_dict(best["state"])
    res = {L: round(evaluate(model, va, L), 4) for L in (128, 256, 512, 2048)}
    out = {"name": args.name, "variant": variant, "seed": args.seed, "seq": args.seq, "steps": args.steps,
           "params": model.num_params(), "best_step": best["step"], "val_loss": res, "seconds": round(time.time() - t0)}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "lm_variants.jsonl", "a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
