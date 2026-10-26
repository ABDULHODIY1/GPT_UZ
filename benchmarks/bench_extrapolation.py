"""Uzunlikka ekstrapolyatsiya: ikkala model ham 128 tokenli oynalarda o'qitiladi,
so'ng ko'rilmagan matnda 128 … 2048 tokenli oynalarda baholanadi.

Ma'lumot kichik bo'lgani uchun har bir model uchun eng yaxshi val loss bergan checkpoint
olinadi (early stopping) — overfitting solishtirishni buzmasligi uchun.

Transformer (RoPE) o'qitilganidan uzun pozitsiyalarni ko'rmagan — loss keskin oshadi.
Umid'da pozitsiya tushunchasi yo'q, xotira esa qat'iy o'lchamli — loss barqaror qoladi.

    python -m benchmarks.bench_extrapolation
"""

import argparse
import copy

import torch

from umid.data import default_threads, evaluate, prepare_tokens, random_batch, split_tokens

from .common import build, save_results, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="chat-data/Data.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--models", default="umid,gpt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--train-len", type=int, default=128)
    ap.add_argument("--eval-lens", default="128,256,512,1024,2048")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    default_threads()

    tok, tokens = prepare_tokens(args.data, args.tokenizer)
    train_tok, val_tok = split_tokens(tokens, 0.1)
    eval_lens = [int(x) for x in args.eval_lens.split(",")]
    print(f"Train: {len(train_tok):,} token, val: {len(val_tok):,} token")

    results = {"args": vars(args), "models": {}}
    for kind in args.models.split(","):
        torch.manual_seed(args.seed)
        model = build(kind, tok.vocab_size, args.d_model, args.n_layers, args.n_heads, dropout=0.1)
        g = torch.Generator().manual_seed(args.seed)
        print(f"\n== {kind}: {model.num_params() / 1e6:.2f}M parametr ==")
        best = {"val": float("inf"), "step": 0, "state": None}

        def checkpoint(step):
            val = evaluate(model, val_tok, args.train_len)
            if val < best["val"]:
                best.update(val=val, step=step, state=copy.deepcopy(model.state_dict()))

        train_loss, sec = train(model, lambda s: random_batch(train_tok, args.batch_size, args.train_len, g),
                                args.steps, args.lr, name=kind, callback=checkpoint, callback_every=args.eval_every)
        model.load_state_dict(best["state"])
        print(f"  eng yaxshi checkpoint: {best['step']}-qadam")
        losses = {L: evaluate(model, val_tok, L) for L in eval_lens}
        for L, v in losses.items():
            print(f"  val loss @ {L:>5} token: {v:.4f}")
        results["models"][kind] = {"params": model.num_params(), "train_loss": train_loss, "best_step": best["step"],
                                   "sec_per_step": sec, "val_loss": losses}

    print("\nKontekst uzunligi | " + " | ".join(results["models"]))
    for L in eval_lens:
        print(f"{L:>17} | " + " | ".join(f"{r['val_loss'][L]:.4f}" for r in results["models"].values()))
    save_results("extrapolation", results)


if __name__ == "__main__":
    main()
