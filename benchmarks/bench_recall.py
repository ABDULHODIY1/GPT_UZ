"""Assotsiativ eslab qolish (MQAR): "kalit qiymat" juftliklari, keyin kalitlar bo'yicha so'rov.

    k1 v1 k2 v2 … kn vn | q1 q2 … qn   →  har bir q uchun mos v ni topish

Bu Transformer kuchli bo'lgan masala (u butun kontekstni so'zma-so'z saqlaydi).
Qat'iy o'lchamli xotirali modellar uchun bu eng qiyin test: biz bu yerda halol solishtiramiz.
O'qitish n=16 juftlikda, sinov n=16, 32, 48 da (o'qitishda ko'rilmagan uzunlik).

    python -m benchmarks.bench_recall
"""

import argparse

import torch
import torch.nn.functional as F

from umid.data import default_threads

from .common import build, save_results, train


def make_batch(batch_size, n_pairs, n_keys, n_values, generator=None):
    keys = torch.stack([torch.randperm(n_keys, generator=generator)[:n_pairs] for _ in range(batch_size)]) + 1
    vals = torch.randint(0, n_values, (batch_size, n_pairs), generator=generator) + 1 + n_keys
    order = torch.stack([torch.randperm(n_pairs, generator=generator) for _ in range(batch_size)])
    queries = keys.gather(1, order)
    answers = vals.gather(1, order)
    pairs = torch.stack([keys, vals], dim=2).reshape(batch_size, 2 * n_pairs)
    x = torch.cat([pairs, queries], dim=1)
    y = torch.cat([torch.full_like(pairs, -100), answers], dim=1)
    return x, y


def recall_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def accuracy(model, n_pairs, n_keys, n_values, n=256, seed=123):
    model.eval()
    x, y = make_batch(n, n_pairs, n_keys, n_values, torch.Generator().manual_seed(seed))
    pred = model(x)[0].argmax(-1)
    mask = y != -100
    model.train()
    return (pred[mask] == y[mask]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="umid,gpt")
    ap.add_argument("--train-pairs", type=int, default=16)
    ap.add_argument("--test-pairs", default="16,32,48")
    ap.add_argument("--n-keys", type=int, default=64)
    ap.add_argument("--n-values", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    default_threads()
    vocab = 1 + args.n_keys + args.n_values
    test_pairs = [int(t) for t in args.test_pairs.split(",")]

    results = {"args": vars(args), "models": {}}
    for kind in args.models.split(","):
        torch.manual_seed(args.seed)
        model = build(kind, vocab, args.d_model, args.n_layers, args.n_heads, chunk_size=32)
        g = torch.Generator().manual_seed(args.seed)
        print(f"\n== {kind}: {model.num_params() / 1e3:.0f}K parametr ==")
        loss, _ = train(model, lambda s: make_batch(args.batch_size, args.train_pairs, args.n_keys, args.n_values, g),
                        args.steps, args.lr, name=kind, log_every=500, loss_fn=recall_loss)
        acc = {n: accuracy(model, n, args.n_keys, args.n_values) for n in test_pairs}
        for n, a in acc.items():
            print(f"  {n:>3} juftlik: {a:.1%}")
        results["models"][kind] = {"params": model.num_params(), "train_loss": loss, "acc": acc}

    print(f"\nAniqlik (tasodifiy taxmin = {1 / args.n_values:.1%})")
    print("juftliklar | " + " | ".join(results["models"]))
    for n in test_pairs:
        print(f"{n:>10} | " + " | ".join(f"{r['acc'][n]:.1%}" for r in results["models"].values()))
    save_results("recall", results)


if __name__ == "__main__":
    main()
