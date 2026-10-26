"""Holatni kuzatish (state tracking): bitlar ketma-ketligining juft/toqligi (parity).

Model har bir pozitsiyada shu paytgacha kelgan "1" lar soni juft yoki toq ekanini aytishi
kerak. O'qitish 40 tokenli ketma-ketliklarda, sinov esa 40 … 640 tokenda.

Transformer bu masalani uzunlikka umumlashtira olmaydi (softmax attention o'rtacha oladi,
hisoblagich saqlay olmaydi). Umid'da β ∈ (0, 2) bo'lgani uchun (I − β k kᵀ) aks ettirish
(reflection) bo'la oladi: har bir "1" xotiradagi ishorani almashtiradi — bu aniq parity.
Ablation: β ∈ (0, 1) bo'lsa (oddiy DeltaNet/Mamba kabi) bu imkoniyat yo'qoladi.

    python -m benchmarks.bench_parity
"""

import argparse

import torch
import torch.nn.functional as F

from umid.data import default_threads

from .common import build, save_results, train

BOS, ZERO, ONE = 0, 1, 2


def make_batch(batch_size, length, generator=None):
    bits = torch.randint(0, 2, (batch_size, length), generator=generator)
    x = torch.cat([torch.full((batch_size, 1), BOS), bits + ZERO], dim=1)
    parity = bits.cumsum(1) % 2
    y = torch.cat([torch.full((batch_size, 1), -100), parity + ZERO], dim=1)
    return x, y


def parity_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def accuracy(model, length, n=256, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x, y = make_batch(n, length, g)
    pred = model(x)[0].argmax(-1)
    mask = y != -100
    all_pos = (pred[mask] == y[mask]).float().mean().item()
    last = (pred[:, -1] == y[:, -1]).float().mean().item()
    model.train()
    return all_pos, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="umid,umid-pos,gpt")
    ap.add_argument("--train-len", type=int, default=40)
    ap.add_argument("--test-lens", default="40,80,160,320,640")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    default_threads()
    test_lens = [int(t) for t in args.test_lens.split(",")]

    results = {"args": vars(args), "models": {}}
    for kind in args.models.split(","):
        runs = []
        for seed in (int(s) for s in args.seeds.split(",")):
            torch.manual_seed(seed)
            model = build(kind, 3, args.d_model, args.n_layers, args.n_heads, chunk_size=32)
            g = torch.Generator().manual_seed(seed)
            print(f"\n== {kind} (seed {seed}): {model.num_params() / 1e3:.0f}K parametr ==")
            loss, _ = train(model, lambda s: make_batch(args.batch_size, args.train_len, g), args.steps,
                            args.lr, name=kind, log_every=500, loss_fn=parity_loss)
            acc = {L: accuracy(model, L) for L in test_lens}
            for L, (a, last) in acc.items():
                print(f"  uzunlik {L:>4}: barcha pozitsiyalar {a:.1%}, oxirgi token {last:.1%}")
            runs.append({"seed": seed, "train_loss": loss, "acc_all": {L: a for L, (a, _) in acc.items()},
                         "acc_last": {L: b for L, (_, b) in acc.items()}})
        best = max(runs, key=lambda r: sum(r["acc_all"].values()))
        results["models"][kind] = {"runs": runs, "best": best}

    print("\nOxirgi token aniqligi (eng yaxshi seed; tasodifiy taxmin = 50%)")
    print("uzunlik | " + " | ".join(results["models"]))
    for L in test_lens:
        print(f"{L:>7} | " + " | ".join(f"{r['best']['acc_last'][L]:.1%}" for r in results["models"].values()))
    save_results("parity", results)


if __name__ == "__main__":
    main()
