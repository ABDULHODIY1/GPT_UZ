"""Tezlik va xotira: kontekst uzunligi oshganda nima bo'ladi?

1) Generatsiya: L tokenli kontekstdan keyin har bir yangi token qancha vaqt va xotira oladi.
   Transformer: KV-cache L ga proporsional o'sadi, har token vaqti ham o'sadi.
   Umid: xotira o'zgarmas, har token vaqti o'zgarmas.
2) O'qitish: bir xil token soni (B·T = const), lekin T har xil. Transformer attention O(T²),
   Umid O(T) — uzun matnlarda Umid tezlashadi.

    python -m benchmarks.bench_speed
"""

import argparse
import time

import torch

from umid.data import default_threads

from .common import build, save_results


def timed(fn, repeat=3):
    fn()
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


@torch.no_grad()
def decode_cost(model, ctx_len, vocab, n_new=32):
    model.eval()
    idx = torch.randint(0, vocab, (1, ctx_len))
    _, _, states = model(idx, return_states=True)
    nxt = torch.randint(0, vocab, (1, 1))

    def run():
        st = states
        for _ in range(n_new):
            _, _, st = model(nxt, states=st, return_states=True)

    sec = timed(run, repeat=2) / n_new
    return sec * 1e3, type(model).state_bytes(states)


def train_cost(model, T, tokens_per_batch, vocab):
    model.train()
    B = max(tokens_per_batch // T, 1)
    x = torch.randint(0, vocab, (B, T))

    def run():
        _, loss = model(x, targets=x)
        loss.backward()
        model.zero_grad(set_to_none=True)

    return B * T / timed(run, repeat=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-lens", default="256,1024,4096,16384")
    ap.add_argument("--train-lens", default="256,1024,4096,8192")
    ap.add_argument("--tokens-per-batch", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--models", default="umid,gpt")
    ap.add_argument("--tag", default="", help="natija fayli nomiga qo'shimcha")
    args = ap.parse_args()
    default_threads()
    torch.manual_seed(0)
    ap_models = args.models.split(",")
    models = {k: build(k, args.vocab, args.d_model, args.n_layers, args.n_heads) for k in ap_models}
    results = {"args": vars(args), "decode": {}, "train": {}}

    names = list(models)
    print("1) Generatsiya: kontekstdan keyin har bir yangi token (ms/token, holat MB)")
    print(f"{'kontekst':>9} | " + " | ".join(f"{k:>22}" for k in names))
    for L in (int(x) for x in args.ctx_lens.split(",")):
        row = {k: decode_cost(m, L, args.vocab) for k, m in models.items()}
        results["decode"][L] = {k: {"ms_per_token": ms, "state_bytes": b} for k, (ms, b) in row.items()}
        print(f"{L:>9} | " + " | ".join(f"{row[k][0]:>8.2f} ms {row[k][1] / 2**20:>8.2f} MB" for k in names), flush=True)

    print(f"\n2) O'qitish (forward+backward), {args.tokens_per_batch} token/batch, token/s")
    print(f"{'T':>6} | " + " | ".join(f"{k:>12}" for k in names))
    for T in (int(x) for x in args.train_lens.split(",")):
        row = {k: train_cost(m, T, args.tokens_per_batch, args.vocab) for k, m in models.items()}
        results["train"][T] = row
        print(f"{T:>6} | " + " | ".join(f"{row[k]:>12,.0f}" for k in names), flush=True)
    save_results("speed" + (f"_{args.tag}" if args.tag else ""), results)


if __name__ == "__main__":
    main()
