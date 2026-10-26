"""'Aql' benchmark'i: teng parametrli modellar mantiqiy masalalarda.

Barcha Umid variantlari Transformer bilan bir xil parametr soniga keltiriladi (MLP o'lchami
orqali, ±1%). Masalalar: hop (2 bosqichli zanjir), s3 (almashtirishlarni kuzatish), mqar.

    python -m benchmarks.bench_reasoning --task hop
    python -m benchmarks.bench_reasoning --task s3 --models gpt,umid,umid-opt
"""

import argparse

import torch
import torch.nn.functional as F

from umid import UmidConfig, UmidLM
from umid.baseline import GPT, GPTConfig
from umid.data import default_threads

from . import tasks
from .common import save_results, train

VARIANTS = {
    "umid": dict(),
    "umid-w2": dict(n_writes=2),
    "umid-loop2": dict(n_loops=2),
    "umid-opt": dict(mixer="optimal"),
    "umid-opt-loop2": dict(mixer="optimal", n_loops=2),
    "umid-frame": dict(mixer="optimal", frame=True),   # aylanadigan koordinatalardagi optimal xotira
}


def build_matched(kind, vocab, d, layers, heads, target):
    """Umid variantini Transformer parametr soniga (target) MLP o'lchami orqali tenglashtiradi."""
    if kind == "gpt":
        return GPT(GPTConfig(vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads))
    base = dict(vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads, head_dim=d // heads,
                chunk_size=32, **VARIANTS[kind])
    best = None
    for hidden in range(8, 8 * d, 8):
        n = UmidLM(UmidConfig(mlp_hidden_size=hidden, **base)).num_params()
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (hidden, n)
        if n > target:
            break
    return UmidLM(UmidConfig(mlp_hidden_size=best[0], **base))


def masked_ce(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    pred = model(x)[0].argmax(-1)
    model.train()
    m = y != -100
    return (pred[m] == y[m]).float().mean().item(), (pred[:, -1] == y[:, -1]).float().mean().item()


TASKS = {
    # nom: (lug'at, o'qitish batch funksiyasi, {sinov nomi: batch funksiyasi}, model o'lchami)
    "hop": (tasks.HOP_VOCAB,
            lambda g: tasks.hop_batch(64, 16, 2, 8, g),
            {"2-hop, 16 juft": lambda g: tasks.hop_batch(512, 16, 2, 8, g),
             "2-hop, 24 juft": lambda g: tasks.hop_batch(512, 24, 2, 8, g)},
            (128, 2, 4)),
    "s3": (tasks.S3_VOCAB,
           lambda g: tasks.s3_batch(64, 32, g),
           {"uzunlik 32": lambda g: tasks.s3_batch(512, 32, g),
            "uzunlik 64": lambda g: tasks.s3_batch(512, 64, g),
            "uzunlik 128": lambda g: tasks.s3_batch(512, 128, g)},
           (64, 2, 2)),
    "mqar": (tasks.mqar_vocab(),
             lambda g: tasks.mqar_batch(64, 32, generator=g),
             {"32 juft": lambda g: tasks.mqar_batch(512, 32, generator=g),
              "48 juft": lambda g: tasks.mqar_batch(512, 48, generator=g),
              "64 juft": lambda g: tasks.mqar_batch(512, 64, generator=g)},
             (64, 2, 2)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--models", default="gpt," + ",".join(VARIANTS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="", help="natija fayli nomiga qo'shimcha")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    else:
        default_threads()
    vocab, train_fn, tests, (d, layers, heads) = TASKS[args.task]
    target = GPT(GPTConfig(vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads)).num_params()

    results = {"args": vars(args), "models": {}}
    for kind in args.models.split(","):
        torch.manual_seed(args.seed)
        model = build_matched(kind, vocab, d, layers, heads, target)
        g = torch.Generator().manual_seed(args.seed)
        print(f"\n== {args.task} / {kind}: {model.num_params() / 1e3:.1f}K parametr ==", flush=True)
        loss, sec = train(model, lambda s: train_fn(g), args.steps, args.lr, name=kind, log_every=500,
                          loss_fn=masked_ce)
        acc = {}
        for name, fn in tests.items():
            acc[name] = accuracy(model, *fn(torch.Generator().manual_seed(1234)))
            print(f"  {name}: {acc[name][0]:.1%} (oxirgi token {acc[name][1]:.1%})", flush=True)
        results["models"][kind] = {"params": model.num_params(), "train_loss": loss, "sec_per_step": sec,
                                   "acc": {k: a for k, (a, _) in acc.items()},
                                   "acc_last": {k: b for k, (_, b) in acc.items()}}

    print(f"\n{args.task}: aniqlik (barcha baholanadigan pozitsiyalar)")
    print("model | params | " + " | ".join(tests))
    for kind, r in results["models"].items():
        print(f"{kind} | {r['params'] / 1e3:.0f}K | " + " | ".join(f"{r['acc'][t]:.1%}" for t in tests))
    save_results(f"reasoning_{args.task}" + (f"_{args.tag}" if args.tag else ""), results)


if __name__ == "__main__":
    main()
