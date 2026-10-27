"""Umid modelini o'qitish.

Misollar:
    python train.py                                   # chat-data/Data.txt, "tiny" model
    python train.py --preset small --steps 5000 --seq-len 512
    python train.py --data mening_matnlarim/ --out checkpoints/katta --preset base --compile
    python train.py --stream                          # holatni uzatib o'qitish (cheksiz kontekst)
    python train.py --out checkpoints/umid --resume   # to'xtagan joydan davom ettirish
    python train.py --arch gpt --out checkpoints/gpt  # taqqoslash uchun Transformer, aynan shu sharoitda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from umid import PRESETS, BPETokenizer, UmidConfig, UmidLM, load_model
from umid.baseline import GPT, GPTConfig
from umid.model import generate
from umid.data import StreamBatcher, cosine_lr, evaluate, prepare_tokens, random_batch, split_tokens


def parse_args():
    ap = argparse.ArgumentParser(description="Umid LM o'qitish")
    ap.add_argument("--data", default="chat-data/Data.txt!,chat-data/savol-javob*8",
                    help=".txt fayl/papka yoki ro'yxat: '!' — tozalash, '*N' — N marta takrorlash")
    ap.add_argument("--out", default="checkpoints/umid", help="checkpoint papkasi")
    ap.add_argument("--tokenizer", default=None, help="tayyor tokenizer.json (bo'lmasa o'qitiladi)")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--arch", default="umid", choices=["umid", "gpt"], help="gpt — taqqoslash uchun Transformer")
    ap.add_argument("--preset", default="tiny", choices=list(PRESETS))
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--n-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--no-neg-eigval", action="store_true", help="β ∈ (0,1) (ablation uchun)")
    ap.add_argument("--reset-on-eot", action="store_true",
                    help="EOT da xotirani tozalash (Umid) / hujjat maskasi (gpt): hujjatlar aralashmaydi")
    ap.add_argument("--decouple-erase", action="store_true", help="o'chirish kuchi alohida darvoza (Umid)")
    ap.add_argument("--persistent-heads", type=int, default=0, help="unutmaydigan head'lar soni (Umid)")
    ap.add_argument("--canon", default="", help="Canon qatlamlari (Umid): masalan ACD")
    ap.add_argument("--mixer", default="delta", choices=["delta", "optimal", "tanlov"],
                    help="xotira algoritmi: delta (Gated Delta) yoki tanlov (eng muhim K ta token, umid/tanlov.py)")
    ap.add_argument("--select-k", type=int, default=32, help="tanlov: har head saqlaydigan tokenlar soni")
    ap.add_argument("--select-enrich", action="store_true", help="tanlov: o'qib boyitish")
    ap.add_argument("--select-no-conv", action="store_true", help="tanlov: konvolyutsiyasiz (sof Tanlov)")
    ap.add_argument("--n-reads", type=int, default=None, help="fikrlash rejimi: xotiradan qayta so'rashlar soni (standart 1)")
    ap.add_argument("--decay", choices=["head", "channel"], default=None)
    ap.add_argument("--read-mix-init", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=0, help="0 = butun val to'plami (katta korpusda cheklang)")
    ap.add_argument("--stream", action="store_true", help="ketma-ket oynalar, xotira holati uzatiladi")
    ap.add_argument("--sample-prompt", default="Salom")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="tayyor model.pt dan boshlash (fine-tuning): og'irliklar va tokenizer")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args, vocab_size):
    kw = dict(PRESETS[args.preset])
    for key in ("d_model", "n_layers", "n_heads", "head_dim"):
        if getattr(args, key) is not None:
            kw[key] = getattr(args, key)
    if args.n_reads is not None:
        kw["n_reads"] = args.n_reads
    if args.decay is not None:
        kw["decay"] = args.decay
    if args.read_mix_init is not None:
        kw["read_mix_init"] = args.read_mix_init
    if args.arch == "gpt":
        return GPTConfig(vocab_size=vocab_size, d_model=kw["d_model"], n_layers=kw["n_layers"], n_heads=kw["n_heads"],
                         doc_mask=args.reset_on_eot, canon=args.canon, dropout=args.dropout)
    return UmidConfig(vocab_size=vocab_size, dropout=args.dropout, allow_neg_eigval=not args.no_neg_eigval,
                      reset_on_eot=args.reset_on_eot, decouple_erase=args.decouple_erase,
                      persistent_heads=args.persistent_heads, canon=args.canon, mixer=args.mixer,
                      select_k=args.select_k, select_enrich=args.select_enrich,
                      select_conv=not args.select_no_conv, **kw)


def detach_states(states):
    return [{k: v.detach() for k, v in st.items()} for st in states]


def main():
    args = parse_args()
    if args.stream and args.arch == "gpt":
        raise SystemExit("--stream faqat Umid uchun: Transformer holati (KV-cache) o'zgarmas hajmli emas")
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tok_path = Path(args.tokenizer) if args.tokenizer else out / "tokenizer.json"
    init_model = None
    if args.init:   # fine-tuning: model va tokenizer tayyor checkpoint'dan
        init_model, init_ckpt = load_model(args.init)
        BPETokenizer.from_dict(init_ckpt["tokenizer"]).save(tok_path)
    tok, tokens = prepare_tokens(args.data, tok_path, args.vocab_size)
    train_tok, val_tok = split_tokens(tokens, args.val_frac)
    print(f"Ma'lumot: {len(tokens):,} token (train {len(train_tok):,}, val {len(val_tok):,}), vocab {tok.vocab_size}")

    last_path, best_path = out / "last.pt", out / "model.pt"
    start, best_val = 0, float("inf")
    if args.resume and last_path.exists():
        model, ckpt = load_model(last_path)
        start, best_val = ckpt["step"], ckpt.get("best_val", best_val)
        print(f"Davom ettirilmoqda: {start}-qadamdan")
    elif init_model is not None:
        model, ckpt = init_model, None
        if args.dropout is not None:
            model.cfg.dropout = args.dropout
            for mod in model.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = args.dropout
        print(f"Boshlang'ich og'irliklar: {args.init}")
    else:
        cfg = build_config(args, tok.vocab_size)
        model, ckpt = (GPT(cfg) if args.arch == "gpt" else UmidLM(cfg)), None
    model.to(device)
    cfg = model.cfg
    kind = ("Transformer (taqqoslash uchun)" if isinstance(model, GPT) else
            f"Umid-Tanlov, K={cfg.select_k}" if cfg.mixer == "tanlov" else
            f"Umid, β∈(0,{2 if cfg.allow_neg_eigval else 1})")
    print(f"Model: {model.num_params() / 1e6:.2f}M parametr | {kind} | {cfg.n_layers} qatlam, d={cfg.d_model}, "
          f"{cfg.n_heads} head | qurilma: {device}")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    if ckpt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    fwd = torch.compile(model) if args.compile else model

    g = torch.Generator().manual_seed(args.seed + start)
    streamer = StreamBatcher(train_tok, args.batch_size, args.seq_len, device) if args.stream else None
    states = None

    def save(path, step, with_opt):
        extra = {"tokenizer": tok.to_dict(), "step": step, "best_val": best_val, "train_args": vars(args)}
        if with_opt:
            extra["optimizer"] = opt.state_dict()
        model.save(path, **extra)

    model.train()
    t0, tokens_seen, recent = time.time(), 0, []
    for step in range(start, args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        for _ in range(args.grad_accum):
            if streamer is not None:
                x, y, reset = streamer.next()
                if reset:
                    states = None
                with autocast():
                    _, loss, states = fwd(x, targets=y, states=states, return_states=True)
                states = detach_states(states)
            else:
                x, y = random_batch(train_tok, args.batch_size, args.seq_len, g, device)
                with autocast():
                    _, loss = fwd(x, targets=y)
            (loss / args.grad_accum).backward()
            tokens_seen += y.numel()
            recent = (recent + [loss.item()])[-20:]
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"qadam {step + 1:>6}/{args.steps} | loss {sum(recent) / len(recent):.4f} | lr {lr:.2e} | "
                  f"{tokens_seen / dt:,.0f} token/s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = evaluate(model, val_tok, args.seq_len, max_windows=args.eval_windows, device=device, autocast=autocast())
            val_long = evaluate(model, val_tok, args.seq_len * 4, max_windows=max(args.eval_windows // 4, 1) if args.eval_windows else None,
                                device=device, autocast=autocast())
            mark = ""
            if val < best_val:
                best_val = val
                save(best_path, step + 1, with_opt=False)
                mark = "  ✓ eng yaxshi model saqlandi"
            print(f"== baholash: val loss {val:.4f} (ppl {torch.tensor(val).exp():.1f}) | "
                  f"{args.seq_len * 4} tokenli kontekstda: {val_long:.4f}{mark}")
            save(last_path, step + 1, with_opt=True)
            if args.sample_prompt:
                ids = torch.tensor([tok.encode(args.sample_prompt)], device=device)
                new, _ = generate(model, ids, 40, temperature=0.8)
                print(f"   namuna: {args.sample_prompt}{tok.decode(new[0].tolist())}")
            model.train()

    print(f"Tayyor. Eng yaxshi val loss: {best_val:.4f}. Model: {best_path}")
    print(f"Suhbat: python chat.py --ckpt {best_path}")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()
