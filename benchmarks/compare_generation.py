"""Umid va Transformer javoblarini yonma-yon ko'rsatish.

    python -m benchmarks.compare_generation --umid-ckpt checkpoints/umid/model.pt

Transformer Umid checkpoint'i bilan bir xil sharoitda o'qitiladi (yoki --gpt-ckpt dan yuklanadi):
bir xil tokenizer, train/val bo'linishi, o'lcham (d, qatlamlar, head'lar), kontekst, qadamlar,
batch va seed. Adolat uchun bir nechta lr sinaladi va val bo'yicha eng yaxshisi olinadi.
Keyin ikkala model bir xil promptlardan bir xil sampling sozlamalari bilan javob yozadi.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from umid import BPETokenizer, UmidLM
from umid.baseline import GPT, GPTConfig
from umid.data import cosine_lr, default_threads, evaluate, prepare_tokens, random_batch, split_tokens
from umid.model import generate

from .common import RESULTS, make_optimizer

PROMPTS = [
    "Python dasturlash tili",
    "Bobur Hayitov Sun'iy intellekt nima AI SI",
    "Bir kuni kampir",
    "Inson hayotida eng muhim narsa",
    "Salom",
]


def train_gpt(cfg, train_tok, val_tok, args, lr):
    torch.manual_seed(args.seed)
    model = GPT(cfg)
    opt = make_optimizer(model, lr)
    g = torch.Generator().manual_seed(args.seed)
    best = {"val": float("inf"), "state": None, "step": 0}
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for gr in opt.param_groups:
            gr["lr"] = cosine_lr(step, args.steps, lr, args.warmup)
        x, y = random_batch(train_tok, args.batch_size, args.seq_len, g)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            val = evaluate(model, val_tok, args.seq_len)
            print(f"  [gpt lr={lr:g}] qadam {step + 1}/{args.steps}  train {loss.item():.3f}  val {val:.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
            if val < best["val"]:
                best.update(val=val, step=step + 1, state=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best["state"])
    return model, best


@torch.no_grad()
def positional_loss(model, val_tok, length, spans):
    """Uzun oynalarda turli pozitsiyalardagi o'rtacha loss (masalan 0–256 va 1792–2048)."""
    model.eval()
    n = (len(val_tok) - 1) // length
    totals = {s: 0.0 for s in spans}
    for i in range(n):
        x = val_tok[i * length:(i + 1) * length][None]
        y = val_tok[i * length + 1:(i + 1) * length + 1][None]
        logits, _ = model(x)
        nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
        for a, b in spans:
            totals[(a, b)] += nll[a:b].mean().item() / n
    return {f"{a}-{b}": round(v, 4) for (a, b), v in totals.items()}


@torch.no_grad()
def document_loss(model, val_tok, window, stateful, block=8192):
    """Val matnini hujjat sifatida boshidan oxirigacha o'qib, har bir token uchun loss.

    stateful=True (Umid): matn `window` lik bo'laklarda o'qiladi, xotira bo'lakdan bo'lakka
    uzatiladi — model butun oldingi matnni "eslaydi".
    stateful=False (Transformer): siljuvchi oyna — har bir token o'zidan oldingi ko'pi bilan
    `window` token kontekst bilan bashorat qilinadi (o'qitilgan uzunlikdan oshmaydi).
    Val bloklari korpusning turli joylaridan olingani uchun har blok alohida hujjat.
    """
    model.eval()
    total, count = 0.0, 0
    for b0 in range(0, len(val_tok) - 1, block):
        doc = val_tok[b0:b0 + block + 1]
        x_all, y_all = doc[:-1], doc[1:]
        if stateful:
            states = None
            for s in range(0, len(x_all), window):
                logits, _, states = model(x_all[s:s + window][None], states=states, return_states=True)
                nll = F.cross_entropy(logits[0].float(), y_all[s:s + window], reduction="sum")
                total, count = total + nll.item(), count + y_all[s:s + window].numel()
        else:
            stride = window // 2
            for s in range(0, len(x_all), stride):
                lo = max(0, s + stride - window)
                logits, _ = model(x_all[lo:s + stride][None])
                keep = slice(s - lo, None)
                nll = F.cross_entropy(logits[0, keep].float(), y_all[s:s + stride], reduction="sum")
                total, count = total + nll.item(), count + y_all[s:s + stride].numel()
    return round(total / count, 4)


@torch.no_grad()
def generate_long(model, ids, n_new, window, stateful, temperature, seed):
    """Uzun javob. Umid — butun holat bilan; Transformer — oxirgi `window` token bilan (siljuvchi
    oyna, har qadamda qayta hisoblash: RoPE pozitsiyalari doim o'qitilgan oraliqda qoladi)."""
    from umid.model import sample_next
    torch.manual_seed(seed)
    model.eval()
    seq = ids
    if stateful:
        logits, _, states = model(ids, return_states=True)
    else:
        logits, _ = model(ids[:, -window:])
    for _ in range(n_new):
        nxt = sample_next(logits[:, -1], seq, temperature, 50, 0.9, 1.15)
        seq = torch.cat([seq, nxt], 1)
        if stateful:
            logits, _, states = model(nxt, states=states, return_states=True)
        else:
            logits, _ = model(seq[:, -window:])
    return seq[0, ids.size(1):].tolist()


def word_stats(text, vocab_words):
    words = [w.strip(".,!?;:\"'«»()[]").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return {"valid": 0.0, "distinct2": 0.0}
    valid = sum(w in vocab_words for w in words) / len(words)
    bigrams = list(zip(words, words[1:]))
    distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
    return {"valid": round(valid, 3), "distinct2": round(distinct2, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umid-ckpt", default="checkpoints/umid/model.pt")
    ap.add_argument("--gpt-ckpt", default="checkpoints/gpt_baseline/model.pt")
    ap.add_argument("--data", default="chat-data/Data.txt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lrs", default="1e-3,2e-3")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--long-context", type=int, default=1536)
    ap.add_argument("--long-seeds", type=int, default=3)
    args = ap.parse_args()
    default_threads()

    umid, ck = UmidLM.load(args.umid_ckpt)
    umid.eval()
    tok_path = Path(args.umid_ckpt).parent / "tokenizer.json"
    if not tok_path.exists():
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        BPETokenizer.from_dict(ck["tokenizer"]).save(tok_path)
    tok, tokens = prepare_tokens(args.data, tok_path, verbose=False)
    train_tok, val_tok = split_tokens(tokens, 0.1)

    gcfg = GPTConfig(vocab_size=tok.vocab_size, d_model=umid.cfg.d_model, n_layers=umid.cfg.n_layers,
                     n_heads=umid.cfg.n_heads, dropout=args.dropout)
    gpt_path = Path(args.gpt_ckpt)
    if gpt_path.exists():
        saved = torch.load(gpt_path, weights_only=False)
        gpt = GPT(GPTConfig(**saved["config"]))
        gpt.load_state_dict(saved["model"])
        gpt_info = saved["info"]
        print(f"Transformer yuklandi: {gpt_path} ({gpt_info})")
    else:
        print("Transformer bir xil sharoitda o'qitilmoqda...")
        runs = {}
        for lr in (float(x) for x in args.lrs.split(",")):
            runs[lr] = train_gpt(gcfg, train_tok, val_tok, args, lr)
        lr, (gpt, best) = min(runs.items(), key=lambda kv: kv[1][1]["val"])
        gpt_info = {"lr": lr, "best_step": best["step"], "lr_sweep": {str(k): round(v[1]["val"], 4) for k, v in runs.items()}}
        gpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(gcfg), "model": gpt.state_dict(), "info": gpt_info}, gpt_path)
        print(f"Eng yaxshi Transformer: lr={lr:g}, val {best['val']:.4f}")
    gpt.eval()

    models = {"Umid": umid, "Transformer": gpt}
    report = {"settings": vars(args), "transformer": gpt_info,
              "params": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
              "val_loss": {}, "positional_loss_2048": {}, "generations": []}
    for name, m in models.items():
        report["val_loss"][name] = {L: round(evaluate(m, val_tok, L), 4) for L in (256, 1024, 2048)}
        report["positional_loss_2048"][name] = positional_loss(m, val_tok, 2048, [(0, 256), (1792, 2048)])
    report["document_loss"] = {
        "Umid (butun hujjat, xotira uzatiladi)": document_loss(umid, val_tok, args.seq_len, stateful=True),
        "Umid (faqat 256 tokenli siljuvchi oyna)": document_loss(umid, val_tok, args.seq_len, stateful=False),
        "Transformer (256 tokenli siljuvchi oyna)": document_loss(gpt, val_tok, args.seq_len, stateful=False),
    }
    print("\nHujjat bo'yicha loss:", json.dumps(report["document_loss"], ensure_ascii=False, indent=1))

    vocab_words = {w.strip(".,!?;:\"'«»()[]").lower() for w in tok.decode(train_tok.tolist()).split()}
    long_stats = {name: {"boshi (0-250)": [], "oxiri (450-700)": []} for name in models}
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt)])
        for seed in range(args.long_seeds):
            for name, m in models.items():
                new = generate_long(m, ids, 700, args.seq_len, name == "Umid", args.temperature, seed)
                for part, sl in (("boshi (0-250)", slice(0, 250)), ("oxiri (450-700)", slice(450, 700))):
                    long_stats[name][part].append(word_stats(tok.decode(new[sl]), vocab_words))
    report["long_generation"] = {
        name: {part: {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in ("valid", "distinct2")}
               for part, rows in parts.items()} for name, parts in long_stats.items()}
    print("\nUzun javob (700 token): yaroqli so'zlar ulushi va takrorlanmaslik (distinct-2):")
    print(json.dumps(report["long_generation"], ensure_ascii=False, indent=1))

    print("\nval loss (past = yaxshi):", json.dumps(report["val_loss"], ensure_ascii=False))
    print("2048 tokenli oynada pozitsiya bo'yicha loss:", json.dumps(report["positional_loss_2048"], ensure_ascii=False))

    def gen(m, ids, temperature, seed):
        torch.manual_seed(seed)
        out, _ = generate(m, ids, args.max_new_tokens, temperature=temperature, top_k=50, top_p=0.9,
                          repetition_penalty=1.15)
        return tok.decode(out[0].tolist())

    long_ids = val_tok[: args.long_context]
    cases = [(p, torch.tensor([tok.encode(p)])) for p in PROMPTS]
    cases.append((f"[val matnidan {args.long_context} tokenli kontekst] …" + tok.decode(long_ids[-25:].tolist()),
                  long_ids[None]))
    for prompt, ids in cases:
        row = {"prompt": prompt, "prompt_tokens": ids.size(1)}
        for name, m in models.items():
            row[name] = {"greedy": gen(m, ids, 0.0, 0), "sampled": gen(m, ids, args.temperature, 0)}
        report["generations"].append(row)
        print(f"\n### {prompt}  ({ids.size(1)} token)")
        for name in models:
            print(f"  {name:<12} greedy : {row[name]['greedy']!r}")
            print(f"  {name:<12} sampled: {row[name]['sampled']!r}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "generation_comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nNatijalar saqlandi: {out}")


if __name__ == "__main__":
    main()
