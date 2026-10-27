"""Savol-javob va suhbat bo'yicha fine-tuning (SFT): loss faqat javob tokenlarida hisoblanadi.

    python train.py --out checkpoints/umid_base                         # 1) til: umumiy matn + savol-javoblar
    python sft.py --init checkpoints/umid_base/model.pt --out checkpoints/umid   # 2) suhbat

Pretraining (train.py) tilni o'rgatadi; bu bosqich modelni savolga javob berishga yo'naltiradi:
har bir misol "Savol: …\\nJavob:" (loss yo'q) + " javob" + EOT (loss bor). Tilni unutmaslik uchun
qatorlarning bir qismi (--mix-frac) umumiy matndan olinadi va to'liq loss bilan o'qitiladi.
--dialogs: suhbatdan o'rganish uchun sintetik dialoglar (umid/dialog.py) — foydalanuvchi aytgan yangi
faktni (ism, o'ylab topilgan so'z, raqam) oradagi boshqa savollardan keyin eslab javob berish.
Umid va taqqoslash uchun Transformer (train.py --arch gpt) checkpoint'lari bir xil ishlaydi.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from umid import BPETokenizer, load_model
from umid.data import (cosine_lr, default_threads, encode_dialog, encode_qa, prepare_tokens, qa_batch,
                       random_batch, read_qa_pairs, split_tokens)
from umid.dialog import make_dialog


def parse_args():
    ap = argparse.ArgumentParser(description="Umid: savol-javob fine-tuning")
    ap.add_argument("--init", required=True, help="pretraining checkpoint'i (train.py natijasi)")
    ap.add_argument("--out", default="checkpoints/umid")
    ap.add_argument("--qa", default="chat-data/savol-javob", help="savol-javob fayllari (papka yoki ro'yxat)")
    ap.add_argument("--mix", default="chat-data/Data.txt!", help="tilni unutmaslik uchun umumiy matn ('' — yo'q)")
    ap.add_argument("--mix-frac", type=float, default=0.25, help="umumiy matndan olinadigan qatorlar ulushi")
    ap.add_argument("--dialogs", type=int, default=0, help="sintetik suhbatlar soni (o'qitish qiymatlari bilan)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    args = parse_args()
    default_threads()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, init_ckpt = load_model(args.init)
    tok = BPETokenizer.from_dict(init_ckpt["tokenizer"])
    model.cfg.dropout = args.dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = args.dropout

    pairs = read_qa_pairs(args.qa)
    examples = encode_qa(tok, pairs)
    rng, n_dialogs = random.Random(args.seed), 0
    while n_dialogs < args.dialogs:
        ids, mask = encode_dialog(tok, make_dialog(rng, "train", pairs)[0])
        if len(ids) <= args.seq_len + 1:   # tekshiruv savollari qirqilib qolmasin
            examples.append((ids, mask))
            n_dialogs += 1
    mix_tok = None
    if args.mix and args.mix_frac > 0:
        tok_path = out / "tokenizer.json"
        tok.save(tok_path)
        mix_tok, _ = split_tokens(prepare_tokens(args.mix, tok_path, verbose=False)[1])
    n_mix = round(args.batch_size * args.mix_frac) if mix_tok is not None else 0
    print(f"Misollar: {len(examples)} (shundan {n_dialogs} suhbat) | har qatorda {args.batch_size - n_mix} savol-javob, "
          f"{n_mix} umumiy matn | {model.num_params() / 1e6:.2f}M parametr")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)
    model.train()
    t0, recent = time.time(), []
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y = qa_batch(examples, args.batch_size - n_mix, args.seq_len, g)
        if n_mix:
            mx, my = random_batch(mix_tok, n_mix, args.seq_len, g)
            x, y = torch.cat([x, mx]), torch.cat([y, my])
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        recent = (recent + [loss.item()])[-20:]
        if (step + 1) % 20 == 0:
            print(f"qadam {step + 1:>5}/{args.steps} | loss {sum(recent) / len(recent):.4f} | lr {lr:.2e} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    path = out / "model.pt"
    model.save(path, tokenizer=tok.to_dict(), sft_args=vars(args), init=args.init)
    print(f"Tayyor: {path}\nSuhbat: python chat.py --ckpt {path}")


if __name__ == "__main__":
    main()
