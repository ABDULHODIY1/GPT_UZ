"""Umid bilan suhbat (yoki bir martalik generatsiya).

    python chat.py --ckpt checkpoints/umid/model.pt
    python chat.py --ckpt checkpoints/umid/model.pt --prompt "Bir kuni" --max-new-tokens 200

Suhbat davomida model butun suhbatni qat'iy o'lchamli xotirada saqlaydi: 1000-xabarda ham
har bir token uchun xotira va tezlik birinchi xabardagidek (KV-cache o'smaydi). Siz aytgan yangi
faktlarni (ismingiz, yangi so'z ta'rifi) shu xotiraga yozib, keyingi savollarda ishlatadi
(sft.py --dialogs bilan o'qitilgan bo'lsa).

Buyruqlar: /yangi — xotirani tozalash, /xotira — xotira holati, /chiqish — chiqish.
"""

from __future__ import annotations

import argparse
import sys

import torch

from umid import BPETokenizer, load_model
from umid.data import qa_prompt
from umid.model import generate


def load(ckpt_path, device):
    model, ckpt = load_model(ckpt_path, map_location=device)
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    return model.to(device).eval(), tok


class Streamer:
    """Tokenlarni darhol chiqaradi; UTF-8 belgining yarmi kelganda kutib turadi."""

    def __init__(self, tok, stop_at_newline):
        self.tok, self.stop_at_newline = tok, stop_at_newline
        self.ids, self.printed = [], 0

    def __call__(self, token):
        self.ids.append(int(token[0, 0]))
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return False
        new = text[self.printed:]
        if self.stop_at_newline and "\n" in new:
            new = new[:new.index("\n")]
            sys.stdout.write(new)
            sys.stdout.flush()
            return True
        sys.stdout.write(new)
        sys.stdout.flush()
        self.printed = len(text)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/umid/model.pt")
    ap.add_argument("--prompt", default=None, help="bir martalik generatsiya")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok = load(args.ckpt, args.device)
    gen_kw = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                  repetition_penalty=args.repetition_penalty, eos_id=tok.eot_id)

    if args.prompt is not None:
        print(args.prompt, end="", flush=True)
        ids = torch.tensor([tok.encode(args.prompt)], device=args.device)
        generate(model, ids, args.max_new_tokens, on_token=Streamer(tok, False), **gen_kw)
        print()
        return

    print(f"Umid AI ({model.num_params() / 1e6:.1f}M parametr). Buyruqlar: /yangi, /xotira, /chiqish")
    states, n_tokens, line_start = None, 0, True
    while True:
        try:
            user = input("\nSiz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/chiqish", "exit"):
            print("Xayr! Siz bilan suhbat qurish yoqimli bo'ldi.")
            break
        if user == "/yangi":
            states, n_tokens, line_start = None, 0, True
            print("Xotira tozalandi.")
            continue
        if user == "/xotira":
            size = type(model).state_bytes(states) if states else 0
            print(f"Suhbatda {n_tokens} token o'qildi; xotira hajmi {size / 1024:.1f} KB.")
            continue
        # O'qitish formati (sft.py): "Savol: …\nJavob: …\n" navbatlari, suhbat oxirida EOT. Javob qator oxiri
        # bilan tugamagan bo'lsa (token chegarasi), keyingi savol oldidan "\n" qo'yiladi. EOT xotirani tozalaydi.
        ids = ([] if line_start else tok.encode("\n")) + tok.encode(qa_prompt(user))
        ids = torch.tensor([ids], device=args.device)
        print("Umid: ", end="", flush=True)
        new, states = generate(model, ids, args.max_new_tokens, states=states,
                               on_token=Streamer(tok, True), **gen_kw)
        line_start = new.size(1) == 0 or int(new[0, -1]) == tok.eot_id or tok.decode(new[0, -1:].tolist()).endswith("\n")
        n_tokens += ids.size(1) + new.size(1)
        print()


if __name__ == "__main__":
    main()
