"""Savol-javob sifatini halol baholash: o'qitishda BO'LMAGAN savollar (benchmarks/qa_heldout.txt)
va solishtirish uchun o'qitishda bo'lgan savollardan namuna.

    python -m benchmarks.qa_eval --ckpt checkpoints/umid/model.pt

Har bir javob greedy rejimda olinadi; javobda kutilgan kalit so'z (yoki muqobillardan biri) bo'lsa,
to'g'ri hisoblanadi. Bu qo'pol, lekin avtomatik va takrorlanadigan o'lchov.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from umid import BPETokenizer, UmidLM
from umid.data import default_threads
from umid.model import generate

from .common import RESULTS

HERE = Path(__file__).parent


def answer(model, tok, question, max_new_tokens=60):
    ids = torch.tensor([tok.encode(f"Savol: {question}\nJavob:")])
    out, _ = generate(model, ids, max_new_tokens, temperature=0.0, repetition_penalty=1.15, eos_id=tok.eot_id,
                      on_token=None)
    return tok.decode(out[0].tolist()).split("\n")[0].replace("<|endoftext|>", "").strip()


def load_heldout():
    rows = []
    for line in (HERE / "qa_heldout.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            q, keys = line.split("|")
            rows.append((q.strip(), [k.strip() for k in keys.split("/")]))
    return rows


def load_seen(n, seed=0):
    pairs = []
    for f in sorted((HERE.parent / "chat-data" / "savol-javob").glob("*.txt")):
        for block in f.read_text(encoding="utf-8").split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) == 2 and lines[0].startswith("Savol:"):
                pairs.append((lines[0][6:].strip(), lines[1][6:].strip()))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/umid/model.pt")
    ap.add_argument("--seen", type=int, default=15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    default_threads()
    model, ck = UmidLM.load(args.ckpt)
    model.eval()
    tok = BPETokenizer.from_dict(ck["tokenizer"])

    report = {"ckpt": args.ckpt, "heldout": [], "seen": []}
    hits = 0
    print("=== O'qitishda bo'lmagan savollar ===")
    for q, keys in load_heldout():
        a = answer(model, tok, q)
        ok = any(k.lower() in a.lower() for k in keys)
        hits += ok
        report["heldout"].append({"savol": q, "javob": a, "kalit": keys, "togri": ok})
        print(f"{'✓' if ok else '✗'} {q}\n    → {a}")
    n = len(report["heldout"])
    report["heldout_accuracy"] = hits / n
    print(f"\nO'qitishda bo'lmagan savollar: {hits}/{n} = {hits / n:.0%}")

    exact = 0
    print("\n=== O'qitishda bo'lgan savollardan namuna ===")
    for q, ref in load_seen(args.seen):
        a = answer(model, tok, q)
        same = a.strip() == ref.strip()
        exact += same
        report["seen"].append({"savol": q, "javob": a, "asl": ref, "aynan": same})
        print(f"{'=' if same else '≈'} {q}\n    → {a}")
    report["seen_exact"] = exact / max(args.seen, 1)
    print(f"\nO'qitishdagi savollar, aynan qaytarildi: {exact}/{args.seen}")
    out = RESULTS / f"qa_eval{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Natijalar saqlandi: {out}")


if __name__ == "__main__":
    main()
