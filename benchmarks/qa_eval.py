"""Savol-javob sifatini halol baholash: o'qitishda BO'LMAGAN savollar (benchmarks/qa_heldout.txt va
qa_heldout2.txt) va solishtirish uchun o'qitishda bo'lgan savollardan namuna.

    python -m benchmarks.qa_eval --ckpt checkpoints/umid/model.pt

Har bir javob greedy rejimda olinadi; javobda kutilgan kalit so'z (yoki muqobillardan biri; "&" bilan
yozilganlarning hammasi) bo'lsa, to'g'ri hisoblanadi. Bu qo'pol, lekin avtomatik va takrorlanadigan o'lchov.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from umid import BPETokenizer, load_model
from umid.data import default_threads, qa_prompt
from umid.model import beam_search, generate

from .common import RESULTS

HERE = Path(__file__).parent


def answer(model, tok, question, max_new_tokens=60, repetition_penalty=1.15, penalize_prompt=True, beams=1):
    ids = torch.tensor([tok.encode(qa_prompt(question))])
    if beams > 1:   # jarima beam'da faqat javob tokenlariga
        out = beam_search(model, ids, max_new_tokens, beams, tok.eot_id, repetition_penalty)
    else:
        out, _ = generate(model, ids, max_new_tokens, temperature=0.0, repetition_penalty=repetition_penalty,
                          eos_id=tok.eot_id, on_token=None, penalize_prompt=penalize_prompt)
    return tok.decode(out[0].tolist()).split("\n")[0].replace("<|endoftext|>", "").strip()


HELDOUT_SETS = ("qa_heldout.txt", "qa_heldout2.txt")


def load_heldout(name="qa_heldout.txt"):
    rows = []
    for line in (HERE / name).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            q, keys = line.split("|")
            rows.append((q.strip(), [k.strip() for k in keys.split("/")]))
    return rows


def is_correct(answer, keys):
    a = answer.lower()
    return any(all(part.strip().lower() in a for part in k.split("&")) for k in keys)


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
    ap.add_argument("--rep-penalty", type=float, default=1.15, help="takrorlash jarimasi (1.0 — yo'q)")
    ap.add_argument("--penalty-answer-only", action="store_true", help="jarima faqat javob tokenlariga")
    ap.add_argument("--beams", type=int, default=1, help=">1 — beam search (greedy o'rniga)")
    args = ap.parse_args()
    default_threads()
    model, ck = load_model(args.ckpt)
    model.eval()
    tok = BPETokenizer.from_dict(ck["tokenizer"])

    report = {"ckpt": args.ckpt, "rep_penalty": args.rep_penalty, "penalty_answer_only": args.penalty_answer_only,
              "beams": args.beams,
              "heldout": [], "seen": []}
    for name in HELDOUT_SETS:
        hits = 0
        print(f"=== O'qitishda bo'lmagan savollar ({name}) ===")
        rows = load_heldout(name)
        for q, keys in rows:
            a = answer(model, tok, q, repetition_penalty=args.rep_penalty, penalize_prompt=not args.penalty_answer_only,
                       beams=args.beams)
            ok = is_correct(a, keys)
            hits += ok
            report["heldout"].append({"toplam": name, "savol": q, "javob": a, "kalit": keys, "togri": ok})
            print(f"{'✓' if ok else '✗'} {q}\n    → {a}")
        report[f"accuracy_{name.removesuffix('.txt')}"] = hits / len(rows)
        print(f"\n{name}: {hits}/{len(rows)} = {hits / len(rows):.0%}\n")
    total = report["heldout"]
    report["heldout_accuracy"] = sum(r["togri"] for r in total) / len(total)
    print(f"Jami o'qitishda bo'lmagan savollar: {sum(r['togri'] for r in total)}/{len(total)} = "
          f"{report['heldout_accuracy']:.0%}")

    exact = 0
    print("\n=== O'qitishda bo'lgan savollardan namuna ===")
    for q, ref in load_seen(args.seen):
        a = answer(model, tok, q, repetition_penalty=args.rep_penalty, penalize_prompt=not args.penalty_answer_only,
                       beams=args.beams)
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
