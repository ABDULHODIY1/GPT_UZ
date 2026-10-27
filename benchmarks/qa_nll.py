"""Savol-javob umumlashtirishining uzluksiz o'lchovi (qa_eval'dagi 60 ta ha/yo'q baholashdan kamroq shovqinli).

    python -m benchmarks.qa_nll --ckpt a/model.pt b/model.pt

Har bir nazorat savoli uchun to'g'ri javob — o'qitishdagi mos savolning javobi (benchmarks/qa_reference.txt).
O'lchanadi (nat/token, teacher forcing):
  nazorat  — javobning o'rtacha NLL'i, savol BOSHQACHA so'zlar bilan berilganda (umumlashtirish);
  asl      — xuddi shu javob, o'qitishdagi asl savol bilan (yodlash);
  boshi    — javobning birinchi 4 tokeni (qaysi faktni tanlash) nazorat savolida.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from umid import BPETokenizer, load_model
from umid.data import default_threads, qa_prompt, qa_repeat, read_qa_pairs

HERE = Path(__file__).parent


@torch.no_grad()
def answer_nll(model, tok, question, answer, repeat=1):
    prompt, ans = tok.encode(qa_prompt(question, repeat)), tok.encode(f" {answer}") + [tok.eot_id]
    ids = torch.tensor([prompt + ans])
    logits, _ = model(ids[:, :-1])
    nll = F.cross_entropy(logits[0, len(prompt) - 1:].float(), ids[0, len(prompt):], reduction="none")
    return nll.mean().item(), nll[:4].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    args = ap.parse_args()
    default_threads()
    answers = dict(read_qa_pairs(HERE.parent / "chat-data" / "savol-javob"))
    rows = [tuple(x.strip() for x in line.split("|")) for line in (HERE / "qa_reference.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    print(f"{'model':40s} {'nazorat':>8s} {'asl':>8s} {'boshi':>8s}")
    for path in args.ckpt:
        model, ck = load_model(path)
        model.eval()
        tok = BPETokenizer.from_dict(ck["tokenizer"])
        held = [answer_nll(model, tok, q, answers[ref], qa_repeat(ck)) for q, ref in rows]
        seen = [answer_nll(model, tok, ref, answers[ref], qa_repeat(ck))[0] for _, ref in rows]
        n = len(rows)
        print(f"{Path(path).parent.name:40s} {sum(h[0] for h in held) / n:8.3f} {sum(seen) / n:8.3f} "
              f"{sum(h[1] for h in held) / n:8.3f}")


if __name__ == "__main__":
    main()
