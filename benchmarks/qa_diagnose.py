"""Savol-javobda model qayerda yutqazishini aniq o'lchash (Umid va Transformer'ni yonma-yon).

    python -m benchmarks.qa_diagnose --ckpt umid/model.pt gpt/model.pt

Uchta o'lchov (barchasi benchmarks/qa_reference.txt dagi 59 ta nazorat savoli bo'yicha):

1. Bilim (generatsiyaga bog'liq emas): har bir nazorat savoli uchun o'qitishdagi ~300 ta turli javobning
   har birining ehtimoli (teacher forcing, o'rtacha NLL) hisoblanadi. To'g'ri javob 1-o'rinda bo'lsa —
   model javobni "biladi". Greedy generatsiya xatosi va bilim yetishmasligini ajratadi.
2. So'zlar ta'siri (occlusion): savoldan bitta so'z olib tashlansa, to'g'ri javobning NLL'i qanchaga
   oshadi. Pozitsiya bo'yicha (birinchi so'z … oxirgi so'z) o'rtacha — model savolning boshini
   unutayotgan bo'lsa, birinchi so'zlarning ta'siri kichik bo'ladi.
3. Savol tasviri (har qatlam): "Javob:" dagi yashirin holat bo'yicha eng yaqin o'qitish savoli
   to'g'ri javobga ega savolmi (kosinus o'xshashlik). Qaysi qatlamda tushunish shakllanishini ko'rsatadi.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from umid import BPETokenizer, load_model
from umid.data import default_threads, qa_prompt, read_qa_pairs

from .common import RESULTS

HERE = Path(__file__).parent


def load_reference():
    rows = []
    for line in (HERE / "qa_reference.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(tuple(x.strip() for x in line.split("|")))
    return rows


@torch.no_grad()
def batch_nll(model, tok, prompts, answers, bs=64):
    """Har bir (prompt, javob) uchun javob tokenlarining o'rtacha NLL'i."""
    out = []
    for i in range(0, len(prompts), bs):
        seqs, starts = [], []
        for p, a in zip(prompts[i:i + bs], answers[i:i + bs]):
            pi, ai = tok.encode(p), tok.encode(f" {a}") + [tok.eot_id]
            seqs.append(pi + ai)
            starts.append(len(pi))
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), tok.eot_id)
        y = torch.full((len(seqs), T), -100)
        for j, (s, st) in enumerate(zip(seqs, starts)):
            x[j, :len(s)] = torch.tensor(s)
            y[j, st - 1:len(s) - 1] = torch.tensor(s[st:])
        logits, _ = model(x)
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none")
        out += (nll.sum(1) / (y != -100).sum(1)).tolist()
    return out


@torch.no_grad()
def hidden_at_end(model, tok, prompt):
    """Har qatlam chiqishidagi oxirgi token (":" — "Javob:") holati."""
    feats = []
    hooks = [b.register_forward_hook(lambda m, i, o: feats.append(o[0][0, -1].float())) for b in model.blocks]
    model(torch.tensor([tok.encode(prompt)]))
    for h in hooks:
        h.remove()
    return feats


def words(q):
    return re.findall(r"\S+", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", default=None, help="vergul bilan: jadval uchun nomlar")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    default_threads()
    pairs = read_qa_pairs(HERE.parent / "chat-data" / "savol-javob")
    answer_of = dict(pairs)
    distinct = sorted(set(a for _, a in pairs))
    rows = load_reference()
    names = args.names.split(",") if args.names else [Path(c).parent.name for c in args.ckpt]
    report = {}
    for name, path in zip(names, args.ckpt):
        model, ck = load_model(path)
        model.eval()
        tok = BPETokenizer.from_dict(ck["tokenizer"])
        r = {}

        # 1. Bilim: to'g'ri javobning o'rni barcha javoblar orasida
        ranks, margins = [], []
        for q, ref in rows:
            nll = batch_nll(model, tok, [qa_prompt(q)] * len(distinct), distinct)
            correct = nll[distinct.index(answer_of[ref])]
            ranks.append(sum(n < correct for n in nll) + 1)
            margins.append(min(n for n, a in zip(nll, distinct) if a != answer_of[ref]) - correct)
        r["top1"] = sum(k == 1 for k in ranks) / len(ranks)
        r["top5"] = sum(k <= 5 for k in ranks) / len(ranks)
        r["median_rank"] = sorted(ranks)[len(ranks) // 2]
        r["ranks"] = ranks

        # 2. So'zlar ta'siri: pozitsiya bo'yicha (nisbiy: 0 = birinchi so'z, 1 = oxirgi so'z)
        bins = {"bosh": [], "o'rta": [], "oxir": []}
        for q, ref in rows:
            ws = words(q)
            base = batch_nll(model, tok, [qa_prompt(q)], [answer_of[ref]])[0]
            occl = batch_nll(model, tok, [qa_prompt(" ".join(ws[:i] + ws[i + 1:])) for i in range(len(ws))],
                             [answer_of[ref]] * len(ws))
            for i, n in enumerate(occl):
                rel = i / max(len(ws) - 1, 1)
                bins["bosh" if rel < 1 / 3 else "o'rta" if rel <= 2 / 3 else "oxir"].append(n - base)
        r["occlusion"] = {k: sum(v) / max(len(v), 1) for k, v in bins.items()}

        # 3. Savol tasviri: eng yaqin o'qitish savolining javobi to'g'rimi (har qatlam)
        train_q = [q for q, _ in pairs]
        train_feats = [hidden_at_end(model, tok, qa_prompt(q)) for q in train_q]
        per_layer = []
        for li in range(len(model.blocks)):
            M = F.normalize(torch.stack([f[li] for f in train_feats]), dim=-1)
            hits = 0
            for q, ref in rows:
                h = F.normalize(hidden_at_end(model, tok, qa_prompt(q))[li], dim=0)
                hits += answer_of[train_q[int((M @ h).argmax())]] == answer_of[ref]
            per_layer.append(hits / len(rows))
        r["probe_per_layer"] = per_layer
        report[name] = r
        print(f"\n=== {name}")
        print(f"bilim: to'g'ri javob 1-o'rinda {r['top1']:.0%}, top-5 da {r['top5']:.0%}, median o'rin {r['median_rank']}")
        print("so'z olib tashlanganda NLL o'sishi: " + ", ".join(f"{k} {v:.3f}" for k, v in r["occlusion"].items()))
        print("eng yaqin savol bo'yicha to'g'ri (qatlamlar): " + " ".join(f"{v:.0%}" for v in per_layer))
    out = RESULTS / f"qa_diagnose{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaqlandi: {out}")


if __name__ == "__main__":
    main()
