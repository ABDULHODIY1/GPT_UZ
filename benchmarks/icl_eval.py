"""Suhbatdan o'rganish: model suhbatda aytilgan YANGI faktni (o'qitishda hech uchramagan ism, shahar,
raqam, o'ylab topilgan so'z va uning ta'rifi) oradan boshqa savollar o'tgach eslay oladimi.

    python -m benchmarks.icl_eval --ckpt umid/model.pt gpt/model.pt

Dialoglar umid/dialog.py dan "test" qiymatlari bilan (o'qitishdagilar bilan kesishmaydi). Har bir
masofa (oradagi boshqa savol-javoblar soni) uchun bir xil dialoglar barcha modellarga beriladi.
Oldingi navbatlar to'g'ri javoblar bilan beriladi (teacher forcing), model faqat tekshiruv savollariga
o'zi javob yozadi (greedy). Javobda kutilgan qiymat bo'lsa — to'g'ri.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from umid import BPETokenizer, load_model
from umid.data import default_threads, qa_prompt, read_qa_pairs
from umid.dialog import make_dialog
from umid.model import generate

from .common import RESULTS

HERE = Path(__file__).parent


@torch.no_grad()
def run_dialog(model, tok, turns, checks, max_new_tokens=40):
    """Qaytaradi: [(kutilgan, javob, to'g'ri, tekshiruv savoligacha tokenlar soni)]."""
    check_at = dict(checks)
    states, results, n_tok = None, [], 0
    nl = tok.encode("\n")
    stop = lambda t: "\n" in tok.decode([int(t[0, 0])])  # noqa: E731
    for i, (q, a) in enumerate(turns):
        prompt = torch.tensor([tok.encode(qa_prompt(q))])
        if i in check_at:   # model o'zi javob yozadi (holat o'zgarmaydi — u keyin to'g'ri javob bilan davom etadi)
            new, _ = generate(model, prompt, max_new_tokens, temperature=0.0, repetition_penalty=1.0,
                              eos_id=tok.eot_id, states=states, on_token=stop)
            out = tok.decode(new[0].tolist()).split("\n")[0].replace("<|endoftext|>", "").strip()
            results.append((check_at[i], out, check_at[i].lower() in out.lower(), n_tok))
        ids = torch.tensor([tok.encode(qa_prompt(q)) + tok.encode(f" {a}") + nl])
        _, _, states = model(ids, states=states, return_states=True)
        n_tok += ids.size(1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--distances", default="0,2,5,10,20,40", help="oradagi boshqa savol-javoblar soni")
    ap.add_argument("--n", type=int, default=40, help="har masofa uchun dialoglar soni")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    default_threads()
    fillers = read_qa_pairs(HERE.parent / "chat-data" / "savol-javob")
    dists = [int(d) for d in args.distances.split(",")]
    dialogs = {d: [make_dialog(random.Random(args.seed * 1000 + d * 100 + i), "test", fillers, n_fill=d)
                   for i in range(args.n)] for d in dists}
    names = args.names.split(",") if args.names else [Path(c).parent.name for c in args.ckpt]
    report = {"distances": dists, "n": args.n, "models": {}}
    for name, path in zip(names, args.ckpt):
        model, ck = load_model(path)
        model.eval()
        tok = BPETokenizer.from_dict(ck["tokenizer"])
        rows, examples = {}, []
        for d in dists:
            res = [r for turns, checks in dialogs[d] for r in run_dialog(model, tok, turns, checks)]
            rows[d] = {"accuracy": sum(r[2] for r in res) / len(res),
                       "mean_tokens": sum(r[3] for r in res) / len(res)}
            examples += [{"masofa": d, "kutilgan": r[0], "javob": r[1], "togri": r[2]} for r in res[:3]]
        report["models"][name] = {"by_distance": rows, "examples": examples}
        print(f"{name:24s} " + "  ".join(f"d={d}: {rows[d]['accuracy']:.0%} (~{rows[d]['mean_tokens']:.0f} tok)"
                                          for d in dists), flush=True)
    out = RESULTS / f"icl_eval{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saqlandi: {out}")


if __name__ == "__main__":
    main()
