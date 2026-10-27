"""Umid LLM — hammasi bitta buyruqda: ma'lumot (tozalash bilan) → tokenizer → pretraining → sekin SFT →
baholash → namunaviy javoblar. Colab/Kaggle'da GPU bilan:

    python pipeline.py --save /content/drive/MyDrive/umid --preset small --mixer tanlov

Har bosqich tugagan bo'lsa qayta bajarilmaydi, pretraining uzilgan joyidan davom etadi (--resume),
shuning uchun sessiya uzilsa xuddi shu buyruqni qayta ishga tushirish kifoya.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEMO = ["Salom", "Sen kimsan?", "O'zbekistonning poytaxti qaysi?", "Kitob o'qishning foydasi nima?",
        "Sog'lom bo'lish uchun nima qilish kerak?", "Amir Temur kim bo'lgan?"]


def run(*cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, "-u", *map(str, cmd)], check=True)


def main():
    ap = argparse.ArgumentParser(description="Umid LLM: to'liq o'qitish jarayoni bitta buyruqda")
    ap.add_argument("--save", default="checkpoints/pipeline", help="natijalar papkasi (Drive yoki /kaggle/working)")
    ap.add_argument("--data", default="data", help="tayyorlangan ma'lumot papkasi")
    ap.add_argument("--mixer", default="tanlov", choices=["tanlov", "delta"], help="tanlov — yangi algoritm")
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base", "large"])
    ap.add_argument("--sources", default="wiki,news,local")
    ap.add_argument("--sft-sources", default="alpaca,instruct")
    ap.add_argument("--max-mb", type=int, default=600, help="tez sinov uchun 50")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sft-steps", type=int, default=6000, help="SFT sekin va ehtiyotkor: ko'p qadam,")
    ap.add_argument("--sft-lr", type=float, default=1e-4, help="kichik o'rganish tezligi,")
    ap.add_argument("--sft-warmup", type=int, default=300, help="uzun qizish bosqichi")
    ap.add_argument("--compare-gpt", action="store_true", help="xuddi shu sharoitda Transformer ham o'qitilsin")
    args = ap.parse_args()

    save, data = Path(args.save), Path(args.data)
    save.mkdir(parents=True, exist_ok=True)

    # 1) Ma'lumot: yuklash, tozalash (umid/clean.py), tokenizer, tokenlar, SFT juftliklari
    if not (data / "tokens.npy").exists():
        run("prepare_data.py", "--out", data, "--sources", args.sources, "--max-mb", args.max_mb,
            "--vocab-size", args.vocab_size, "--tok-sample-mb", 20, "--sft", args.sft_sources)
    else:
        print(f"1) {data / 'tokens.npy'} tayyor — o'tkazib yuborildi")

    qa = "chat-data/savol-javob" + (f",{data / 'sft'}" if (data / "sft").exists() else "")
    archs = [("umid", args.mixer)] + ([("gpt", None)] if args.compare_gpt else [])
    for arch, mixer in archs:
        name = "base" if arch == "umid" else "gpt_base"
        chat = save / ("chat" if arch == "umid" else "gpt_chat")
        # 2) Pretraining (uzilsa davom etadi)
        extra = (["--arch", "gpt"] if arch == "gpt" else
                 ["--mixer", mixer] + (["--select-no-conv"] if mixer == "tanlov" else ["--persistent-heads", 3]))
        run("train.py", "--tokens", data / "tokens.npy", "--tokenizer", data / "tokenizer.json", "--out", save / name,
            "--preset", args.preset, *extra, "--reset-on-eot", "--seq-len", args.seq_len, "--batch-size",
            args.batch_size, "--steps", args.steps, "--lr", args.lr, "--warmup", 500, "--dropout", 0.0,
            "--eval-every", 1000, "--eval-windows", 200, "--sample-prompt", "O'zbekiston", "--resume")
        # 3) Sekin SFT
        if not (chat / "model.pt").exists():
            run("sft.py", "--init", save / name / "model.pt", "--out", chat, "--qa", qa, "--steps", args.sft_steps,
                "--seq-len", args.seq_len, "--batch-size", args.batch_size, "--lr", args.sft_lr,
                "--warmup", args.sft_warmup, "--dialog-frac", 0.25)
        else:
            print(f"3) {chat / 'model.pt'} tayyor — SFT o'tkazib yuborildi (qayta qilish uchun o'chiring)")
        # 4) Baholash: o'qitishda bo'lmagan 60 savol
        run("-m", "benchmarks.qa_eval", "--ckpt", chat / "model.pt", "--rep-penalty", 1.0,
            "--device", "cuda" if _cuda() else "cpu", "--tag", f"pipeline_{arch}")

    # 5) Namunaviy javoblar
    demo(save / "chat" / "model.pt")
    print(f"\nTayyor. Suhbat: python chat.py --ckpt {save / 'chat' / 'model.pt'}")


def _cuda():
    import torch
    return torch.cuda.is_available()


def demo(ckpt):
    import torch

    from umid import BPETokenizer, load_model
    from umid.data import qa_prompt
    from umid.model import generate
    device = "cuda" if _cuda() else "cpu"
    model, ck = load_model(ckpt)
    model.to(device).eval()
    tok = BPETokenizer.from_dict(ck["tokenizer"])
    print(f"\n=== Namunaviy javoblar ({model.num_params() / 1e6:.1f}M parametr) ===")
    for q in DEMO:
        ids = torch.tensor([tok.encode(qa_prompt(q))], device=device)
        out, _ = generate(model, ids, 120, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.15,
                          eos_id=tok.eot_id, penalize_prompt=False,
                          on_token=lambda t: "\n" in tok.decode([int(t[0, 0])]))
        print(f"Savol: {q}\nUmid:  {tok.decode(out[0].tolist()).split(chr(10))[0].replace('<|endoftext|>', '').strip()}\n")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)   # skriptlar va chat-data shu papkaga nisbatan
    main()
