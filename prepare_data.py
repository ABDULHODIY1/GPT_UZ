"""Katta o'zbekcha korpusni tayyorlash (Colab/Kaggle uchun): yuklab olish → filtrlash → tokenizer → tokenlar.

    python prepare_data.py --out data --sources wiki,news,local --max-mb 600 --vocab-size 16000 --sft alpaca,instruct

Natija:
    data/tokenizer.json        — BPE tokenizer (namunada o'qitiladi)
    data/tokens.npy            — barcha hujjatlar tokenlari (int32, hujjatlar orasida EOT) → train.py --tokens
    data/sft/*.txt             — savol-javob formatidagi ko'rsatmalar → sft.py --qa

Manbalar (Hugging Face, `pip install datasets`):
    wiki     — wikimedia/wikipedia, 20231101.uz (CC BY-SA)
    news     — MLDataScientist/Uzbek_news_dataset (CC BY 4.0, ~120 mln so'z)
    local    — chat-data/Data.txt (tozalangan) va chat-data/savol-javob (×--qa-repeat)
    yo'l     — istalgan .txt fayl yoki papka (masalan --sources wiki,mening_matnlarim/)
SFT:
    alpaca   — saillab/alpaca-uzbek-cleaned (instruction/input/output)
    instruct — UAzimov/uzbek-instruct-llm (messages; Apache-2.0)
Litsenziyalarni manba sahifasida tekshiring.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from array import array
from pathlib import Path

from umid.data import clean_text, read_corpus
from umid.tokenizer import EOT, BPETokenizer

_LATIN = re.compile(r"[A-Za-zʻʼ‘’'`]")
_CYR = re.compile(r"[Ѐ-ӿ]")
_SPACE = re.compile(r"[ \t ]+")


def good_doc(text, min_chars=200, max_cyrillic=0.2):
    """Juda qisqa va asosan kirillcha hujjatlarni tashlaydi (kichik model bitta yozuvni yaxshiroq o'rganadi)."""
    if len(text) < min_chars:
        return False
    lat, cyr = len(_LATIN.findall(text[:2000])), len(_CYR.findall(text[:2000]))
    return cyr <= max_cyrillic * max(lat + cyr, 1)


def normalize(text):
    lines = [_SPACE.sub(" ", ln).strip() for ln in text.replace("\r", "").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def hf_texts(name, config=None, field="text"):
    from datasets import load_dataset   # faqat HF manbalari uchun kerak
    ds = load_dataset(name, config, split="train", streaming=True)
    for row in ds:
        yield row[field] or ""


def iter_source(src, args):
    if src == "wiki":
        yield from hf_texts("wikimedia/wikipedia", "20231101.uz")
    elif src == "news":
        yield from hf_texts("MLDataScientist/Uzbek_news_dataset")
    elif src == "local":
        yield read_corpus(f"chat-data/Data.txt!,chat-data/savol-javob*{args.qa_repeat}")
    else:   # fayl yoki papka
        p = Path(src)
        for f in sorted(p.rglob("*.txt")) if p.is_dir() else [p]:
            yield f.read_text(encoding="utf-8", errors="replace")


def collect(args, raw_path):
    """Hujjatlarni filtrlab, bitta faylga yozadi (EOT bilan ajratilgan). Qaytaradi: hujjatlar soni."""
    limit, total, n_docs = args.max_mb * 1_000_000, 0, 0
    per_source = limit // max(len(args.sources), 1)
    with open(raw_path, "w", encoding="utf-8") as f:
        for src in args.sources:
            got, t0 = 0, time.time()
            for text in iter_source(src, args):
                text = normalize(text)
                if src != "local" and not good_doc(text):
                    continue
                if src not in ("local",) and args.clean:
                    text = clean_text(text)
                    if len(text) < 200:
                        continue
                f.write(text + EOT)
                got += len(text)
                n_docs += 1
                if got >= per_source and src != "local":
                    break
            total += got
            print(f"  {src}: {got / 1e6:.1f} MB ({time.time() - t0:.0f}s)", flush=True)
    print(f"Jami: {total / 1e6:.1f} MB, {n_docs:,} hujjat")
    return n_docs


def sft_pairs(name):
    from datasets import load_dataset
    ds = load_dataset({"alpaca": "saillab/alpaca-uzbek-cleaned", "instruct": "UAzimov/uzbek-instruct-llm"}[name],
                      split="train")
    for row in ds:
        if name == "alpaca":
            q = row["instruction"].strip() + (f" {row['input'].strip()}" if row.get("input") else "")
            yield q, row["output"]
        else:
            msgs = row["messages"]
            for a, b in zip(msgs, msgs[1:]):
                if a["role"] == "user" and b["role"] == "assistant":
                    yield a["content"], b["content"]


def flat(s):
    """Javob bitta qatorda bo'lishi kerak (chat.py qator oxirida to'xtaydi)."""
    return _SPACE.sub(" ", s.replace("\r", " ").replace("\n", " ")).strip()


def write_sft(args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sft:
        n = 0
        with open(out_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for q, a in sft_pairs(name):
                q, a = flat(q), flat(a)
                if 0 < len(q) <= args.sft_max_chars and 0 < len(a) <= args.sft_max_chars and good_doc(q + " " + a, 1):
                    f.write(f"Savol: {q}\nJavob: {a}\n\n")
                    n += 1
        print(f"  SFT {name}: {n:,} juftlik → {out_dir / (name + '.txt')}")


def main():
    ap = argparse.ArgumentParser(description="Katta korpus: yuklash, tokenizer, tokenlar")
    ap.add_argument("--out", default="data")
    ap.add_argument("--sources", default="wiki,news,local", help="vergul bilan: wiki, news, local yoki yo'l")
    ap.add_argument("--max-mb", type=int, default=600, help="umumiy matn hajmi chegarasi (MB, manbalarga teng bo'linadi)")
    ap.add_argument("--clean", action="store_true", help="HF matnlariga ham clean_text (raqamlarni ham kesadi — ehtiyot bo'ling)")
    ap.add_argument("--qa-repeat", type=int, default=4, help="local: savol-javoblar necha marta takrorlanadi")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--tok-sample-mb", type=int, default=40, help="tokenizer shuncha MB namunada o'qitiladi")
    ap.add_argument("--tokenizer", default=None, help="tayyor tokenizer.json (berilsa, o'qitilmaydi)")
    ap.add_argument("--sft", default="", help="vergul bilan: alpaca, instruct")
    ap.add_argument("--sft-max-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.sources = [s for s in args.sources.split(",") if s]
    args.sft = [s for s in args.sft.split(",") if s]
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = out / "corpus.txt"
    if args.sources:
        print("1) Hujjatlar yig'ilmoqda...")
        collect(args, raw)

        tok_path = out / "tokenizer.json"
        if args.tokenizer:
            tok = BPETokenizer.load(args.tokenizer)
        else:
            print(f"2) Tokenizer o'qitilmoqda: vocab={args.vocab_size}, namuna {args.tok_sample_mb} MB...")
            t0 = time.time()
            with open(raw, encoding="utf-8") as f:
                sample = f.read(args.tok_sample_mb * 1_000_000)
            tok = BPETokenizer.train(sample.replace(EOT, "\n"), args.vocab_size, verbose=True)
            print(f"   {time.time() - t0:.0f}s")
        tok.save(tok_path)

        print("3) Tokenlash...")
        t0, ids, buf = time.time(), array("i"), ""
        with open(raw, encoding="utf-8") as f:
            while True:
                block = f.read(8_000_000)
                if not block:
                    break
                docs = (buf + block).split(EOT)
                buf = docs.pop()             # oxirgi (tugallanmagan) hujjat keyingi blokka
                for d in docs:
                    ids.extend(tok.encode(d, allow_special=False))
                    ids.append(tok.eot_id)
                print(f"   {len(ids) / 1e6:.1f}M token ({time.time() - t0:.0f}s)", flush=True)
        if buf:
            ids.extend(tok.encode(buf, allow_special=False))
            ids.append(tok.eot_id)
        import numpy as np
        np.save(out / "tokens.npy", np.frombuffer(ids, dtype=np.int32))
        print(f"Tayyor: {out / 'tokens.npy'} ({len(ids):,} token), {tok_path}")
    if args.sft:
        print("4) SFT ma'lumotlari...")
        write_sft(args, out / "sft")


if __name__ == "__main__":
    main()
