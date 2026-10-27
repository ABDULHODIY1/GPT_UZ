"""Katta o'zbekcha korpusni tayyorlash (Colab/Kaggle uchun): yuklab olish → filtrlash → tokenizer → tokenlar.

    python prepare_data.py --out data --sources wiki,news,local --max-mb 600 --vocab-size 16000 --sft alpaca,instruct

Tozalash standart yoqiq (umid/clean.py): tutuq belgilari, URL/HTML/vikimatn, "Manbalar" bo'limlari,
raqam/belgi qatorlari, kirillcha hujjatlar, takror hujjat va abzaslar; SFT juftliklari ham.

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

from umid.clean import Deduper, clean_doc, cyrillic_ratio, letter_ratio, normalize_apostrophes
from umid.data import read_corpus
from umid.tokenizer import EOT, BPETokenizer

_SPACE = re.compile(r"[ \t\u00a0]+")


def good_doc(text, min_chars=300, max_cyrillic=0.2):
    """Juda qisqa va asosan kirillcha hujjatlarni tashlaydi (kichik model bitta yozuvni yaxshiroq o'rganadi)."""
    return len(text) >= min_chars and cyrillic_ratio(text[:3000]) <= max_cyrillic


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
    dedup = Deduper()
    with open(raw_path, "w", encoding="utf-8") as f:
        for src in args.sources:
            got, seen, dropped, t0 = 0, 0, 0, time.time()
            for text in iter_source(src, args):
                seen += len(text)
                if src != "local":   # loyiha ma'lumoti allaqachon tozalangan (read_corpus "!")
                    if args.clean:
                        text = dedup(clean_doc(text))
                    if not good_doc(text):
                        dropped += 1
                        continue
                f.write(text + EOT)
                got += len(text)
                n_docs += 1
                if got >= per_source and src != "local":
                    break
            total += got
            print(f"  {src}: {got / 1e6:.1f} MB olindi ({seen / 1e6:.1f} MB ko'rildi, {dropped:,} hujjat tashlandi, "
                  f"{time.time() - t0:.0f}s)", flush=True)
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
    return _SPACE.sub(" ", normalize_apostrophes(s).replace("\r", " ").replace("\n", " ")).strip()


def good_pair(q, a, max_chars):
    """SFT juftligi: uzunlik chegarasi, asosan harflar, kirillcha emas, javob savolni takrorlamaydi."""
    return (3 <= len(q) <= max_chars and 2 <= len(a) <= max_chars and letter_ratio(q + a) >= 0.5
            and cyrillic_ratio(q + " " + a) <= 0.2 and a.lower() != q.lower())


def write_sft(args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for name in args.sft:
        n = dropped = 0
        with open(out_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for q, a in sft_pairs(name):
                q, a = flat(q), flat(a)
                if q.lower() in seen or not good_pair(q, a, args.sft_max_chars):
                    dropped += 1
                    continue
                seen.add(q.lower())
                f.write(f"Savol: {q}\nJavob: {a}\n\n")
                n += 1
        print(f"  SFT {name}: {n:,} juftlik ({dropped:,} tashlandi: takror yoki sifatsiz) → {out_dir / (name + '.txt')}")


def main():
    ap = argparse.ArgumentParser(description="Katta korpus: yuklash, tokenizer, tokenlar")
    ap.add_argument("--out", default="data")
    ap.add_argument("--sources", default="wiki,news,local", help="vergul bilan: wiki, news, local yoki yo'l")
    ap.add_argument("--max-mb", type=int, default=600, help="umumiy matn hajmi chegarasi (MB, manbalarga teng bo'linadi)")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="tozalashni o'chirish (standart yoqiq: umid/clean.py — belgilar, URL, vikimatn, takrorlar)")
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
