"""Ma'lumotni tayyorlash: matnni o'qish, tokenizer o'qitish/yuklash, batch'lar, baholash."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
from pathlib import Path

import torch

from .tokenizer import EOT, BPETokenizer


_JUNK = re.compile(r"arxivlangan|olindi|archived|retrieved|isbn|doi|https?|www|\.org|\.com|\.net", re.I)
_ENGLISH = {"the", "and", "of", "to", "in", "is", "for", "on", "with", "from", "by", "as", "that", "this",
            "an", "are", "be", "or", "it", "at", "was", "which", "were", "has", "its"}
_GLUED = re.compile(r"(?<=[a-zʼ‘'’])(?=[A-Z][a-z])")


def clean_text(text, window=30, max_junk=0.25):
    """Korpusni tozalash: yopishib qolgan so'zlarni ajratadi ("otishRivojlanish" → "otish Rivojlanish")
    va chiqindi zichligi yuqori bo'laklarni (manbalar ro'yxati, yillar, URL, inglizcha matn) olib tashlaydi."""
    out_lines = []
    for line in text.split("\n"):
        words = _GLUED.sub(" ", line).split()
        kept = []
        for i in range(0, len(words), window):
            chunk = words[i:i + window]
            junk = sum(bool(_JUNK.search(w)) or any(c.isdigit() for c in w) or w.lower() in _ENGLISH for w in chunk)
            if junk / len(chunk) < max_junk:
                kept.extend(chunk)
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines)


def read_corpus(path, clean=False) -> str:
    """Korpusni o'qish. path — .txt fayl, .txt fayllar papkasi yoki vergul bilan ajratilgan ro'yxat.
    Har bir element oxirida: `!` — shu manbani tozalash (clean_text), `*N` — N marta takrorlash
    (takrorlashda bo'sh qator bilan ajratilgan bloklar har safar aralashtiriladi).
    Masalan: "chat-data/Data.txt!,chat-data/savol-javob*8". clean=True barcha manbalarni tozalaydi.
    Hujjatlar orasiga EOT qo'yiladi."""
    texts = []
    for spec in str(path).split(","):
        spec = spec.strip()
        do_clean = clean or spec.endswith("!")
        spec, _, rep = spec.rstrip("!").partition("*")
        p = Path(spec)
        files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(f"{spec} ichida .txt fayl topilmadi")
        part = [f.read_text(encoding="utf-8") for f in files]
        if do_clean:
            part = [clean_text(t) for t in part]
        n = int(rep or 1)
        if n == 1:
            texts.extend(part)
            continue
        # Takrorlanadigan manba (masalan savol-javoblar): bo'sh qator bilan ajratilgan bloklar har
        # takrorlashda aralashtiriladi va EOT bilan ajratiladi — aks holda model javobni savoldan emas,
        # oldingi blokdan (tartibni yodlab) bashorat qilishni o'rganadi.
        blocks = [b.strip() for t in part for b in t.split("\n\n") if b.strip()]
        rng = random.Random(len(blocks))
        for _ in range(n):
            rng.shuffle(blocks)
            texts.append(EOT.join(blocks))
    return EOT.join(texts)


def prepare_tokens(data_path, tokenizer_path, vocab_size=4096, verbose=True, clean=False):
    """Tokenizer'ni o'qitadi (yoki mavjudini yuklaydi) va korpusni tokenlaydi (keshlab)."""
    text = read_corpus(data_path, clean=clean)
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        tok = BPETokenizer.load(tokenizer_path)
    else:
        if verbose:
            print(f"Tokenizer o'qitilmoqda (vocab={vocab_size})...")
        tok = BPETokenizer.train(text, vocab_size)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(tokenizer_path)
    cache = tokenizer_path.with_suffix(".tokens.pt")
    stamp = (str(data_path), hashlib.sha1(text.encode("utf-8")).hexdigest(), tok.vocab_size, clean)
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved.get("stamp") == stamp:
            return tok, saved["tokens"]
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save({"stamp": stamp, "tokens": tokens}, cache)
    return tok, tokens


def read_qa_pairs(path):
    """"Savol: …\nJavob: …" bloklarini o'qiydi (path — fayl, papka yoki vergul bilan ajratilgan ro'yxat)."""
    pairs = []
    for spec in str(path).split(","):
        p = Path(spec.strip())
        for f in sorted(p.rglob("*.txt")) if p.is_dir() else [p]:
            for block in f.read_text(encoding="utf-8").split("\n\n"):
                lines = block.strip().split("\n")
                if len(lines) == 2 and lines[0].startswith("Savol:") and lines[1].startswith("Javob:"):
                    pairs.append((lines[0][6:].strip(), lines[1][6:].strip()))
    return pairs


def qa_prompt(question):
    """Savol-javob formati (o'qitish, baholash va chat.py uchun bir xil)."""
    return f"Savol: {question}\nJavob:"


def encode_dialog(tok, turns):
    """Dialog (bir yoki bir nechta navbat) → (tokenlar, maska). Har javob "\n" bilan tugaydi (suhbat davom
    etishi mumkin), dialog oxirida EOT. Maska faqat javoblarda (+ "\n" va EOT) True: model savollarni
    o'qiydi, lekin faqat javob berishni o'rganadi."""
    ids, mask = [], []
    for q, a in turns:
        prompt, answer = tok.encode(qa_prompt(q)), tok.encode(f" {a}\n")
        ids += prompt + answer
        mask += [False] * len(prompt) + [True] * len(answer)
    return torch.tensor(ids + [tok.eot_id]), torch.tensor(mask + [True])


def encode_qa(tok, pairs):
    """Har bir savol-javob juftligi — bitta navbatli dialog."""
    return [encode_dialog(tok, [(q, a)]) for q, a in pairs]


def pack_rows(sample, batch_size, seq_len, device="cpu"):
    """Har bir qator sample() bergan (tokenlar, maska) misollar bilan to'ldiriladi (misol boshidan
    boshlanadi); javob bo'lmagan tokenlarning maqsadi -100 (loss hisoblanmaydi)."""
    xs, ys = [], []
    for _ in range(batch_size):
        ids, mask = [], []
        while sum(len(t) for t in ids) < seq_len + 1:
            t, m = sample()
            ids.append(t)
            mask.append(m)
        ids, mask = torch.cat(ids)[:seq_len + 1], torch.cat(mask)[:seq_len + 1]
        xs.append(ids[:-1])
        ys.append(torch.where(mask[1:], ids[1:], -100))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def qa_batch(examples, batch_size, seq_len, generator=None, device="cpu"):
    """Tasodifiy tayyor misollardan qatorlar (pack_rows)."""
    return pack_rows(lambda: examples[int(torch.randint(len(examples), (1,), generator=generator))],
                     batch_size, seq_len, device)


def split_tokens(tokens, val_frac=0.1, block=8192):
    """Validatsiya uchun korpusning turli joylaridan bloklar olinadi (faqat oxiridan emas),
    shunda val loss domen siljishini emas, umumlashtirishni o'lchaydi."""
    n_blocks = len(tokens) // block
    every = max(round(1 / val_frac), 2)
    if n_blocks < every:
        n_val = max(int(len(tokens) * val_frac), 1)
        return tokens[:-n_val], tokens[-n_val:]
    is_val = torch.arange(n_blocks) % every == every // 2
    blocks = tokens[: n_blocks * block].view(n_blocks, block)
    train = torch.cat([blocks[~is_val].reshape(-1), tokens[n_blocks * block:]])
    return train, blocks[is_val].reshape(-1)


def random_batch(tokens, batch_size, seq_len, generator=None, device="cpu"):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + seq_len] for i in ix]).long()   # tokenlar ixcham int32 bo'lishi mumkin
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in ix]).long()
    return x.to(device), y.to(device)


class StreamBatcher:
    """Uzluksiz oqimlar: har bir qator matnning ketma-ket bo'lagini oladi, model holati
    (xotira) qadamdan qadamga uzatiladi — kontekst seq_len bilan cheklanmaydi."""

    def __init__(self, tokens, batch_size, seq_len, device="cpu"):
        self.seq_len, self.device = seq_len, device
        n = (len(tokens) - 1) // batch_size
        self.x = tokens[: n * batch_size].long().view(batch_size, n)
        self.y = tokens[1: n * batch_size + 1].long().view(batch_size, n)
        self.pos = 0

    def next(self):
        """Qaytaradi: x, y, reset (True bo'lsa holatni nolga qaytarish kerak)."""
        reset = self.pos + self.seq_len > self.x.size(1)
        if reset:
            self.pos = 0
        sl = slice(self.pos, self.pos + self.seq_len)
        self.pos += self.seq_len
        return self.x[:, sl].to(self.device), self.y[:, sl].to(self.device), reset or self.pos == self.seq_len


@torch.no_grad()
def evaluate(model, tokens, seq_len, batch_size=8, max_windows=None, device="cpu", autocast=None):
    """Kesishmaydigan oynalar bo'yicha o'rtacha loss (nat/token)."""
    was_training = model.training
    model.eval()
    n = (len(tokens) - 1) // seq_len
    if max_windows:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"Baholash uchun matn juda qisqa: {len(tokens)} token < {seq_len}")
    total, count = 0.0, 0
    for s in range(0, n, batch_size):
        ix = range(s, min(s + batch_size, n))
        x = torch.stack([tokens[i * seq_len:(i + 1) * seq_len] for i in ix]).long().to(device)
        y = torch.stack([tokens[i * seq_len + 1:(i + 1) * seq_len + 1] for i in ix]).long().to(device)
        with autocast or _null():
            _, loss = model(x, targets=y)
        total += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _NoScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        pass


def mixed_precision(device, mode="auto"):
    """(autocast_fn, scaler). mode: auto | bf16 | fp16 | off. auto: CUDA'da bf16 bo'lsa bf16, aks holda fp16
    (Colab/Kaggle T4, P100), CPU'da o'chiq. fp16 da loss GradScaler bilan masshtablanadi."""
    if mode == "auto":
        mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    if dtype is None:
        return _null, _NoScaler()
    scaler = torch.amp.GradScaler("cuda") if mode == "fp16" else _NoScaler()
    return (lambda: torch.autocast(device.type, dtype=dtype)), scaler


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cosine_lr(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def default_threads():
    """OMP_NUM_THREADS berilgan bo'lsa shuni, aks holda barcha yadrolarni ishlatadi."""
    n = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
    torch.set_num_threads(n)
    return n
