"""Ma'lumotni tayyorlash: matnni o'qish, tokenizer o'qitish/yuklash, batch'lar, baholash."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from .tokenizer import EOT, BPETokenizer


def read_corpus(path) -> str:
    """Bitta .txt fayl yoki papkadagi barcha .txt fayllar (hujjatlar orasiga EOT qo'yiladi)."""
    p = Path(path)
    files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"{path} ichida .txt fayl topilmadi")
    return EOT.join(f.read_text(encoding="utf-8") for f in files)


def prepare_tokens(data_path, tokenizer_path, vocab_size=4096, verbose=True):
    """Tokenizer'ni o'qitadi (yoki mavjudini yuklaydi) va korpusni tokenlaydi (keshlab)."""
    text = read_corpus(data_path)
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
    stamp = (str(Path(data_path).resolve()), len(text), tok.vocab_size)
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved.get("stamp") == stamp:
            return tok, saved["tokens"]
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save({"stamp": stamp, "tokens": tokens}, cache)
    return tok, tokens


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
    x = torch.stack([tokens[i:i + seq_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


class StreamBatcher:
    """Uzluksiz oqimlar: har bir qator matnning ketma-ket bo'lagini oladi, model holati
    (xotira) qadamdan qadamga uzatiladi — kontekst seq_len bilan cheklanmaydi."""

    def __init__(self, tokens, batch_size, seq_len, device="cpu"):
        self.seq_len, self.device = seq_len, device
        n = (len(tokens) - 1) // batch_size
        self.x = tokens[: n * batch_size].view(batch_size, n)
        self.y = tokens[1: n * batch_size + 1].view(batch_size, n)
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
        x = torch.stack([tokens[i * seq_len:(i + 1) * seq_len] for i in ix]).to(device)
        y = torch.stack([tokens[i * seq_len + 1:(i + 1) * seq_len + 1] for i in ix]).to(device)
        with autocast or _null():
            _, loss = model(x, targets=y)
        total += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


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
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    return n
