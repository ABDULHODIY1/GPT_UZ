"""Bayt darajasidagi BPE tokenizer (tashqi kutubxonasiz).

O'zbek tili uchun so'z ichidagi tutuq belgilari (o‘, g‘, ʼ, ') so'zning bir qismi sifatida
olinadi, shuning uchun "o‘g‘il", "bo'lsa" kabi so'zlar bo'linib ketmaydi.
Har qanday matnni (emoji, kirill, lotin) kodlay oladi — noma'lum token yo'q.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

PRETOKENIZE = re.compile(r" ?[^\W\d_]+(?:['‘’ʼ`][^\W\d_]+)*['‘’ʼ`]?| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+")
EOT = "<|endoftext|>"


def _merge(ids, pair, new_id):
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges, special_tokens=(EOT,)):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.ranks.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(special_tokens)}
        self.special_inv = {i: tok for tok, i in self.special.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in self.special) + ")") if self.special else None
        self._cache = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    @property
    def eot_id(self):
        return self.special.get(EOT)

    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens=(EOT,), verbose=False):
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0, "vocab_size juda kichik"
        words = Counter(PRETOKENIZE.findall(text))
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        counts = Counter()
        where = defaultdict(set)
        for i, ids in enumerate(seqs):
            for p in zip(ids, ids[1:]):
                counts[p] += freqs[i]
                where[p].add(i)
        merges = []
        for m in range(n_merges):
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            new_id = 256 + m
            merges.append(best)
            for i in where.pop(best):
                ids, f = seqs[i], freqs[i]
                new_ids = _merge(ids, best, new_id)
                if len(new_ids) == len(ids):
                    continue
                for p in zip(ids, ids[1:]):
                    counts[p] -= f
                for p in zip(new_ids, new_ids[1:]):
                    counts[p] += f
                    where[p].add(i)
                seqs[i] = new_ids
            counts.pop(best, None)
            if m % 256 == 0:
                for p in [p for p, c in counts.items() if c <= 0]:
                    del counts[p]
            if verbose and (m + 1) % 500 == 0:
                print(f"  BPE: {m + 1}/{n_merges} merge")
        return cls(merges, special_tokens)

    def _encode_chunk(self, chunk: str):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.ranks[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special=True):
        parts = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        out = []
        for part in parts:
            if part in self.special and allow_special:
                out.append(self.special[part])
                continue
            for chunk in PRETOKENIZE.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids) -> str:
        chunks, buf = [], b""
        for i in ids:
            i = int(i)
            if i in self.special_inv:
                chunks.append(buf.decode("utf-8", errors="replace"))
                chunks.append(self.special_inv[i])
                buf = b""
            else:
                buf += self.vocab[i]
        chunks.append(buf.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def to_dict(self):
        return {"type": "byte-bpe", "merges": [list(m) for m in self.merges], "special_tokens": list(self.special)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"], d["special_tokens"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
