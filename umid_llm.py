"""Umid LLM — HAMMASI BITTA FAYLDA: ma'lumot yuklash va tozalash, tokenizer, Tanlov modeli, pretraining,
sekin SFT, sinash va suhbat. Faqat torch, numpy va (Hugging Face ma'lumotlari uchun) datasets kerak.

Colab / Kaggle (GPU):
    * butun faylni bitta katakka joylab ishga tushiring (keyingi katakda suhbat: chat(MODEL, TOK)), yoki
    * !python umid_llm.py --preset small --steps 20000 --chat

Sozlamalar quyidagi CONFIG da; buyruq qatorida ham o'zgartirish mumkin: --steps 1000 --max_mb 50.
Har bosqich tugagan bo'lsa qayta bajarilmaydi, pretraining uzilgan joyidan davom etadi.

Tanlov xotirasi (attention o'rniga): har head hujjatdagi eng muhim K ta tokenni o'zi tanlab saqlaydi
(muhimlik o'rganiladi, eng kami chiqariladi — xotira hajmi o'zgarmas), saqlanganlarni softmax'siz, sigmoid
bilan alohida o'qiydi; masofa jarimasi logarifmik (pozitsiya embedding yo'q).
"""

# ============================== SOZLAMALAR ==============================
CONFIG = dict(
    save="",                    # bo'sh: Colab — Google Drive, Kaggle — /kaggle/working, aks holda ./umid_natijalar
    preset="small",             # tiny (~7.6M) | small (~17.7M) | base (~104M parametr)
    sources="wiki,news",        # pretraining: wiki, news (Hugging Face) yoki .txt fayl/papka yo'llari, vergul bilan
    sft_sources="gptuz,alpaca,instruct",  # SFT: gptuz (loyiha savol-javoblari), alpaca, instruct yoki fayl/papka
    max_mb=600,                 # pretraining matni hajmi (MB); tez sinov: 50
    vocab_size=16000,
    tok_sample_mb=20,           # tokenizer shuncha MB namunada o'qitiladi
    seq_len=512,
    batch_size=32,
    steps=20000,                # pretraining qadamlari; tez sinov: 1000
    lr=1e-3,
    warmup=500,
    eval_every=1000,
    sft_steps=6000,             # SFT sekin: ko'p qadam,
    sft_lr=1e-4,                # kichik o'rganish tezligi,
    sft_warmup=300,             # uzun qizish bosqichi,
    sft_mix=0.25,               # va umumiy matn aralashmasi (tilni unutmaslik uchun)
    select_k=32,                # Tanlov: har head saqlaydigan tokenlar soni
    seed=0,
    allow_cpu=False,            # GPU bo'lmasa o'qitish boshlanmaydi (CPU da kichik sinovdan boshqasi juda uzoq)
)
# ========================================================================

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EOT = "<|endoftext|>"

# ============================== 1. TOZALASH ==============================
_APOS = re.compile(r"[‘’ʻʼ`´′]")
_URL = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\w+")
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>[\s\S]*?</ref>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]{1,200}>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}|\{\|[\s\S]*?\|\}")
_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_BOLD = re.compile(r"'{2,}")
_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
_SPACE = re.compile(r"[ \t ​]+")
_LETTER = re.compile(r"[^\W\d_]")
_CYR = re.compile(r"[Ѐ-ӿ]")
_STOP = ("manbalar", "havolalar", "adabiyotlar", "izohlar", "yana qarang", "tashqi havolalar", "manba",
         "references", "external links", "see also")


def letter_ratio(s):
    return len(_LETTER.findall(s)) / max(len(s), 1)


def cyrillic_ratio(s):
    letters = _LETTER.findall(s)
    return sum(bool(_CYR.match(c)) for c in letters) / max(len(letters), 1)


def clean_doc(text, min_line=30, min_letters=0.6):
    """Belgilar, URL, HTML/vikimatn, "Manbalar" bo'limlari, raqam/belgi qatorlari olib tashlanadi."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = _APOS.sub("'", text)
    for rx, rep in ((_REF, " "), (_TEMPLATE, " "), (_LINK, r"\1"), (_TAG, " "), (_URL, " "), (_BOLD, "")):
        text = rx.sub(rep, text)
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        m = _HEADING.match(line)
        if (m.group(1) if m else line).strip().lower().rstrip(":") in _STOP:
            break
        if m:
            continue
        line = _SPACE.sub(" ", line).strip()
        if len(line) >= min_line and letter_ratio(line) >= min_letters:
            out.append(line)
    return "\n".join(out)


class Deduper:
    """Bir xil hujjat va abzaslarni bir martadan qoldiradi."""

    def __init__(self):
        self.docs, self.paras = set(), set()

    @staticmethod
    def _key(s):
        return hashlib.blake2b(re.sub(r"\W+", "", s.lower()).encode(), digest_size=8).digest()

    def __call__(self, text):
        k = self._key(text[:2000])
        if k in self.docs:
            return ""
        self.docs.add(k)
        kept = []
        for p in text.split("\n"):
            pk = self._key(p)
            if pk not in self.paras:
                self.paras.add(pk)
                kept.append(p)
        return "\n".join(kept)


def good_doc(text, min_chars=300):
    return len(text) >= min_chars and cyrillic_ratio(text[:3000]) <= 0.2


def flat(s):
    return _SPACE.sub(" ", _APOS.sub("'", s).replace("\r", " ").replace("\n", " ")).strip()


# ============================== 2. TOKENIZER (bayt darajasidagi BPE) ==============================
PRETOKENIZE = re.compile(r" ?[^\W\d_]+(?:['‘’ʼ`][^\W\d_]+)*['‘’ʼ`]?| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+")


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


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {p: 256 + i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.ranks.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.vocab_size = self.eot_id + 1
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size):
        words = Counter(PRETOKENIZE.findall(text))
        seqs, freqs = [list(w.encode("utf-8")) for w in words], list(words.values())
        counts, where = Counter(), defaultdict(set)
        for i, ids in enumerate(seqs):
            for p in zip(ids, ids[1:]):
                counts[p] += freqs[i]
                where[p].add(i)
        merges = []
        for m in range(vocab_size - 257):
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            merges.append(best)
            for i in where.pop(best):
                ids, f = seqs[i], freqs[i]
                new = _merge(ids, best, 256 + m)
                if len(new) == len(ids):
                    continue
                for p in zip(ids, ids[1:]):
                    counts[p] -= f
                for p in zip(new, new[1:]):
                    counts[p] += f
                    where[p].add(i)
                seqs[i] = new
            counts.pop(best, None)
            if m % 256 == 0:
                for p in [p for p, c in counts.items() if c <= 0]:
                    del counts[p]
            if (m + 1) % 2000 == 0:
                print(f"   BPE: {m + 1}/{vocab_size - 257}", flush=True)
        return cls(merges)

    def _word(self, w):
        ids = self._cache.get(w)
        if ids is None:
            ids = list(w.encode("utf-8"))
            while len(ids) >= 2:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1e18))
                if pair not in self.ranks:
                    break
                ids = _merge(ids, pair, self.ranks[pair])
            if len(self._cache) < 300_000:
                self._cache[w] = ids
        return ids

    def encode(self, text):
        out = []
        for w in PRETOKENIZE.findall(text):
            out.extend(self._word(w))
        return out

    def decode(self, ids):
        b = b"".join(self.vocab.get(int(i), b"") for i in ids if int(i) != self.eot_id)
        return b.decode("utf-8", errors="replace")


# ============================== 3. MA'LUMOT ==============================
GPTUZ_RAW = "https://raw.githubusercontent.com/ABDULHODIY1/GPT_UZ/main/chat-data/savol-javob/"
GPTUZ_FILES = ["01_muloqot", "02_ozbekiston", "03_fan", "04_texnologiya", "05_salomatlik_hayot", "06_vaqt_hisob",
               "07_tabiat_turmush", "08_turli_ifodalar", "09_qayta_ifodalar"]
IDENTITY = [("Sen kimsan?", "Men Umid, o'zbek tilida suhbatlashadigan sun'iy intellekt yordamchisiman."),
            ("Isming nima?", "Mening ismim Umid."),
            ("Seni kim yaratgan?", "Meni Muhiddinov Abdulhodiy yaratgan."),
            ("Salom", "Assalomu alaykum! Men Umid. Sizga qanday yordam bera olaman?")]


def hf_rows(name, config=None, streaming=True):
    try:
        from datasets import load_dataset
    except ImportError:                  # bitta fayl: kerakli kutubxona o'zi o'rnatiladi
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset
    return load_dataset(name, config, split="train", streaming=streaming)


def read_paths(spec):
    for p in (Path(x) for x in spec.split(",") if x):
        for f in sorted(p.rglob("*.txt")) if p.is_dir() else [p]:
            yield f.read_text(encoding="utf-8", errors="replace")


def pretrain_texts(src):
    if src == "wiki":
        for r in hf_rows("wikimedia/wikipedia", "20231101.uz"):
            yield r["text"]
    elif src == "news":
        for r in hf_rows("MLDataScientist/Uzbek_news_dataset"):
            yield r["text"] or ""
    else:
        yield from read_paths(src)


def parse_qa(text):
    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) == 2 and lines[0].startswith("Savol:") and lines[1].startswith("Javob:"):
            yield lines[0][6:].strip(), lines[1][6:].strip()


def sft_pairs(src):
    if src == "gptuz":
        for name in GPTUZ_FILES:
            with urllib.request.urlopen(f"{GPTUZ_RAW}{name}.txt", timeout=60) as r:
                yield from parse_qa(r.read().decode("utf-8"))
    elif src == "alpaca":
        for r in hf_rows("saillab/alpaca-uzbek-cleaned", streaming=False):
            yield r["instruction"].strip() + (f" {r['input'].strip()}" if r.get("input") else ""), r["output"]
    elif src == "instruct":
        for r in hf_rows("UAzimov/uzbek-instruct-llm", streaming=False):
            m = r["messages"]
            for a, b in zip(m, m[1:]):
                if a["role"] == "user" and b["role"] == "assistant":
                    yield a["content"], b["content"]
    else:
        for text in read_paths(src):
            yield from parse_qa(text)


def prepare_data(cfg, out):
    """Pretraining tokenlari (tokens.npy), tokenizer.json va tozalangan SFT juftliklari (sft.json)."""
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "tokens.npy").exists():
        print("== 1) Ma'lumot: yuklash va tozalash", flush=True)
        srcs = [s for s in cfg["sources"].split(",") if s]
        per_source, dedup = cfg["max_mb"] * 1_000_000 // max(len(srcs), 1), Deduper()
        files = []
        for src in srcs:   # har manba alohida faylga; tugagani belgilanadi — uzilsa qayta yuklanmaydi
            path = out / f"corpus_{re.sub(r'[^\w]+', '_', src).strip('_')}.txt"
            files.append(path)
            if path.with_suffix(".done").exists():
                print(f"   {src}: tayyor ({path.stat().st_size / 1e6:.0f} MB) — o'tkazib yuborildi", flush=True)
                continue
            got = dropped = 0
            t0, next_report = time.time(), 50_000_000
            with open(path, "w", encoding="utf-8") as f:
                for text in pretrain_texts(src):
                    text = dedup(clean_doc(text))
                    if not good_doc(text):
                        dropped += 1
                        continue
                    f.write(text + EOT)
                    got += len(text)
                    if got >= next_report:
                        print(f"   {src}: {got / 1e6:.0f}/{per_source / 1e6:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
                        next_report += 50_000_000
                    if got >= per_source:
                        break
            path.with_suffix(".done").touch()
            print(f"   {src}: {got / 1e6:.1f} MB olindi, {dropped:,} hujjat tashlandi ({time.time() - t0:.0f}s)", flush=True)
        print(f"== 2) Tokenizer (vocab {cfg['vocab_size']}, namuna {cfg['tok_sample_mb']} MB)", flush=True)
        t0, sample = time.time(), []
        for path in files:   # namuna har manbadan teng — tokenizer barcha uslublarni ko'radi
            with open(path, encoding="utf-8") as f:
                sample.append(f.read(cfg["tok_sample_mb"] * 1_000_000 // len(files)).replace(EOT, "\n"))
        tok = Tokenizer.train("\n".join(sample), cfg["vocab_size"])
        (out / "tokenizer.json").write_text(json.dumps({"merges": [list(m) for m in tok.merges]}))
        print(f"   tayyor ({time.time() - t0:.0f}s)", flush=True)
        print("== 3) Tokenlash", flush=True)
        ids, t0 = array("i"), time.time()
        for path in files:
            buf = ""
            with open(path, encoding="utf-8") as f:
                while block := f.read(8_000_000):
                    docs = (buf + block).split(EOT)
                    buf = docs.pop()
                    for d in docs:
                        ids.extend(tok.encode(d))
                        ids.append(tok.eot_id)
                    print(f"   {len(ids) / 1e6:.1f}M token ({time.time() - t0:.0f}s)", flush=True)
            if buf:
                ids.extend(tok.encode(buf))
                ids.append(tok.eot_id)
        np.save(out / "tokens.npy", np.frombuffer(ids, dtype=np.int32))
    if not (out / "sft.json").exists():
        print("== 4) SFT juftliklari (tozalash, takrorlarni olib tashlash)", flush=True)
        pairs, seen = list(IDENTITY), {q.lower() for q, _ in IDENTITY}
        for src in (s for s in cfg["sft_sources"].split(",") if s):
            n = 0
            try:
                for q, a in sft_pairs(src):
                    q, a = flat(q), flat(a)
                    if (q.lower() in seen or not 3 <= len(q) <= 600 or not 2 <= len(a) <= 600
                            or letter_ratio(q + a) < 0.5 or cyrillic_ratio(q + " " + a) > 0.2):
                        continue
                    seen.add(q.lower())
                    pairs.append((q, a))
                    n += 1
            except Exception as e:   # bitta manba ishlamasa ham davom etadi
                print(f"   {src}: yuklab bo'lmadi ({e})")
            print(f"   {src}: {n:,} juftlik", flush=True)
        (out / "sft.json").write_text(json.dumps(pairs, ensure_ascii=False))
    tok = Tokenizer(json.loads((out / "tokenizer.json").read_text())["merges"])
    tokens = torch.from_numpy(np.load(out / "tokens.npy"))
    return tok, tokens, json.loads((out / "sft.json").read_text())


# ============================== 4. MODEL: TANLOV XOTIRASI ==============================
PRESETS = {"tiny": (256, 4, 4), "small": (384, 6, 6), "base": (768, 12, 12)}   # d_model, qatlamlar, head'lar


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    select_k: int = 32
    select_chunk: int = 128
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self):
        return 64 * math.ceil(self.d_model * 8 / 3 / 64)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _segment_start(seg):
    idx = torch.arange(seg.size(1), device=seg.device).expand_as(seg)
    is_start = torch.ones_like(seg, dtype=torch.bool)
    is_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return torch.where(is_start, idx, torch.zeros_like(idx)).cummax(1).values


def selective_read(q, k, v, s, pos, seg, K, tau, bias, slope, cache=None):
    """Bir bo'lak uchun Tanlov xotirasi (parallel; tokenma-token generatsiya bilan aynan bir xil).
    q, k: [B,H,T,D] (normallangan), v: [B,H,T,D], s: [B,H,T] muhimlik, pos/seg: [B,T]; cache — oldingi bo'lakdan."""
    B, H, T, D = k.shape
    if cache is None:
        cache = (k.new_zeros(B, H, 0, D), v.new_zeros(B, H, 0, v.size(-1)), s.new_zeros(B, H, 0),
                 pos.new_zeros(B, H, 0), torch.zeros(B, H, 0, dtype=torch.bool, device=k.device))
    ck, cv, cs, cp, cm = cache
    C = ck.size(2)
    N = C + T
    k_all, v_all, s_all = torch.cat([ck, k], 2), torch.cat([cv, v], 2), torch.cat([cs, s], 2)
    p_all = torch.cat([cp, pos[:, None, :].expand(B, H, T)], 2)
    valid = torch.cat([cm, torch.ones(B, H, T, dtype=torch.bool, device=k.device)], 2)
    seg_all = torch.cat([seg.new_zeros(B, C), seg], 1)
    idx = torch.arange(N, device=k.device)
    si, sj = s_all[..., :, None], s_all[..., None, :]
    beats = ((si > sj) | ((si == sj) & (idx[:, None] > idx[None, :]))) & valid[..., :, None]
    cum = F.pad(beats.to(torch.int32).cumsum(-2), (0, 0, 1, 0))
    start = _segment_start(seg_all)
    g = C + torch.arange(T, device=k.device)
    rank = cum[:, :, g] - cum.gather(2, start[:, None, g, None].expand(B, H, T, N))
    same_seg = seg_all[:, None, g, None] == seg_all[:, None, None, :]
    allowed = ((rank < K) & valid[:, :, None, :] & (idx[None, :] < g[:, None]) & same_seg) | (idx[None, :] == g[:, None])
    dist = (pos[:, None, :, None] - p_all[:, :, None, :]).clamp(min=0).to(q.dtype)
    logits = (tau[:, None, None] * (q @ k_all.transpose(-1, -2)) + bias[:, None, None]
              + F.logsigmoid(s_all)[:, :, None, :] - slope[:, None, None] * torch.log1p(dist))
    o = (torch.sigmoid(logits) * allowed) @ v_all
    last = start[:, -1]
    rank_end = cum[:, :, N] - cum.gather(2, last[:, None, None, None].expand(B, H, 1, N)).squeeze(2)
    keep = (rank_end < K) & valid & (seg_all == seg_all[:, -1:])[:, None, :]
    n_keep = min(K, N)
    order = torch.where(keep, (N - idx).to(s.dtype), torch.full_like(s_all, -1.0)).topk(n_keep, -1).indices
    take = lambda x: x.gather(2, order if x.dim() == 3 else order[..., None].expand(*order.shape, x.size(-1)))  # noqa: E731
    new = (take(k_all), take(v_all), take(s_all), take(p_all), keep.gather(2, order))
    if n_keep < K:
        new = tuple(F.pad(x, (0, 0, 0, K - n_keep)) if x.dim() == 4 else F.pad(x, (0, K - n_keep)) for x in new)
    return o, new


class SelectiveMemory(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        H, D = c.n_heads, c.head_dim
        self.qkv = nn.Linear(c.d_model, 3 * H * D, bias=False)
        self.s_proj = nn.Linear(c.d_model, H)              # muhimlik (boshida 0 → "oxirgi K ta token")
        self.gate = nn.Linear(c.d_model, H * D, bias=False)
        self.o_norm = RMSNorm(D)
        self.out = nn.Linear(H * D, c.d_model, bias=False)
        self.tau_log = nn.Parameter(torch.full((H,), math.log(8.0)))
        self.attn_offset = nn.Parameter(torch.full((H,), -math.log(c.select_k)))
        self.slope_raw = nn.Parameter(torch.log(torch.expm1(torch.logspace(math.log10(0.05), 0.0, H))))

    def forward(self, x, state, reset):
        c, (B, T, _) = self.c, x.shape
        H, D = c.n_heads, c.head_dim
        with torch.autocast(x.device.type, enabled=False):   # xotira hisob-kitobi doim fp32 da
            q, k, v = self.qkv(x).float().view(B, T, 3, H, D).unbind(2)
            q, k, v = F.normalize(q, dim=-1).transpose(1, 2), F.normalize(k, dim=-1).transpose(1, 2), v.transpose(1, 2)
            s = self.s_proj(x).float().transpose(1, 2)
            st = state or {}
            n0 = st.get("n", torch.zeros(B, dtype=torch.long, device=x.device))
            pos = n0[:, None] + torch.arange(T, device=x.device)
            seg = reset.long().cumsum(1)
            cache = tuple(st[n] for n in "kvspm") if "k" in st else None
            tau, bias, slope = self.tau_log.exp().float(), self.attn_offset.float(), F.softplus(self.slope_raw).float()
            outs = []
            for a in range(0, T, c.select_chunk):
                b = min(a + c.select_chunk, T)
                rel = seg[:, a:b] - seg[:, a - 1:a] if a else seg[:, a:b]
                o, cache = selective_read(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], s[:, :, a:b], pos[:, a:b], rel,
                                          c.select_k, tau, bias, slope, cache)
                outs.append(o)
        o = self.o_norm(torch.cat(outs, 2).transpose(1, 2).to(x.dtype)) * F.silu(self.gate(x)).view(B, T, H, D)
        return self.out(o.reshape(B, T, H * D)), {"n": n0 + T, **dict(zip("kvspm", cache))}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.mem, self.n2 = RMSNorm(c.d_model), SelectiveMemory(c), RMSNorm(c.d_model)
        self.w_in = nn.Linear(c.d_model, 2 * c.mlp_hidden, bias=False)
        self.w_out = nn.Linear(c.mlp_hidden, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, state, reset):
        h, state = self.mem(self.n1(x), state, reset)
        x = x + self.drop(h)
        a, b = self.w_in(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.w_out(F.silu(a) * b)), state


class UmidLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        for name, p in self.named_parameters():
            if name.endswith(("tau_log", "attn_offset", "slope_raw")) or "norm" in name or "n1" in name or "n2" in name:
                continue
            if name.endswith("bias") or "s_proj" in name:
                nn.init.zeros_(p)
            elif name.endswith(("out.weight", "w_out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layers))
            else:
                nn.init.normal_(p, std=0.02)
        self.head.weight = self.embed.weight

    def forward(self, idx, targets=None, states=None):
        reset = idx == self.c.vocab_size - 1          # EOT: xotira bo'shaydi, hujjatlar aralashmaydi
        x, new = self.embed(idx), []
        for i, blk in enumerate(self.blocks):
            x, st = blk(x, None if states is None else states[i], reset)
            new.append(st)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                             targets.reshape(-1), ignore_index=-100)
        return logits, loss, new


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new=150, temperature=0.7, top_p=0.9, rep=1.15, states=None):
    """Javob qator oxiri yoki EOT da to'xtaydi. Xotira o'zgarmas hajmda (har token O(K))."""
    model.eval()
    dev = next(model.parameters()).device
    logits, _, states = model(torch.tensor([prompt_ids], device=dev), states=states)
    out = []
    for _ in range(max_new):
        lg = logits[0, -1].float()
        if out and rep != 1.0:                         # takrorlash jarimasi faqat javobning o'ziga
            seen = torch.tensor(sorted(set(out)), device=dev)
            lg[seen] = torch.where(lg[seen] > 0, lg[seen] / rep, lg[seen] * rep)
        if temperature <= 0:
            nxt = int(lg.argmax())
        else:
            probs = (lg / temperature).softmax(-1)
            sp, si = probs.sort(descending=True)
            sp[(sp.cumsum(0) - sp) > top_p] = 0
            nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
        logits, _, states = model(torch.tensor([[nxt]], device=dev), states=states)
        if nxt == tok.eot_id:
            break
        out.append(nxt)
        if "\n" in tok.decode([nxt]):
            break
    return tok.decode(out).split("\n")[0].strip(), states


# ============================== 5. O'QITISH ==============================
def device_and_amp():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        return dev, None, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # T4/P100 — fp16
    return dev, dtype, (torch.amp.GradScaler("cuda") if dtype == torch.float16 else None)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def make_opt(model, lr):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def train_steps(model, opt, batch_fn, start, total, peak, warmup, dev, dtype, scaler, on_eval=None, every=1000):
    model.train()
    t0, recent = time.time(), []
    for step in range(start, total):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, total, peak, warmup)
        x, y = batch_fn()
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            _, loss, _ = model(x.to(dev), y.to(dev))
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt) if scaler else opt.step()
        if scaler:
            scaler.update()
        opt.zero_grad(set_to_none=True)
        recent = (recent + [loss.item()])[-50:]
        if (step + 1) % 50 == 0:
            tps = (step + 1 - start) * x.numel() / (time.time() - t0)
            print(f"qadam {step + 1}/{total} | loss {sum(recent) / len(recent):.3f} | {tps:,.0f} token/s", flush=True)
        if on_eval and ((step + 1) % every == 0 or step + 1 == total):
            on_eval(step + 1)
            model.train()


@torch.no_grad()
def val_loss(model, val, seq_len, dev, dtype, n=50, bs=8):
    model.eval()
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n // bs):
        ix = torch.randint(0, len(val) - seq_len - 1, (bs,), generator=g)
        x = torch.stack([val[i:i + seq_len] for i in ix]).long().to(dev)
        y = torch.stack([val[i + 1:i + seq_len + 1] for i in ix]).long().to(dev)
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            tot += model(x, y)[1].item()
    return tot / max(n // bs, 1)


def pretrain(cfg, tok, tokens, save):
    dev, dtype, scaler = device_and_amp()
    d, L, H = PRESETS[cfg["preset"]]
    mc = ModelConfig(vocab_size=tok.vocab_size, d_model=d, n_layers=L, n_heads=H, select_k=cfg["select_k"])
    model = UmidLM(mc).to(dev)
    opt = make_opt(model, cfg["lr"])
    n_val = max(len(tokens) // 50, cfg["seq_len"] * 20)
    train, val = tokens[:-n_val], tokens[-n_val:]
    start, best = 0, float("inf")
    if (save / "last.pt").exists():                      # uzilgan joydan davom etish
        ck = torch.load(save / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]
        print(f"   davom ettirilmoqda: {start}-qadamdan", flush=True)
    print(f"== 5) Pretraining: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametr, "
          f"{len(train) / 1e6:.1f}M token, {dev}{'' if dtype is None else ', ' + str(dtype)}", flush=True)
    g = torch.Generator().manual_seed(cfg["seed"] + start)
    L_ = cfg["seq_len"]

    def batch():
        ix = torch.randint(0, len(train) - L_ - 1, (cfg["batch_size"],), generator=g)
        return (torch.stack([train[i:i + L_] for i in ix]).long(), torch.stack([train[i + 1:i + L_ + 1] for i in ix]).long())

    def on_eval(step):
        nonlocal best
        v = val_loss(model, val, L_, dev, dtype)
        if v < best:
            best = v
            torch.save({"config": asdict(mc), "model": model.state_dict(), "merges": tok.merges}, save / "base.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, save / "last.pt")
        sample, _ = generate(model, tok, tok.encode("O'zbekiston"), 30)
        print(f"== baholash {step}: val loss {v:.3f} (perplexity {math.exp(v):.1f}), eng yaxshi {best:.3f} | "
              f"namuna: O'zbekiston{sample}", flush=True)

    if start < cfg["steps"]:
        train_steps(model, opt, batch, start, cfg["steps"], cfg["lr"], cfg["warmup"], dev, dtype, scaler, on_eval,
                    cfg["eval_every"])
    return val


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = UmidLM(ModelConfig(**ck["config"])).to(dev)
    model.load_state_dict(ck["model"])
    return model, ck


def sft(cfg, tok, tokens, pairs, save):
    """Sekin SFT: loss faqat javob tokenlarida, kichik lr, uzun qizish, umumiy matn aralashmasi."""
    dev, dtype, scaler = device_and_amp()
    model, ck = load_model(save / "base.pt", dev)
    opt = make_opt(model, cfg["sft_lr"])
    examples = []
    for q, a in pairs:
        p, r = tok.encode(f"Savol: {q}\nJavob:"), tok.encode(f" {a}\n") + [tok.eot_id]
        examples.append((torch.tensor(p + r), torch.tensor([False] * len(p) + [True] * len(r))))
    L_, bs = cfg["seq_len"], cfg["batch_size"]
    n_mix = round(bs * cfg["sft_mix"])
    rng, g = random.Random(cfg["seed"]), torch.Generator().manual_seed(cfg["seed"])
    print(f"== 6) SFT: {len(examples):,} juftlik, {cfg['sft_steps']} qadam, lr {cfg['sft_lr']}", flush=True)

    def batch():
        xs, ys = [], []
        for _ in range(bs - n_mix):                    # qatorlar savol-javoblar bilan to'ldiriladi
            ids, mask = [], []
            while sum(len(t) for t in ids) < L_ + 1:
                t, m = examples[rng.randrange(len(examples))]
                ids.append(t)
                mask.append(m)
            ids, mask = torch.cat(ids)[:L_ + 1], torch.cat(mask)[:L_ + 1]
            xs.append(ids[:-1])
            ys.append(torch.where(mask[1:], ids[1:], -100))
        for i in torch.randint(0, len(tokens) - L_ - 1, (n_mix,), generator=g):
            xs.append(tokens[i:i + L_].long())
            ys.append(tokens[i + 1:i + L_ + 1].long())
        return torch.stack(xs), torch.stack(ys)

    train_steps(model, opt, batch, 0, cfg["sft_steps"], cfg["sft_lr"], cfg["sft_warmup"], dev, dtype, scaler)
    torch.save({"config": ck["config"], "model": model.state_dict(), "merges": tok.merges}, save / "chat.pt")


# ============================== 6. SINASH ==============================
# O'qitishda BO'LMAGAN savollar (boshqacha so'z bilan) va kutilgan kalit so'zlar ("/" — muqobil)
HELDOUT = [("O'zbekistonning bosh shahri qaysi?", "Toshkent"), ("Seni kim yasagan?", "Abdulhodiy"),
           ("Ismingni ayt", "Umid"), ("Quyosh tizimida nechta sayyora mavjud?", "8 ta/sakkiz"),
           ("Suvning qaynash harorati qancha?", "100 gradus"), ("Haftaning kunlari qaysilar?", "dushanba"),
           ("Yil necha oydan iborat?", "12 ta oy/12 oy/o'n ikki oy"), ("Python tilini kim yaratgan?", "Gvido/Rossum"),
           ("Necha soat uxlashim kerak?", "7-9/7 soat/8 soat/9 soat"), ("Amir Temur haqida gapir", "sarkarda/1336"),
           ("Navro'z bayrami qachon?", "mart"), ("Samarqandda qanday yodgorliklar bor?", "Registon/Bibixonim/Go'ri"),
           ("Mars qanday sayyora?", "qizil/temir"), ("Bir yilda necha kun bor?", "365"),
           ("Fotosintez haqida tushuntir", "o'simlik/kislorod"), ("Alisher Navoiy qanday asar yozgan?", "Xamsa/shoir"),
           ("Kitob o'qish nima beradi?", "bilim/so'z boyligi/fikrlash"), ("Mustaqillik bayrami qaysi kuni?", "1-sentabr"),
           ("Osh qanday tayyorlanadi?", "guruch/sabzi/go'sht"), ("Bir soat necha daqiqadan iborat?", "60 daqiqa/60 minut"),
           ("Dasturlashni o'rganmoqchiman, nimadan boshlay?", "Python"), ("Uchburchakning burchaklari yig'indisi qancha?", "180"),
           ("Kit qanday hayvon?", "sut emizuvchi/o'pka"), ("Toshkent qaysi daryo vodiysida joylashgan?", "Chirchiq"),
           ("Ulug'bek nima bilan mashhur?", "astronom/rasadxona"), ("Ibn Sinoning mashhur kitobi qaysi?", "Tib qonunlari"),
           ("Bobur qayerda tug'ilgan?", "Andijon"), ("Havoning qancha qismi kislorod?", "21 foiz"),
           ("Odam tanasida qancha suyak bor?", "206"), ("Okeanlarning eng kattasi qaysi?", "Tinch")]
DEMO = ["Salom", "Sen kimsan?", "O'zbekiston haqida gapirib ber", "Sog'lom bo'lish uchun nima qilish kerak?",
        "Sun'iy intellekt nima?", "Menga maslahat ber: ingliz tilini qanday o'rgansam bo'ladi?"]


def test(cfg, tok, val, save):
    dev, dtype, _ = device_and_amp()
    model, _ = load_model(save / "chat.pt", dev)
    print("\n== 7) Sinash", flush=True)
    v = val_loss(model, val, cfg["seq_len"], dev, dtype)
    print(f"val loss {v:.3f} (perplexity {math.exp(v):.1f})")
    hits = 0
    for q, keys in HELDOUT:
        a, _ = generate(model, tok, tok.encode(f"Savol: {q}\nJavob:"), 80, temperature=0)
        ok = any(k.lower() in a.lower() for k in keys.split("/"))
        hits += ok
        print(f"{'✓' if ok else '✗'} {q}\n    → {a}")
    print(f"\nO'qitishda bo'lmagan savollar: {hits}/{len(HELDOUT)} = {hits / len(HELDOUT):.0%}")
    print("\n== Namunaviy javoblar")
    for q in DEMO:
        print(f"Savol: {q}\nUmid:  {generate(model, tok, tok.encode(f'Savol: {q}' + chr(10) + 'Javob:'))[0]}\n")
    return model


def chat(model, tok):
    """Suhbat: butun suhbat o'zgarmas hajmli xotirada saqlanadi. /yangi — xotirani tozalash, /chiqish."""
    states = None
    while True:
        try:
            user = input("Siz: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("/chiqish", "exit", ""):
            break
        if user == "/yangi":
            states = None
            continue
        ids = tok.encode(f"Savol: {user}\nJavob:")
        reply, states = generate(model, tok, ids, states=states)
        _, _, states = model(torch.tensor([tok.encode("\n")], device=next(model.parameters()).device), states=states)
        print(f"Umid: {reply}")


# ============================== 7. HAMMASI ==============================
def check_gpu(cfg):
    """Eng boshida: GPU bormi. Bo'lmasa — vaqt sarflamay, nima qilish kerakligini aytib to'xtaydi."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB — o'qitish GPU da ({mode} aralash aniqlik)", flush=True)
    elif not cfg["allow_cpu"]:
        sys.exit("\n!!! GPU topilmadi. Bu kod GPU uchun yozilgan; CPU da o'qitish kunlab davom etadi.\n"
                 "    Colab: Runtime → Change runtime type → T4 GPU → Save, so'ng katakni qayta ishga tushiring.\n"
                 "           (\"GPU mavjud emas\" desa — bepul kvota tugagan, Kaggle'dan foydalaning.)\n"
                 "    Kaggle: Settings → Accelerator → GPU T4 x2 yoki P100, Internet → On.\n"
                 "    Oldin tayyorlangan ma'lumot saqlangan — qayta yuklanmaydi.\n"
                 "    (Baribir CPU da sinash uchun: --allow_cpu 1)")
    else:
        print("GPU yo'q — CPU da (faqat kichik sinov uchun)", flush=True)


def main(argv=None):
    cfg = dict(CONFIG)
    ap = argparse.ArgumentParser(description="Umid LLM: hammasi bitta faylda")
    for k, v in cfg.items():
        conv = (lambda x: str(x).lower() in ("1", "true", "yes", "ha")) if isinstance(v, bool) else type(v)
        ap.add_argument(f"--{k}", type=conv, default=v)
    ap.add_argument("--chat", action="store_true", help="oxirida interaktiv suhbat")
    args, _ = ap.parse_known_args(argv)                 # notebook argumentlari (-f …) e'tiborsiz qoldiriladi
    cfg.update({k: getattr(args, k) for k in cfg})
    check_gpu(cfg)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    if not cfg["save"]:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            cfg["save"] = "/content/drive/MyDrive/umid"
        except Exception:
            cfg["save"] = "/kaggle/working/umid" if os.path.exists("/kaggle/working") else "umid_natijalar"
    save = Path(cfg["save"])
    save.mkdir(parents=True, exist_ok=True)
    print(f"Natijalar: {save}", flush=True)
    tok, tokens, pairs = prepare_data(cfg, save / "data")
    val = pretrain(cfg, tok, tokens, save)
    if not (save / "chat.pt").exists():
        sft(cfg, tok, tokens, pairs, save)
    else:
        print("== 6) chat.pt tayyor — SFT o'tkazib yuborildi (qayta qilish uchun o'chiring)")
    model = test(cfg, tok, val, save)
    if args.chat:
        chat(model, tok)
    return model, tok


if __name__ == "__main__":
    MODEL, TOK = main(sys.argv[1:])
