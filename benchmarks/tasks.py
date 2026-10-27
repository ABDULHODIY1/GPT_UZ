"""'Aql' testlari uchun sintetik masalalar. Har biri (x, y) qaytaradi; y = −100 — baholanmaydi.

hop      — ko'p bosqichli mantiqiy zanjir: kontekstda f funksiyasi juftliklar sifatida
           beriladi ("x f(x)"), so'ng so'rov q uchun f(f(q)) ni topish kerak. Ikki ketma-ket
           qidiruvni bitta o'tishda bajarish — chuqur fikrlashni talab qiladi.
s3       — almashtirishlar guruhi S3 (6 element, kommutativ emas): har bir pozitsiyada shu
           paytgacha kelgan almashtirishlar ko'paytmasini aytish. Parity'dan qiyinroq holat.
mqar     — ko'p kalitli assotsiativ eslab qolish (kalit qiymat ... | so'rovlar).
"""

from __future__ import annotations

import itertools

import torch

# ---------------------------------------------------------------------------- hop
HOP_SYMBOLS = 32


def hop_batch(batch_size, n_pairs=16, hops=2, n_queries=8, generator=None):
    """Lug'at: 0 = ajratuvchi, 1..S — belgilar. f — tasodifiy funksiya (kontekstdagi kalitlar ichida)."""
    S = HOP_SYMBOLS
    xs, ys = [], []
    for _ in range(batch_size):
        keys = torch.randperm(S, generator=generator)[:n_pairs] + 1
        # f kalitlarni kalitlarga o'tkazadi, shunda f(f(q)) ham kontekstda aniqlangan
        f = {int(k): int(keys[torch.randint(0, n_pairs, (1,), generator=generator)]) for k in keys}
        order = torch.randperm(n_pairs, generator=generator)
        ctx = []
        for i in order:
            k = int(keys[i])
            ctx += [k, f[k]]
        q = keys[torch.randint(0, n_pairs, (n_queries,), generator=generator)].tolist()
        ans = []
        for a in q:
            for _ in range(hops):
                a = f[a]
            ans.append(a)
        xs.append(ctx + [0] + q)
        ys.append([-100] * (len(ctx) + 1) + ans)
    return torch.tensor(xs), torch.tensor(ys)


HOP_VOCAB = HOP_SYMBOLS + 1
HOP1, HOP2 = HOP_SYMBOLS + 1, HOP_SYMBOLS + 2
HOPMIX_VOCAB = HOP_SYMBOLS + 3


def hopmix_batch(batch_size, n_pairs=16, n_queries=8, p_two=0.5, generator=None):
    """Aralash zanjir: har so'rov oldidan belgi — HOP1 (javob f(q)) yoki HOP2 (javob f(f(q))).
    Qaytaradi: x, y va har bir baholanadigan pozitsiya necha bosqichli ekani (1/2, aks holda 0)."""
    S = HOP_SYMBOLS
    xs, ys, hs = [], [], []
    for _ in range(batch_size):
        keys = torch.randperm(S, generator=generator)[:n_pairs] + 1
        f = {int(k): int(keys[torch.randint(0, n_pairs, (1,), generator=generator)]) for k in keys}
        ctx = []
        for i in torch.randperm(n_pairs, generator=generator):
            k = int(keys[i])
            ctx += [k, f[k]]
        x, y, h = ctx + [0], [-100] * (len(ctx) + 1), [0] * (len(ctx) + 1)
        for _ in range(n_queries):
            q = int(keys[torch.randint(0, n_pairs, (1,), generator=generator)])
            two = bool(torch.rand(1, generator=generator) < p_two)
            x += [HOP2 if two else HOP1, q]
            y += [-100, f[f[q]] if two else f[q]]
            h += [0, 2 if two else 1]
        xs.append(x), ys.append(y), hs.append(h)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(hs)

# ----------------------------------------------------------------------------- s3
PERMS = list(itertools.permutations(range(3)))
_INDEX = {p: i for i, p in enumerate(PERMS)}
_COMPOSE = torch.tensor([[_INDEX[tuple(a[b[i]] for i in range(3))] for b in PERMS] for a in PERMS])
S3_VOCAB = len(PERMS)


def s3_batch(batch_size, length, generator=None):
    """x_t — tasodifiy almashtirish, y_t = x_t ∘ … ∘ x_1 (o'ngdan chapga ko'paytma)."""
    x = torch.randint(0, S3_VOCAB, (batch_size, length), generator=generator)
    y = torch.empty_like(x)
    acc = torch.zeros(batch_size, dtype=torch.long)       # 0 — birlik almashtirish
    for t in range(length):
        acc = _COMPOSE[x[:, t], acc]
        y[:, t] = acc
    return x, y


# --------------------------------------------------------------------------- mqar
def mqar_batch(batch_size, n_pairs, n_keys=64, n_values=64, generator=None):
    keys = torch.stack([torch.randperm(n_keys, generator=generator)[:n_pairs] for _ in range(batch_size)]) + 1
    vals = torch.randint(0, n_values, (batch_size, n_pairs), generator=generator) + 1 + n_keys
    order = torch.stack([torch.randperm(n_pairs, generator=generator) for _ in range(batch_size)])
    pairs = torch.stack([keys, vals], dim=2).reshape(batch_size, 2 * n_pairs)
    x = torch.cat([pairs, keys.gather(1, order)], dim=1)
    y = torch.cat([torch.full_like(pairs, -100), vals.gather(1, order)], dim=1)
    return x, y


def mqar_vocab(n_keys=64, n_values=64):
    return 1 + n_keys + n_values
