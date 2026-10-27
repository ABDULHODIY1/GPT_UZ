"""Tanlov xotirasi — Umid'ning ikkinchi, delta qoidasidan butunlay boshqa xotira algoritmi.

Xotira matritsa emas, hujjatdagi eng muhim K ta tokenning o'zi (kalit, qiymat, muhimlik, pozitsiya):

    s_t  = w_s · x_t                                   muhimlik (o'rganiladi)
    C_t  = C_{t−1} ∪ {t} dan eng muhim K tasi             tanlab saqlash (eng kami chiqariladi)
    o_t  = Σ_{j ∈ C_{t−1} ∪ {t}} σ(ℓ_tj) · v_j           aniq o'qish, softmax'siz
    ℓ_tj = τ · q̂_t·k̂_j + b + log σ(s_j) − λ · log(1 + t − j)

  * Xotira hajmi o'zgarmas (K) — KV-cache o'smaydi, generatsiyada har token O(K).
  * Har bir saqlangan token alohida o'qiladi — chiziqli xotiradagidek boshqalar bilan aralashmaydi.
  * Sigmoid: "1 ga yig'indi" majburiyati va attention sink yo'q.
  * log σ(s_j) — muhimlik o'qishga ham ta'sir qiladi, shuning uchun gradient muhim tokenlarni
    topishni o'rgatadi (saqlash qarori o'zi differensiallanmaydi).
  * λ·log(1 + masofa) — pozitsiya embedding'siz; har head yaqin yoki uzoqni o'zi tanlaydi.
  * Chiqarilgan token hech qachon qaytmaydi (undan muhimroq K ta token doim qoladi), shuning uchun
    o'qitishda chunk'li parallel hisob va generatsiyadagi tokenma-token hisob aynan bir xil.
  * Teng muhimlikda keyingi token ustun — boshlanishda (s = 0) bu "oxirgi K ta token" oynasi.

O'qib boyitish (ixtiyoriy, u va η berilsa): har token o'qigan yozuvlariga ma'lumot qaytarib yozadi,

    v_j ← v_j + η · σ(ℓ_tj) · u_t          (j ∈ C_{t−1}, ya'ni o'zidan oldingi saqlangan tokenlar)

shuning uchun xotiradagi yozuv keyingi kontekst bilan to'lib boradigan "tushuncha yozuvi"ga aylanadi
(masalan "Ibn" yozuvi keyin kelgan "Sino", "kitobi" bilan). Transformer'da o'tgan tokenning KV
yozuvi hech qachon o'zgarmaydi. Kalitlar o'zgarmagani uchun bu ham aniq parallel formulaga ega:
    o_t = Σ_j w_tj v_j + η · Σ_{i<t} (Σ_j w_tj w_ij) u_i
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


def fp32_on_cuda(fn):
    """Xotira yadrosi doim fp32 da: CUDA'da autocast (fp16/bf16) shu funksiya ichida o'chiriladi —
    holat ko'p qadam to'planadi, past aniqlik xatoni yig'ib boradi."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0].is_cuda:
            with torch.autocast("cuda", enabled=False):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def _segment_start(seg):
    """seg [B, N] (kamaymaydi) → har pozitsiya uchun o'z segmentining boshlanish indeksi [B, N]."""
    N = seg.size(1)
    idx = torch.arange(N, device=seg.device).expand_as(seg)
    is_start = torch.ones_like(seg, dtype=torch.bool)
    is_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return torch.where(is_start, idx, torch.zeros_like(idx)).cummax(1).values


@fp32_on_cuda
def selective_read(q, k, v, s, pos, seg, K, tau, bias, slope, cache=None, u=None, eta=None):
    """Bir bo'lak uchun tanlov xotirasi.

    q, k: [B, H, T, Dk] (L2-normallangan); v: [B, H, T, Dv]; s: [B, H, T] muhimlik;
    pos: [B, T] mutlaq pozitsiyalar; seg: [B, T] segmentlar (0 — kesh bilan bir hujjat, EOT dan keyin
    1, 2, …); tau, bias, slope: [H]; cache: (k, v, s, pos, valid) — [B, H, K, …] yoki None;
    u: [B, H, T, Dv], eta: [H] — o'qib boyitish (None — o'chiq).
    Qaytaradi: o [B, H, T, Dv], yangi kesh.
    """
    B, H, T, Dk = k.shape
    if cache is None:
        cache = (k.new_zeros(B, H, 0, Dk), v.new_zeros(B, H, 0, v.size(-1)), s.new_zeros(B, H, 0),
                 pos.new_zeros(B, H, 0), torch.zeros(B, H, 0, dtype=torch.bool, device=k.device))
    ck, cv, cs, cp, cm = cache
    C = ck.size(2)
    N = C + T
    k_all, v_all, s_all = torch.cat([ck, k], 2), torch.cat([cv, v], 2), torch.cat([cs, s], 2)
    p_all = torch.cat([cp, pos[:, None, :].expand(B, H, T)], 2)
    valid = torch.cat([cm, torch.ones(B, H, T, dtype=torch.bool, device=k.device)], 2)
    seg_all = torch.cat([seg.new_zeros(B, C), seg], 1)                       # [B, N]
    idx = torch.arange(N, device=k.device)

    # beats[i, j]: i token j ni siqib chiqara oladi (muhimroq, teng bo'lsa keyinroq)
    si, sj = s_all[..., :, None], s_all[..., None, :]
    beats = (si > sj) | ((si == sj) & (idx[:, None] > idx[None, :]))
    beats = beats & valid[..., :, None]
    cum = F.pad(beats.to(torch.int32).cumsum(-2), (0, 0, 1, 0))             # [B, H, N+1, N]: Σ_{i<n}
    start = _segment_start(seg_all)                                          # [B, N]
    g = C + torch.arange(T, device=k.device)                                 # so'rovlarning global indeksi
    before = cum[:, :, g]                                                    # i < g
    at_start = cum.gather(2, start[:, None, g, None].expand(B, H, T, N))     # i < segment boshi
    rank = before - at_start                                                 # [B, H, T, N]
    same_seg = seg_all[:, None, g, None] == seg_all[:, None, None, :]        # [B, 1, T, N]
    allowed = (rank < K) & valid[:, :, None, :] & (idx[None, :] < g[:, None]) & same_seg
    allowed = allowed | (idx[None, :] == g[:, None])                          # o'zi doim

    dist = (pos[:, None, :, None] - p_all[:, :, None, :]).clamp(min=0).to(q.dtype)
    logits = (tau[:, None, None] * (q @ k_all.transpose(-1, -2)) + bias[:, None, None]
              + F.logsigmoid(s_all)[:, :, None, :] - slope[:, None, None] * torch.log1p(dist))
    w = torch.sigmoid(logits) * allowed
    o = w @ v_all
    if u is not None:
        w_ex = w.masked_fill(idx[None, :] == g[:, None], 0.0)              # token o'z yozuvini boyitmaydi
        # bo'lak ichida oldinroq kelgan i tokenlar t o'qiydigan yozuvlarni boyitgan: E[t, i] = Σ_j w_tj w_ij
        E = (w @ w_ex.transpose(-1, -2)).tril(-1)
        o = o + eta[:, None, None] * (E @ u)
        v_all = v_all + eta[:, None, None] * (w_ex.transpose(-1, -2) @ u)  # saqlanadigan yozuvlar yangilanadi

    # yangi kesh: oxirgi segmentdagi eng muhim K ta token
    last = start[:, -1]                                                      # [B]
    rank_end = cum[:, :, N] - cum.gather(2, last[:, None, None, None].expand(B, H, 1, N)).squeeze(2)
    keep = (rank_end < K) & valid & (seg_all == seg_all[:, -1:])[:, None, :]
    n_keep = min(K, N)
    # saqlanganlar vaqt tartibida (teng muhimlikda "keyingisi ustun" qoidasi keyingi bo'lakda ham to'g'ri ishlashi uchun)
    order = torch.where(keep, (N - idx).to(s.dtype), torch.full_like(s_all, -1.0)).topk(n_keep, -1).indices
    take = lambda x: x.gather(2, order if x.dim() == 3 else order[..., None].expand(*order.shape, x.size(-1)))  # noqa: E731
    new = (take(k_all), take(v_all), take(s_all), take(p_all), keep.gather(2, order))
    if n_keep < K:   # kesh har doim K o'lchamda (bo'sh joylar valid=False)
        pad = K - n_keep
        new = tuple(F.pad(x, (0, 0, 0, pad)) if x.dim() == 4 else F.pad(x, (0, pad)) for x in new)
    return o, new


@torch.no_grad()
def selective_read_reference(q, k, v, s, pos, seg, K, tau, bias, slope, u=None, eta=None):
    """Etalon (sekin, aniq ta'rif bo'yicha): har token uchun kesh ro'yxati qo'lda yuritiladi."""
    B, H, T, _ = k.shape
    out = torch.zeros(B, H, T, v.size(-1), dtype=v.dtype)
    for b in range(B):
        for h in range(H):
            cache, val = [], {}
            for t in range(T):
                if t > 0 and seg[b, t] != seg[b, t - 1]:
                    cache = []
                val[t] = v[b, h, t].clone()
                cand = cache + [t]
                for j in cand:
                    d = float(pos[b, t] - pos[b, j])
                    lg = (tau[h] * (q[b, h, t] @ k[b, h, j]) + bias[h] + F.logsigmoid(s[b, h, j])
                          - slope[h] * torch.log1p(torch.tensor(d, dtype=q.dtype)))
                    out[b, h, t] += torch.sigmoid(lg) * val[j]
                    if u is not None and j != t:   # o'qilgan yozuv boyitiladi (o'qishdan keyin)
                        val[j] = val[j] + eta[h] * torch.sigmoid(lg) * u[b, h, t]
                cache = cand
                if len(cache) > K:   # eng kam muhim (teng bo'lsa eng eskisi) chiqadi
                    cache.remove(min(cache, key=lambda j: (s[b, h, j].item(), j)))
    return out
