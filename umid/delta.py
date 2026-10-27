"""Gated Delta Memory — Umid modelining "attention" o'rnini bosuvchi yadrosi.

Har bir head o'zining qat'iy o'lchamli xotira matritsasiga ega: S ∈ R^{Dk×Dv}.
Har bir token uchun xotira quyidagicha yangilanadi (gated delta rule):

    S_t = α_t · (I − b_t k_t k_tᵀ) · S_{t−1} + β_t · k_t v_tᵀ      (odatda b_t = β_t)
    o_t = S_tᵀ q_t

    α_t ∈ (0, 1]  — unutish darvozasi (eskirgan ma'lumotni o'chiradi)
    β_t ∈ (0, 2)  — yozish kuchi; β > 1 bo'lsa (I − β k kᵀ) manfiy xos qiymatga
                     ega bo'ladi va model holatni kuzata oladi (parity, hisoblash)
    b_t           — o'chirish kuchi (erase). b = β — klassik delta rule (eski qiymat yangisiga
                     almashtiriladi); b < β — yig'ish (bir nechta qiymat ustma-ust saqlanadi, masalan
                     savoldagi barcha so'zlarning "xulosasi"); har token o'zi tanlaydi
    k_t, q_t      — L2-normallangan kalit va so'rov

Bu Transformer attention'dan farqli ravishda:
  * softmax yo'q → "attention sink" va majburiy "1 ga yig'indi" muammosi yo'q;
  * KV-cache yo'q → generatsiyada xotira O(1), har token uchun vaqt O(1);
  * o'qitishda vaqt O(T) (chunk'li parallel algoritm), O(T²) emas;
  * pozitsiya embedding kerak emas → uzunroq kontekstga ekstrapolyatsiya.

Fayldagi uch funksiya bir xil matematikani hisoblaydi:
  delta_rule_recurrent — token-ma-token, eng sodda (test uchun etalon);
  delta_rule_chunked   — o'qitish uchun tez, chunk ichida parallel;
  delta_rule_step      — generatsiya uchun bitta qadam.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def delta_rule_recurrent(q, k, v, g, beta, initial_state=None, read_fn=None, n_reads=1, erase=None):
    """Etalon (sekin) implementatsiya.

    q, k: [B, H, T, Dk]; v: [B, H, T, Dv]; g: [B, H, T] yoki [B, H, T, Dk] (log α, ≤ 0);
    beta, erase: [B, H, T] (erase=None → β). Qaytaradi: o [B, H, T, Dv], yakuniy holat [B, H, Dk, Dv].
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        o, S = delta_rule_step(q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t], S, read_fn, n_reads,
                               None if erase is None else erase[:, :, t])
        outs.append(o)
    return torch.stack(outs, dim=2), S


def delta_rule_step(q, k, v, g, beta, state, read_fn=None, n_reads=1, erase=None):
    """Bitta token uchun yangilanish. q, k: [B, H, Dk]; v: [B, H, Dv]; beta, erase: [B, H];
    g: [B, H] (head bo'yicha unutish) yoki [B, H, Dk] (har bir kalit-kanal o'z tezligida).

    n_reads > 1: xotiradan qayta so'rash — o¹ = Sᵀq, q^{r+1} = read_fn(o^r), o^{r+1} = Sᵀq^{r+1};
    natija [B, H, n_reads, Dv]."""
    a = g.exp()
    S = state * (a[..., None, None] if a.dim() == 2 else a[..., :, None])
    pred = torch.einsum("bhkv,bhk->bhv", S, k)            # xotira hozir k uchun nimani "eslaydi"
    b = beta if erase is None else erase
    delta = v * beta[..., None] - pred * b[..., None]    # xato bo'yicha tuzatish (delta rule)
    S = S + k[..., :, None] * delta[..., None, :]
    outs = [torch.einsum("bhkv,bhk->bhv", S, q)]
    for _ in range(n_reads - 1):
        outs.append(torch.einsum("bhkv,bhk->bhv", S, read_fn(outs[-1])))
    return (outs[0] if n_reads == 1 else torch.stack(outs, 2)), S


def delta_rule_chunked(q, k, v, g, beta, chunk_size=64, initial_state=None, read_fn=None, n_reads=1, erase=None):
    """Chunk'li parallel algoritm (WY/UT ko'rinishi), vaqt O(T·C), xotira O(T·C).

    Chunk ichida rekurrensiyani uchburchak tenglamalar sistemasi orqali yechamiz:
      (I + A) · N = β·V − (b·K ⊙ γ) · S₀,   A[t,s] = b_t (k_t ⊙ γ_t/γ_s)·k_s  (s < t)
    bu yerda N — har bir token xotiraga yozgan "yangi" qiymat, γ — chunk boshidan
    to'plangan unutish koeffitsienti. Natija delta_rule_recurrent bilan bir xil.

    g: [B, H, T] (head bo'yicha) yoki [B, H, T, Dk] (kanal bo'yicha). Kanal rejimida γ_t/γ_s
    ko'paytuvchilarga ajratiladi (exp(gc_t)·exp(−gc_s)), shuning uchun chunk 16 tokendan
    oshmaydi va har token uchun g ≥ −5 bo'lishi kerak (fp32 da toshib ketmaslik uchun).

    n_reads > 1 (xotiradan qayta so'rash): har bir chunk'da bir xil (u, w, S) bilan yangi so'rovlar
    q^{r+1} = read_fn(o^r) o'qiladi; natija [B, H, T, n_reads, Dv].
    """
    in_dtype = v.dtype
    dtype = torch.promote_types(in_dtype, torch.float32)   # bf16/fp16 bo'lsa ham fp32 da hisoblaymiz
    erase = beta if erase is None else erase
    q, k, v, g, beta, erase = (x.to(dtype) for x in (q, k, v, g, beta, erase))
    channel = g.dim() == 4
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = min(chunk_size, 16) if channel else chunk_size
    pad = (C - T % C) % C
    if pad:
        # Bo'sh tokenlar: k=0, β=0 (hech narsa yozmaydi), g=0 (unutmaydi) → holat o'zgarmaydi.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g = F.pad(g, (0, 0, 0, pad) if channel else (0, pad))
        beta, erase = F.pad(beta, (0, pad)), F.pad(erase, (0, pad))
    N = (T + pad) // C
    q, k, v = (x.reshape(B, H, N, C, -1) for x in (q, k, v))
    g, beta, erase = g.reshape(B, H, N, C, *g.shape[3:]), beta.reshape(B, H, N, C), erase.reshape(B, H, N, C)

    gc = g.cumsum(3)                                        # log γ_t
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    k_beta = k * erase[..., None]                          # o'chirish qismi (b·k)
    if channel:
        eg = gc.exp()
        k_in = k * (-gc).exp()
        A = (k_beta * eg) @ k_in.transpose(-1, -2)
        qk = ((q * eg) @ k_in.transpose(-1, -2)).masked_fill(~tril, 0.0)
        w_rhs, q_decay = k_beta * eg, q * eg
        k_to_end = k * (gc[..., -1:, :] - gc).exp()
        chunk_decay = gc[..., -1, :].exp()[..., :, None]     # [B, H, N, Dk, 1]
    else:
        decay = torch.where(tril, gc[..., :, None] - gc[..., None, :], float("-inf")).exp()
        A = (k_beta @ k.transpose(-1, -2)) * decay
        qk = (q @ k.transpose(-1, -2)) * decay              # chunk ichidagi o'qish
        w_rhs, q_decay = k_beta * gc.exp()[..., None], q * gc.exp()[..., None]
        k_to_end = k * (gc[..., -1:] - gc).exp()[..., None]
        chunk_decay = gc[..., -1].exp()[..., None, None]     # [B, H, N, 1, 1]
    A = A.masked_fill(~torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1), 0.0)
    L = torch.eye(C, dtype=q.dtype, device=q.device) + A    # pastki uchburchak, diagonal = 1
    uw = torch.linalg.solve_triangular(L, torch.cat([v * beta[..., None], w_rhs], -1), upper=False, unitriangular=True)
    u, w = uw.split([Dv, Dk], dim=-1)

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.to(dtype)
    if n_reads > 1:   # qayta so'rovlar uchun chunk'ning "o'qish" ma'lumotlari
        read_parts = (eg, k_in) if channel else (gc.exp()[..., None], k, decay)
    else:
        read_parts = ()
    outs = []
    # unbind (indekslash emas): backward'da har bir chunk uchun to'liq nol tensor yaratilmaydi
    for u_n, w_n, qd_n, qk_n, ke_n, cd_n, *rp in zip(*(x.unbind(2) for x in (u, w, q_decay, qk, k_to_end, chunk_decay,
                                                                            *read_parts))):
        new = u_n - w_n @ S
        o_n = [qd_n @ S + qk_n @ new]
        for _ in range(n_reads - 1):
            qr = read_fn(o_n[-1])
            if channel:
                qr_decay = qr * rp[0]
                qr_k = (qr_decay @ rp[1].transpose(-1, -2)).masked_fill(~tril, 0.0)
            else:
                qr_decay = qr * rp[0]
                qr_k = (qr @ rp[1].transpose(-1, -2)) * rp[2]
            o_n.append(qr_decay @ S + qr_k @ new)
        outs.append(o_n[0] if n_reads == 1 else torch.stack(o_n, 3))
        S = S * cd_n + ke_n.transpose(-1, -2) @ new
    o = torch.stack(outs, dim=2)
    o = o.reshape(B, H, N * C, *o.shape[4:])[:, :, :T]
    return o.to(in_dtype), S
