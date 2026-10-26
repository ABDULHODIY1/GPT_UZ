"""Gated Delta Memory — Umid modelining "attention" o'rnini bosuvchi yadrosi.

Har bir head o'zining qat'iy o'lchamli xotira matritsasiga ega: S ∈ R^{Dk×Dv}.
Har bir token uchun xotira quyidagicha yangilanadi (gated delta rule):

    S_t = α_t · (I − β_t k_t k_tᵀ) · S_{t−1} + β_t · k_t v_tᵀ
    o_t = S_tᵀ q_t

    α_t ∈ (0, 1]  — unutish darvozasi (eskirgan ma'lumotni o'chiradi)
    β_t ∈ (0, 2)  — yozish kuchi; β > 1 bo'lsa (I − β k kᵀ) manfiy xos qiymatga
                     ega bo'ladi va model holatni kuzata oladi (parity, hisoblash)
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


def delta_rule_recurrent(q, k, v, g, beta, initial_state=None):
    """Etalon (sekin) implementatsiya.

    q, k: [B, H, T, Dk]; v: [B, H, T, Dv]; g: [B, H, T] (log α, ≤ 0);
    beta: [B, H, T]. Qaytaradi: o [B, H, T, Dv], yakuniy holat [B, H, Dk, Dv].
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        o, S = delta_rule_step(q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t], S)
        outs.append(o)
    return torch.stack(outs, dim=2), S


def delta_rule_step(q, k, v, g, beta, state):
    """Bitta token uchun yangilanish. q, k: [B, H, Dk]; v: [B, H, Dv]; g, beta: [B, H]."""
    S = state * g.exp()[..., None, None]
    pred = torch.einsum("bhkv,bhk->bhv", S, k)            # xotira hozir k uchun nimani "eslaydi"
    delta = (v - pred) * beta[..., None]                 # xato bo'yicha tuzatish (delta rule)
    S = S + k[..., :, None] * delta[..., None, :]
    o = torch.einsum("bhkv,bhk->bhv", S, q)
    return o, S


def delta_rule_chunked(q, k, v, g, beta, chunk_size=64, initial_state=None):
    """Chunk'li parallel algoritm (WY/UT ko'rinishi), vaqt O(T·C), xotira O(T·C).

    Chunk ichida rekurrensiyani uchburchak tenglamalar sistemasi orqali yechamiz:
      (I + A) · N = β·V − (β·K ⊙ γ) · S₀,   A[t,s] = β_t (k_t·k_s) γ_t/γ_s  (s < t)
    bu yerda N — har bir token xotiraga yozgan "yangi" qiymat, γ — chunk boshidan
    to'plangan unutish koeffitsienti. Natija delta_rule_recurrent bilan bir xil.
    """
    in_dtype = v.dtype
    dtype = torch.promote_types(in_dtype, torch.float32)   # bf16/fp16 bo'lsa ham fp32 da hisoblaymiz
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    if pad:
        # Bo'sh tokenlar: k=0, β=0 (hech narsa yozmaydi), g=0 (unutmaydi) → holat o'zgarmaydi.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    N = (T + pad) // C
    q, k, v = (x.reshape(B, H, N, C, -1) for x in (q, k, v))
    g, beta = g.reshape(B, H, N, C), beta.reshape(B, H, N, C)

    gc = g.cumsum(-1)                                       # log γ_t
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(tril, gc[..., :, None] - gc[..., None, :], float("-inf")).exp()

    k_beta = k * beta[..., None]
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    A = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0.0)
    L = torch.eye(C, dtype=q.dtype, device=q.device) + A    # pastki uchburchak, diagonal = 1
    u = torch.linalg.solve_triangular(L, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(L, k_beta * gc.exp()[..., None], upper=False, unitriangular=True)

    qk = (q @ k.transpose(-1, -2)) * decay                  # chunk ichidagi o'qish
    q_decay = q * gc.exp()[..., None]
    k_to_end = k * (gc[..., -1:] - gc).exp()[..., None]
    chunk_decay = gc[..., -1].exp()

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.to(dtype)
    outs = []
    # unbind (indekslash emas): backward'da har bir chunk uchun to'liq nol tensor yaratilmaydi
    for u_n, w_n, qd_n, qk_n, ke_n, cd_n in zip(*(x.unbind(2) for x in (u, w, q_decay, qk, k_to_end, chunk_decay))):
        new = u_n - w_n @ S
        outs.append(qd_n @ S + qk_n @ new)
        S = S * cd_n[..., None, None] + ke_n.transpose(-1, -2) @ new
    o = torch.stack(outs, dim=2).reshape(B, H, N * C, Dv)[:, :, :T]
    return o.to(in_dtype), S
