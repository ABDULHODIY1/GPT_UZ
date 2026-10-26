"""Optimal xotira — har bir tokenda eng kichik kvadratlar masalasini aniq yechadigan qatlam.

Delta rule xotirani har tokenda bitta gradient qadami bilan yangilaydi. Bu yerda esa xotira
butun tarix bo'yicha regulyarizatsiyalangan chiziqli regressiyaning aniq yechimini beradi:

    S_t = argmin_S  Σ_{s≤t} γ_{t,s} β_s ‖Sᵀk_s − v_s‖² + ‖Λ^{1/2} S‖²

Yechim ikki matritsa orqali yoziladi (head uchun):
    H_t = α_t H_{t−1} + β_t k_t k_tᵀ          (Dk × Dk, kalitlar kovariatsiyasi)
    G_t = α_t G_{t−1} + β_t k_t v_tᵀ          (Dk × Dv)
    o_t = G_tᵀ (H_t + Λ)⁻¹ q_t

Natijada o'zaro korrelyatsiyalangan kalitlar bir-birini buzmaydi: Dk tagacha chiziqli
erkli kalit aniq eslab qolinadi. Narxi: har token uchun Dk×Dk sistemani yechish (Cholesky).
"""

from __future__ import annotations

import torch


def optimal_memory_step(q, k, v, g, beta, lam, state):
    """q, k: [B, H, Dk]; v: [B, H, Dv]; g, beta: [B, H]; lam: [H, Dk] (> 0);
    state: (Hm [B, H, Dk, Dk], G [B, H, Dk, Dv])."""
    Hm, G = state
    a = g.exp()[..., None, None]
    bk = beta[..., None] * k
    Hm = a * Hm + bk[..., :, None] * k[..., None, :]
    G = a * G + bk[..., :, None] * v[..., None, :]
    L = torch.linalg.cholesky(Hm + torch.diag_embed(lam).expand_as(Hm))
    x = torch.cholesky_solve(q[..., None], L)[..., 0]
    o = torch.einsum("bhkv,bhk->bhv", G, x)
    return o, (Hm, G)


def optimal_memory_recurrent(q, k, v, g, beta, lam, initial_state=None):
    """Etalon: token-ma-token. Shakllar delta_rule_recurrent bilan bir xil."""
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    st = initial_state or (q.new_zeros(B, H, Dk, Dk), q.new_zeros(B, H, Dk, Dv))
    outs = []
    for t in range(T):
        o, st = optimal_memory_step(q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t], lam, st)
        outs.append(o)
    return torch.stack(outs, 2), st


def optimal_memory_chunked(q, k, v, g, beta, lam, chunk_size=32, initial_state=None):
    """Chunk ichida barcha H_t lar parallel quriladi va Cholesky bilan birdaniga yechiladi;
    chunk'lar orasida faqat (H, G) uzatiladi. Xotira O(T·Dk²), vaqt O(T·(C·Dk² + Dk³))."""
    in_dtype = v.dtype
    dtype = torch.promote_types(in_dtype, torch.float32)
    q, k, v, g, beta, lam = (x.to(dtype) for x in (q, k, v, g, beta, lam))
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    Hm, G = initial_state or (q.new_zeros(B, H, Dk, Dk), q.new_zeros(B, H, Dk, Dv))
    Hm, G = Hm.to(dtype), G.to(dtype)
    eye_lam = torch.diag_embed(lam)                          # [H, Dk, Dk]
    outs = []
    for s in range(0, T, chunk_size):
        sl = slice(s, min(s + chunk_size, T))
        qc, kc, vc, gc, bc = q[:, :, sl], k[:, :, sl], v[:, :, sl], g[:, :, sl].cumsum(-1), beta[:, :, sl]
        C = kc.shape[2]
        tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
        decay = torch.where(tril, gc[..., :, None] - gc[..., None, :], float("-inf")).exp()   # [B,H,C,C]
        w = decay * bc[..., None, :]                         # γ_{t,s} β_s
        start = gc.exp()[..., None, None]                    # γ_t (chunk boshidan)
        # H_t = γ_t H₀ + Σ_s w[t,s] k_s k_sᵀ  — barcha t uchun birdaniga
        Ht = start * Hm[:, :, None] + torch.einsum("bhts,bhsi,bhsj->bhtij", w, kc, kc)
        L = torch.linalg.cholesky(Ht + eye_lam[None, :, None])
        x = torch.cholesky_solve(qc[..., None], L)[..., 0]   # (H_t + Λ)⁻¹ q_t
        o = gc.exp()[..., None] * torch.einsum("bhkv,bhtk->bhtv", G, x) + (w * (x @ kc.transpose(-1, -2))) @ vc
        outs.append(o)
        end = (gc[..., -1:] - gc).exp() * bc                 # γ_{C,s} β_s
        a_end = gc[..., -1].exp()[..., None, None]
        Hm = a_end * Hm + torch.einsum("bhs,bhsi,bhsj->bhij", end, kc, kc)
        G = a_end * G + torch.einsum("bhs,bhsi,bhsj->bhij", end, kc, vc)
    return torch.cat(outs, 2).to(in_dtype), (Hm, G)
