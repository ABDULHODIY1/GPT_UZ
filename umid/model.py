"""Umid LM — Transformer attention'siz til modeli.

Blok tuzilishi (pre-norm):
    x = x + GatedDeltaMemory(RMSNorm(x))     # attention o'rniga
    x = x + SwiGLU(RMSNorm(x))

Pozitsiya embedding ishlatilmaydi: tartib haqidagi ma'lumotni qisqa kauzal konvolyutsiya
va unutish darvozasi (α) beradi, shuning uchun model o'qitilganidan uzunroq matnda ham ishlaydi.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta import delta_rule_chunked, delta_rule_step
from .optimal import optimal_memory_chunked, optimal_memory_step


@dataclass
class UmidConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: int = 64            # kalit/so'rov o'lchami (Dk)
    expand_v: float = 1.0         # qiymat o'lchami Dv = head_dim * expand_v
    conv_size: int = 4
    mlp_ratio: float = 8 / 3
    chunk_size: int = 64
    allow_neg_eigval: bool = True  # β ∈ (0, 2): holatni kuzatish (parity, sanash) imkoniyati
    decay: str = "head"           # "head": head uchun bitta α; "channel": har bir kalit-kanal o'z α si
                                  # (LM da biroz yaxshi, lekin eslab qolish va parity'ni yomonlashtirdi)
    decay_rank: int = 32          # "channel" rejimida α proyeksiyasining rangi (D → r → H·Dk)
    n_writes: int = 1             # har token xotiraga necha marta yozadi (Householder ko'paytmasi)
    n_reads: int = 2              # xotiradan qayta so'rash: o¹ = Sᵀq, q² = norm(W·o¹), o² = Sᵀq² (v2; v1 = 1)
    n_loops: int = 1              # fikrlash halqasi: qatlamlar steki necha marta qayta ishlatiladi
    mixer: str = "delta"          # "delta": Gated Delta Memory; "optimal": aniq eng kichik kvadratlar xotirasi
    frame: bool = False           # "optimal" uchun: kalit/so'rovlar aks ettirishlar bilan aylanadigan tizimda
    mlp_hidden_size: int = 0      # 0 bo'lmasa MLP o'lchamini aniq belgilaydi (parametrlarni tenglashtirish uchun)
    dropout: float = 0.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def value_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def mlp_hidden(self) -> int:
        if self.mlp_hidden_size:
            return self.mlp_hidden_size
        return 64 * math.ceil(self.d_model * self.mlp_ratio / 64)


PRESETS = {
    "tiny": dict(d_model=256, n_layers=4, n_heads=4, head_dim=64),
    "small": dict(d_model=384, n_layers=6, n_heads=6, head_dim=64),
    "base": dict(d_model=768, n_layers=12, n_heads=12, head_dim=64),
    "large": dict(d_model=1024, n_layers=24, n_heads=16, head_dim=64),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.to(torch.promote_types(x.dtype, torch.float32))
        if hasattr(F, "rms_norm"):                     # torch ≥ 2.4: bitta fused yadro
            xf = F.rms_norm(xf, (xf.shape[-1],), eps=self.eps)
        else:
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class ShortConv(nn.Module):
    """Kauzal depthwise konvolyutsiya (yaqin tokenlar orasidagi lokal bog'lanish uchun).

    cache — oldingi (K−1) ta kirish; generatsiyada holat sifatida uzatiladi.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, cache=None):
        B, T, C = x.shape
        K = self.kernel_size
        x = x.transpose(1, 2)
        w = self.weight.to(x.dtype)
        if cache is None:
            # o'qitish: chapdan nol bilan to'ldirish conv1d ichida (qo'shimcha nusxasiz)
            y = F.conv1d(x, w, padding=K - 1, groups=C)[..., :T]
            tail = F.pad(x[..., -(K - 1):], (max(K - 1 - T, 0), 0))
        else:
            xx = torch.cat([cache.to(x.dtype), x], dim=-1)
            y = F.conv1d(xx, w, groups=C)
            tail = xx[..., xx.shape[-1] - (K - 1):]
        return F.silu(y).transpose(1, 2), tail


class GatedDeltaMemory(nn.Module):
    """Attention o'rnini bosuvchi qatlam: har bir head — qat'iy o'lchamli assotsiativ xotira."""

    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.cfg = cfg
        D, H, dk, dv = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.value_dim
        n = cfg.n_writes
        self.qkv_proj = nn.Linear(D, H * (dk + n * (dk + dv)), bias=False)
        self.qkv_conv = ShortConv(H * (dk + n * (dk + dv)), cfg.conv_size)
        self.b_proj = nn.Linear(D, n * H, bias=True)  # β — qancha kuch bilan yozish
        if n > 1:   # qo'shimcha yozishlar deyarli o'chiq boshlanadi (β ≈ 0.1), birinchisini buzmaydi
            self._bias_init = {"b_proj": torch.cat([torch.zeros(H), torch.full(((n - 1) * H,), -3.0)])}
        n_decay = H * dk if cfg.decay == "channel" else H
        if cfg.decay == "channel":                    # α — qancha unutish (har bir kanal uchun)
            self.a_proj = nn.Sequential(nn.Linear(D, cfg.decay_rank, bias=False),
                                        nn.Linear(cfg.decay_rank, n_decay, bias=False))
        else:                                         # α — qancha unutish (head uchun bitta)
            self.a_proj = nn.Linear(D, H, bias=False)
        self.g_proj = nn.Linear(D, H * dv, bias=False)  # chiqish darvozasi
        self.o_norm = RMSNorm(dv, cfg.norm_eps)
        self.o_proj = nn.Linear(H * dv, D, bias=False)
        if cfg.n_reads > 1:
            assert n == 1, "n_reads > 1 faqat n_writes = 1 bilan"
            # javobni yangi so'rovga aylantirish (har head uchun Dv → Dk) va o'qishlarni tanlash
            self.read_proj = nn.Parameter(torch.randn(H, dv, dk) / math.sqrt(dv))
            self.read_mix = nn.Linear(D, H * cfg.n_reads, bias=True)

        # Ko'p masshtabli xotira: har bir head turli tezlikda unutadi (Mamba2 uslubidagi init).
        A = torch.empty(H).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(n_decay) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus⁻¹(dt)

    def forward(self, x, state=None):
        cfg = self.cfg
        B, T, _ = x.shape
        H, dk, dv, n = cfg.n_heads, cfg.head_dim, cfg.value_dim, cfg.n_writes
        st = state or {}
        fdt = torch.promote_types(x.dtype, torch.float32)   # xotira hisob-kitobi kamida fp32 da
        qkv, conv_cache = self.qkv_conv(self.qkv_proj(x), st.get("conv"))
        q, k, v = qkv.split([H * dk, n * H * dk, n * H * dv], dim=-1)
        q = F.normalize(q.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
        k = F.normalize(k.reshape(B, T, n, H, dk).to(fdt), dim=-1).permute(0, 3, 1, 2, 4)  # [B, H, T, n, dk]
        v = v.reshape(B, T, n, H, dv).to(fdt).permute(0, 3, 1, 2, 4)

        beta = torch.sigmoid(self.b_proj(x).to(fdt)).view(B, T, n, H).permute(0, 3, 1, 2)   # [B, H, T, n]
        if cfg.allow_neg_eigval:
            beta = beta * 2.0
        g = F.softplus(self.a_proj(x).to(fdt) + self.dt_bias.to(fdt))
        if cfg.decay == "channel":
            g = (-self.A_log.to(fdt).exp()[:, None] * g.view(B, T, H, dk)).clamp(min=-5.0).transpose(1, 2)
        else:
            g = (-self.A_log.to(fdt).exp() * g).transpose(1, 2)    # g: [B, H, T] yoki [B, H, T, Dk]

        S0 = st.get("S")
        R = cfg.n_reads
        read_fn = None
        if R > 1:
            W = self.read_proj.to(fdt)
            read_fn = lambda o_: F.normalize(torch.einsum("bh...v,hvk->bh...k", o_, W), dim=-1)  # noqa: E731
        if T == 1 and S0 is not None:
            S = S0
            for j in range(n):   # unutish faqat birinchi yozishda, o'qish oxirgisidan keyin
                o, S = delta_rule_step(q[:, :, 0], k[:, :, 0, j], v[:, :, 0, j],
                                       g[:, :, 0] if j == 0 else torch.zeros_like(g[:, :, 0]), beta[:, :, 0, j], S,
                                       read_fn, R)
            o = o[:, :, None]
        elif n == 1:
            o, S = delta_rule_chunked(q, k[:, :, :, 0], v[:, :, :, 0], g, beta[..., 0], cfg.chunk_size, S0, read_fn, R)
        else:
            # Har bir token n ta ketma-ket "kichik qadam"ga yoyiladi: S ← Π_j (I − β_j k_j k_jᵀ) S + …
            # Bu xotirani aks ettirishlar ko'paytmasi (aylanish, almashtirish) bilan o'zgartirish imkonini beradi.
            gz = torch.zeros_like(g).unsqueeze(3).expand(*g.shape[:3], n - 1, *g.shape[3:])
            g_seq = torch.cat([g.unsqueeze(3), gz], 3).flatten(2, 3)
            q_seq = torch.cat([q.new_zeros(B, H, T, n - 1, dk), q.unsqueeze(3)], 3).flatten(2, 3)
            o, S = delta_rule_chunked(q_seq, k.flatten(2, 3), v.flatten(2, 3), g_seq, beta.flatten(2, 3),
                                      cfg.chunk_size, S0)
            o = o.view(B, H, T, n, dv)[:, :, :, -1]

        o = self.o_norm(o.transpose(1, 2).to(x.dtype))           # [B, T, H, dv] yoki [B, T, H, R, dv]
        if R > 1:   # har token qaysi "bosqich" javobi kerakligini o'zi tanlaydi
            mix = self.read_mix(x).view(B, T, H, R).softmax(-1).to(o.dtype)
            o = (o * mix[..., None]).sum(3)
        o = o * F.silu(self.g_proj(x)).view(B, T, H, dv)
        out = self.o_proj(o.reshape(B, T, H * dv))
        return out, {"conv": conv_cache, "S": S}


class OptimalMemory(nn.Module):
    """Xotira har tokenda eng kichik kvadratlar masalasini aniq yechadi: o = Gᵀ(H + Λ)⁻¹q.

    H — kalitlar kovariatsiyasi, G — kalit-qiymat bog'lanishi (ikkalasi ham α bilan unutiladi).
    Korrelyatsiyalangan kalitlar bir-birini buzmaydi (delta rule'dan farqli). umid/optimal.py ga qarang.
    """

    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.cfg = cfg
        D, H, dk, dv = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.value_dim
        n_in = H * (2 * dk + dv) + (H * dk if cfg.frame else 0)
        self.qkv_proj = nn.Linear(D, n_in, bias=False)
        self.qkv_conv = ShortConv(n_in, cfg.conv_size)
        self.b_proj = nn.Linear(D, H, bias=True)
        self.a_proj = nn.Linear(D, H, bias=False)
        self.g_proj = nn.Linear(D, H * dv, bias=False)
        self.o_norm = RMSNorm(dv, cfg.norm_eps)
        self.o_proj = nn.Linear(H * dv, D, bias=False)
        A = torch.empty(H).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.lam_raw = nn.Parameter(torch.full((H, dk), math.log(math.e - 1)))   # softplus → Λ = 1
        if cfg.frame:
            # Koordinata tizimi P_t = (I − β_u u uᵀ) P_{t−1}; boshida β_u ≈ 0 (P ≈ I).
            self.frame_proj = nn.Linear(D, H, bias=True)
            self._bias_init = {"frame_proj": torch.full((H,), -4.0)}

    def forward(self, x, state=None):
        cfg = self.cfg
        B, T, _ = x.shape
        H, dk, dv = cfg.n_heads, cfg.head_dim, cfg.value_dim
        st = state or {}
        fdt = torch.promote_types(x.dtype, torch.float32)
        qkv, conv_cache = self.qkv_conv(self.qkv_proj(x), st.get("conv"))
        q, k, v, u = qkv.split([H * dk, H * dk, H * dv, H * dk if cfg.frame else 0], dim=-1)
        q = F.normalize(q.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
        k = F.normalize(k.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
        v = v.reshape(B, T, H, dv).transpose(1, 2).to(fdt)
        beta = torch.sigmoid(self.b_proj(x).to(fdt)).transpose(1, 2)
        g = (-self.A_log.to(fdt).exp() * F.softplus(self.a_proj(x).to(fdt) + self.dt_bias.to(fdt))).transpose(1, 2)
        lam = F.softplus(self.lam_raw.to(fdt)) + 1e-4
        prev = (st["H"], st["G"]) if "H" in st else None
        new_state = {"conv": conv_cache}
        if cfg.frame:
            u = F.normalize(u.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
            bu = (2.0 * torch.sigmoid(self.frame_proj(x).to(fdt))).transpose(1, 2)
            P0 = st["P"] if "P" in st else torch.eye(dk, dtype=fdt, device=x.device).expand(B, H, dk, dk)
            # q̂_t = P_tᵀ q_t, k̂_t = P_tᵀ k_t — delta yadrosi bilan (S₀ = P, v = 0, unutishsiz)
            zeros = torch.zeros_like(bu)
            if T == 1 and "P" in st:
                P = P0 - bu[:, :, 0, None, None] * u[:, :, 0, :, None] * (u[:, :, 0, None, :] @ P0)
                q, k = (torch.einsum("bhij,bhti->bhtj", P, z) for z in (q, k))
            else:
                qk, P = delta_rule_chunked(torch.cat([q, k]), u.repeat(2, 1, 1, 1), torch.zeros_like(k).repeat(2, 1, 1, 1),
                                           zeros.repeat(2, 1, 1), bu.repeat(2, 1, 1), cfg.chunk_size, P0.repeat(2, 1, 1, 1))
                q, k, P = qk[:B], qk[B:], P[:B]
            new_state["P"] = P
        if T == 1 and prev is not None:
            o, (Hm, G) = optimal_memory_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], beta[:, :, 0], lam, prev)
            o = o[:, :, None]
        else:
            o, (Hm, G) = optimal_memory_chunked(q, k, v, g, beta, lam, 8, prev)
        o = self.o_norm(o.transpose(1, 2).to(x.dtype))
        o = o * F.silu(self.g_proj(x)).view(B, T, H, dv)
        new_state.update(H=Hm, G=G)
        return self.o_proj(o.reshape(B, T, H * dv)), new_state


class SwiGLU(nn.Module):
    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.w_in = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.w_out = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = OptimalMemory(cfg) if cfg.mixer == "optimal" else GatedDeltaMemory(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, state=None):
        h, state = self.mixer(self.norm1(x), state)
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, state


class UmidLM(nn.Module):
    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        if cfg.n_loops > 1:   # har bir halqa uchun o'z "qaysi qadamdaman" vektori
            self.loop_embed = nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        for mod in self.modules():
            for name, value in getattr(mod, "_bias_init", {}).items():   # _init_weights biaslarni nolga tushiradi
                getattr(mod, name).bias.data.copy_(value)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith(("A_log", "dt_bias", "lam_raw")) or "conv" in name or "norm" in name:
                continue
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif name.endswith(("o_proj.weight", "w_out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            else:
                nn.init.normal_(p, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, states=None, return_states=False):
        """idx: [B, T]. states — oldingi chaqiruvdan qaytgan xotira (davom ettirish uchun)."""
        x = self.drop(self.embed(idx))
        new_states = []
        for r in range(self.cfg.n_loops):
            if self.cfg.n_loops > 1:
                x = x + self.loop_embed[r]
            for block in self.blocks:
                x, st = block(x, None if states is None else states[len(new_states)])
                new_states.append(st)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.to(torch.promote_types(logits.dtype, torch.float32)).view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if return_states:
            return logits, loss, new_states
        return logits, loss

    @staticmethod
    def state_bytes(states) -> int:
        """Generatsiya holatining hajmi (baytda) — kontekst uzunligiga bog'liq emas."""
        return sum(t.numel() * t.element_size() for st in states for t in st.values())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.1, eos_id=None, states=None, on_token=None):
        """Avtoregressiv generatsiya. Har bir yangi token uchun xotira va vaqt O(1).

        on_token(token) True qaytarsa generatsiya to'xtaydi. Qaytaradi: (yangi tokenlar
        [B, n], holat). Holat prompt va barcha yangi tokenlarni "eslaydi" — uni keyingi
        chaqiruvga uzatsangiz suhbat davom etadi, KV-cache kabi o'smaydi.
        """
        self.eval()
        logits, _, states = self(idx, states=states, return_states=True)
        history, out = idx, []
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1], history, temperature, top_k, top_p, repetition_penalty)
            out.append(nxt)
            history = torch.cat([history, nxt], dim=1)
            logits, _, states = self(nxt, states=states, return_states=True)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if on_token is not None and on_token(nxt):
                break
        new = torch.cat(out, dim=1) if out else idx.new_zeros(idx.size(0), 0)
        return new, states

    def save(self, path, **extra):
        torch.save({"config": asdict(self.cfg), "model": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(UmidConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        return model, ckpt


def sample_next(logits, history, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1):
    logits = logits.float().clone()
    if repetition_penalty and repetition_penalty != 1.0:
        recent = history[:, -256:]
        picked = logits.gather(1, recent)
        logits.scatter_(1, recent, torch.where(picked > 0, picked / repetition_penalty, picked * repetition_penalty))
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)
