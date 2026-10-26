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
    dropout: float = 0.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def value_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def mlp_hidden(self) -> int:
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
        x = x.transpose(1, 2)
        if cache is None:
            cache = x.new_zeros(B, C, self.kernel_size - 1)
        xx = torch.cat([cache.to(x.dtype), x], dim=-1)
        y = F.conv1d(xx, self.weight.to(x.dtype), groups=C)
        return F.silu(y).transpose(1, 2), xx[..., xx.shape[-1] - (self.kernel_size - 1):]


class GatedDeltaMemory(nn.Module):
    """Attention o'rnini bosuvchi qatlam: har bir head — qat'iy o'lchamli assotsiativ xotira."""

    def __init__(self, cfg: UmidConfig):
        super().__init__()
        self.cfg = cfg
        D, H, dk, dv = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.value_dim
        self.qkv_proj = nn.Linear(D, H * (2 * dk + dv), bias=False)
        self.qkv_conv = ShortConv(H * (2 * dk + dv), cfg.conv_size)
        self.b_proj = nn.Linear(D, H, bias=True)      # β — qancha kuch bilan yozish
        self.a_proj = nn.Linear(D, H, bias=False)     # α — qancha unutish
        self.g_proj = nn.Linear(D, H * dv, bias=False)  # chiqish darvozasi
        self.o_norm = RMSNorm(dv, cfg.norm_eps)
        self.o_proj = nn.Linear(H * dv, D, bias=False)

        # Ko'p masshtabli xotira: har bir head turli tezlikda unutadi (Mamba2 uslubidagi init).
        A = torch.empty(H).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus⁻¹(dt)

    def forward(self, x, state=None):
        cfg = self.cfg
        B, T, _ = x.shape
        H, dk, dv = cfg.n_heads, cfg.head_dim, cfg.value_dim
        st = state or {}
        fdt = torch.promote_types(x.dtype, torch.float32)   # xotira hisob-kitobi kamida fp32 da
        qkv, conv_cache = self.qkv_conv(self.qkv_proj(x), st.get("conv"))
        q, k, v = qkv.split([H * dk, H * dk, H * dv], dim=-1)
        q = F.normalize(q.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
        k = F.normalize(k.reshape(B, T, H, dk).to(fdt), dim=-1).transpose(1, 2)
        v = v.reshape(B, T, H, dv).transpose(1, 2).to(fdt)

        beta = torch.sigmoid(self.b_proj(x).to(fdt))
        if cfg.allow_neg_eigval:
            beta = beta * 2.0
        g = -self.A_log.to(fdt).exp() * F.softplus(self.a_proj(x).to(fdt) + self.dt_bias.to(fdt))
        beta, g = beta.transpose(1, 2), g.transpose(1, 2)    # [B, H, T]

        S0 = st.get("S")
        if T == 1 and S0 is not None:
            o, S = delta_rule_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], beta[:, :, 0], S0)
            o = o[:, :, None]
        else:
            o, S = delta_rule_chunked(q, k, v, g, beta, cfg.chunk_size, S0)

        o = self.o_norm(o.transpose(1, 2).to(x.dtype))           # [B, T, H, dv]
        o = o * F.silu(self.g_proj(x)).view(B, T, H, dv)
        out = self.o_proj(o.reshape(B, T, H * dv))
        return out, {"conv": conv_cache, "S": S}


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
        self.mixer = GatedDeltaMemory(cfg)
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
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith(("A_log", "dt_bias")) or "conv" in name or "norm" in name:
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
        for i, block in enumerate(self.blocks):
            x, st = block(x, None if states is None else states[i])
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
