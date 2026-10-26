"""Taqqoslash uchun klassik Transformer (Llama uslubi: RoPE + softmax attention + SwiGLU).

Faqat benchmark'lar uchun: Umid bilan bir xil sharoitda (bir xil tokenizer, o'lcham,
o'qitish rejimi) solishtirish maqsadida.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 8 / 3
    rope_base: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-6

    @property
    def mlp_hidden(self) -> int:
        return 64 * math.ceil(self.d_model * self.mlp_ratio / 64)


def rope(x, pos0):
    """x: [B, H, T, D]; pozitsiyalar pos0 .. pos0+T−1."""
    D = x.shape[-1]
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    t = torch.arange(pos0, pos0 + x.shape[-2], device=x.device, dtype=torch.float32)
    ang = torch.outer(t, inv)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        pos0 = 0 if cache is None else cache[0].shape[2]
        q, k = rope(q, pos0), rope(k, pos0)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, D)), (k, v)


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.w_in = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.w_out = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        h, cache = self.attn(self.norm1(x), cache)
        x = x + self.drop(h)
        a, b = self.w_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.drop(self.w_out(F.silu(a) * b)), cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for name, p in self.named_parameters():
            if "norm" in name:
                continue
            std = 0.02 / math.sqrt(2 * cfg.n_layers) if name.endswith(("o_proj.weight", "w_out.weight")) else 0.02
            nn.init.normal_(p, std=std)
        self.lm_head.weight = self.embed.weight

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, states=None, return_states=False):
        x = self.embed(idx)
        caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if states is None else states[i])
            caches.append(c)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if return_states:
            return logits, loss, caches
        return logits, loss

    @staticmethod
    def state_bytes(caches) -> int:
        return sum(t.numel() * t.element_size() for kv in caches for t in kv)
