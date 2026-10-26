import pytest
import torch
import torch.nn.functional as F

from umid.delta import delta_rule_recurrent
from umid.optimal import optimal_memory_chunked, optimal_memory_recurrent


def inputs(B=2, H=3, T=70, Dk=8, Dv=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.float64
    q = F.normalize(torch.randn(B, H, T, Dk, generator=g, dtype=d), dim=-1)
    k = F.normalize(torch.randn(B, H, T, Dk, generator=g, dtype=d), dim=-1)
    v = torch.randn(B, H, T, Dv, generator=g, dtype=d)
    logα = -torch.rand(B, H, T, generator=g, dtype=d) * 0.3
    beta = torch.rand(B, H, T, generator=g, dtype=d)
    lam = torch.rand(H, Dk, generator=g, dtype=d) + 0.1
    return q, k, v, logα, beta, lam


@pytest.mark.parametrize("chunk", [1, 16, 32, 128])
def test_chunked_matches_recurrent(chunk):
    args = inputs()
    o_ref, (h_ref, g_ref) = optimal_memory_recurrent(*args)
    o, (h, g) = optimal_memory_chunked(*args, chunk_size=chunk)
    torch.testing.assert_close(o, o_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(h, h_ref)
    torch.testing.assert_close(g, g_ref)


def test_gradients_match_recurrent():
    args = [x.requires_grad_() for x in inputs(T=30)]
    o, (h, g) = optimal_memory_recurrent(*args)
    (o.sum() + g.square().sum()).backward()
    ref = [x.grad.clone() for x in args]
    for x in args:
        x.grad = None
    o, (h, g) = optimal_memory_chunked(*args, chunk_size=8)
    (o.sum() + g.square().sum()).backward()
    for a, b in zip(ref, args):
        torch.testing.assert_close(b.grad, a, rtol=1e-8, atol=1e-8)


def test_exact_recall_with_correlated_keys():
    """Kuchli korrelyatsiyalangan kalitlar: optimal xotira aniq eslaydi, delta rule esa yo'q."""
    torch.manual_seed(0)
    Dk, Dv = 8, 12
    d = torch.float64
    K = F.normalize(torch.randn(1, 1, Dk, Dk, dtype=d) + 2.0, dim=-1)        # o'rtacha cos ≈ 0.8
    V = torch.randn(1, 1, Dk, Dv, dtype=d)
    zeros, ones = torch.zeros(1, 1, Dk, dtype=d), torch.ones(1, 1, Dk, dtype=d)
    _, (Hm, G) = optimal_memory_recurrent(K, K, V, zeros, ones, torch.full((1, Dk), 1e-8, dtype=d))
    read_opt = (G[0, 0].T @ torch.linalg.solve(Hm[0, 0] + 1e-8 * torch.eye(Dk, dtype=d), K[0, 0].T)).T
    _, S = delta_rule_recurrent(K, K, V, zeros, ones)
    read_delta = K[0, 0] @ S[0, 0]
    assert (read_opt - V[0, 0]).abs().max() < 1e-3
    assert (read_delta - V[0, 0]).abs().max() > 0.5
