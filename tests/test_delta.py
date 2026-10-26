import pytest
import torch
import torch.nn.functional as F

from umid.delta import delta_rule_chunked, delta_rule_recurrent


def make_inputs(B=2, H=3, T=150, Dk=16, Dv=24, dtype=torch.float64, seed=0, neg=True, channel=False, decay=0.5):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, Dk, generator=g, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(B, H, T, Dk, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, Dv, generator=g, dtype=dtype)
    logα = -torch.rand(B, H, T, *((Dk,) if channel else ()), generator=g, dtype=dtype) * decay
    beta = torch.rand(B, H, T, generator=g, dtype=dtype) * (2.0 if neg else 1.0)
    return q, k, v, logα, beta


@pytest.mark.parametrize("channel", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 16, 64])
@pytest.mark.parametrize("T", [1, 63, 64, 150])
def test_chunked_matches_recurrent(chunk, T, channel):
    q, k, v, g, beta = make_inputs(T=T, channel=channel)
    S0 = torch.randn(2, 3, 16, 24, dtype=torch.float64)
    o_ref, s_ref = delta_rule_recurrent(q, k, v, g, beta, S0)
    o, s = delta_rule_chunked(q, k, v, g, beta, chunk, S0)
    torch.testing.assert_close(o, o_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(s, s_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("channel", [False, True])
def test_gradients_match_recurrent(channel):
    inputs = [x.requires_grad_() for x in make_inputs(T=40, channel=channel)]
    o, s = delta_rule_recurrent(*inputs)
    (o.sum() + s.square().sum()).backward()
    ref = [x.grad.clone() for x in inputs]
    for x in inputs:
        x.grad = None
    o, s = delta_rule_chunked(*inputs, chunk_size=16)
    (o.sum() + s.square().sum()).backward()
    for a, b in zip(ref, inputs):
        torch.testing.assert_close(b.grad, a, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("channel", [False, True])
def test_streaming_equals_full(channel):
    """Uzun matnni bo'laklab (holatni uzatib) ishlatish = birdaniga ishlatish."""
    q, k, v, g, beta = make_inputs(T=200, channel=channel)
    o_full, s_full = delta_rule_chunked(q, k, v, g, beta, 32)
    S, outs = None, []
    for a, b in [(0, 70), (70, 71), (71, 200)]:
        o, S = delta_rule_chunked(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], g[:, :, a:b], beta[:, :, a:b], 32, S)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 2), o_full)
    torch.testing.assert_close(S, s_full)


def test_state_is_bounded_on_long_sequences():
    """β ∈ (0, 2) va unutish bilan holat 20k token davomida portlamaydi."""
    q, k, v, g, beta = make_inputs(B=1, H=2, T=20_000, dtype=torch.float32, seed=1)
    o, S = delta_rule_chunked(q, k, v, g, beta, 64)
    assert torch.isfinite(o).all() and torch.isfinite(S).all()
    assert o.abs().max() < 50 and S.abs().max() < 50


@pytest.mark.parametrize("channel,decay", [(False, 0.5), (True, 0.5), (True, 5.0)])
def test_fp32_close_to_fp64(channel, decay):
    """Kanal rejimida kuchli unutishda ham (g = −5) fp32 toshib ketmaydi."""
    q, k, v, g, beta = make_inputs(T=300, channel=channel, decay=decay)
    o64, _ = delta_rule_chunked(q, k, v, g, beta, 64)
    o32, _ = delta_rule_chunked(*(x.float() for x in (q, k, v, g, beta)), chunk_size=64)
    torch.testing.assert_close(o32.double(), o64, rtol=1e-4, atol=1e-4)


def test_delta_rule_overwrites_instead_of_accumulating():
    """Delta qoidasi: bir xil kalitga yangi qiymat yozilsa, eskisi almashtiriladi
    (oddiy linear attention'da esa ular qo'shilib, xotira "ifloslanadi")."""
    Dk = 8
    key = F.normalize(torch.randn(Dk, dtype=torch.float64), dim=0)
    k = key.expand(1, 1, 2, Dk).clone()
    v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    q = key.expand(1, 1, 2, Dk).clone()
    g = torch.zeros(1, 1, 2, dtype=torch.float64)
    beta = torch.ones(1, 1, 2, dtype=torch.float64)
    o, _ = delta_rule_recurrent(q, k, v, g, beta)
    torch.testing.assert_close(o[0, 0, 1], torch.tensor([0.0, 1.0], dtype=torch.float64))
