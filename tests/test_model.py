import pytest
import torch

from umid import UmidConfig, UmidLM


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = UmidConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=2, head_dim=32, chunk_size=16, **kw)
    return UmidLM(cfg).double().eval()


# Arxitektura variantlari: kanal bo'yicha unutish, ko'p yozish, fikrlash halqasi
VARIANTS = [dict(), dict(decay="channel"), dict(n_writes=2), dict(n_writes=3, decay="channel"), dict(n_loops=2),
            dict(mixer="optimal")]


def test_forward_shapes_and_loss():
    m = tiny_model()
    idx = torch.randint(0, 97, (3, 50))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (3, 50, 97)
    # tied embedding tufayli boshlang'ich loss ln(V) dan biroz yuqori bo'ladi
    assert torch.isfinite(loss) and loss.item() < 2 * torch.log(torch.tensor(97.0)).item()


@pytest.mark.parametrize("variant", VARIANTS)
def test_causality(variant):
    """Kelajakdagi tokenlar o'tmishdagi chiqishlarga ta'sir qilmasligi kerak."""
    m = tiny_model(**variant)
    idx = torch.randint(0, 97, (1, 40))
    a, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, 25:] = torch.randint(0, 97, (15,))
    b, _ = m(idx2)
    torch.testing.assert_close(a[:, :25], b[:, :25])
    assert not torch.allclose(a[:, 25:], b[:, 25:])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("split", [1, 17, 33])
def test_incremental_decoding_matches_full_forward(split, variant):
    """Prefill + token-ma-token generatsiya (O(1) holat) = to'liq forward."""
    m = tiny_model(**variant)
    idx = torch.randint(0, 97, (2, 45))
    full, _ = m(idx)
    logits, _, states = m(idx[:, :split], return_states=True)
    parts = [logits]
    for t in range(split, 45):
        logits, _, states = m(idx[:, t:t + 1], states=states, return_states=True)
        parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, 1), full, rtol=1e-9, atol=1e-9)


def test_state_size_does_not_grow_with_context():
    m = tiny_model()
    sizes = []
    for T in (10, 100, 1000):
        _, _, states = m(torch.randint(0, 97, (1, T)), return_states=True)
        sizes.append(UmidLM.state_bytes(states))
    assert sizes[0] == sizes[1] == sizes[2]


def test_runs_beyond_any_training_length():
    """Pozitsiya embedding yo'q — istalgan uzunlikda ishlaydi."""
    m = tiny_model().float()
    logits, _ = m(torch.randint(0, 97, (1, 5000)))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_backward_reaches_all_parameters(variant):
    m = tiny_model(**variant).train()
    idx = torch.randint(0, 97, (2, 30))
    _, loss = m(idx, targets=idx)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing


def test_generate_and_continue_state():
    m = tiny_model().float()
    prompt = torch.randint(0, 97, (1, 8))
    out, states = m.generate(prompt, 12, temperature=0.0, repetition_penalty=1.0)
    assert out.shape == (1, 12)
    # greedy generatsiya to'liq forward'dagi argmax bilan mos kelishi kerak
    logits, _ = m(torch.cat([prompt, out], 1)[:, :-1])
    assert torch.equal(logits[:, prompt.size(1) - 1:].argmax(-1), out)
    # holat prompt + javobni eslaydi: davom ettirish = hammasini qaytadan o'qish
    nxt = torch.randint(0, 97, (1, 3))
    cont, _ = m.generate(nxt, 5, temperature=0.0, repetition_penalty=1.0, states=states)
    ref, _ = m.generate(torch.cat([prompt, out, nxt], 1), 5, temperature=0.0, repetition_penalty=1.0)
    assert torch.equal(cont, ref)

    stop_after = []
    out3, _ = m.generate(prompt, 50, temperature=0.0, on_token=lambda t: stop_after.append(t) or len(stop_after) == 4)
    assert out3.shape == (1, 4)


def test_save_load_roundtrip(tmp_path):
    m = tiny_model(allow_neg_eigval=False)
    p = tmp_path / "m.pt"
    m.save(p, note="test")
    m2, ckpt = UmidLM.load(p)
    m2 = m2.double().eval()
    idx = torch.randint(0, 97, (1, 20))
    torch.testing.assert_close(m(idx)[0], m2(idx)[0])
    assert ckpt["note"] == "test" and m2.cfg.allow_neg_eigval is False


def test_overfits_a_tiny_batch():
    """Model o'rgana olishini tekshiramiz: kichik batch'ni yodlab olishi kerak."""
    torch.manual_seed(0)
    cfg = UmidConfig(vocab_size=50, d_model=64, n_layers=2, n_heads=2, head_dim=32, chunk_size=16)
    m = UmidLM(cfg)
    idx = torch.randint(0, 50, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(idx[:, :-1], targets=idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()
