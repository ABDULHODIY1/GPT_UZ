import pytest
import torch

from umid import UmidConfig, UmidLM


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = UmidConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=2, head_dim=32, chunk_size=16, **kw)
    return UmidLM(cfg).double().eval()


# Arxitektura variantlari: kanal bo'yicha unutish, ko'p yozish, fikrlash halqasi
VARIANTS = [dict(), dict(decay="head", n_reads=1), dict(decay="channel", n_reads=1), dict(n_writes=2, n_reads=1), dict(n_writes=3, n_reads=1), dict(n_loops=2),
            dict(mixer="optimal"), dict(mixer="optimal", frame=True), dict(n_reads=2), dict(n_reads=3, decay="channel"),
            dict(decouple_erase=True), dict(decouple_erase=True, n_writes=2, n_reads=1), dict(decouple_erase=True, n_reads=2),
            dict(persistent_heads=1, reset_on_eot=True), dict(persistent_heads=1, decay="channel")]


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


def test_load_model_dispatches_on_arch(tmp_path):
    from umid import load_model
    from umid.baseline import GPT, GPTConfig
    gpt = GPT(GPTConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2))
    gpt.save(tmp_path / "gpt.pt", tokenizer={})
    umid = UmidLM(UmidConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, head_dim=16))
    umid.save(tmp_path / "umid.pt")
    g2, ck = load_model(tmp_path / "gpt.pt")
    u2, _ = load_model(tmp_path / "umid.pt")
    assert isinstance(g2, GPT) and isinstance(u2, UmidLM) and ck["tokenizer"] == {}
    x = torch.randint(0, 64, (1, 8))
    assert torch.allclose(g2(x)[0], gpt(x)[0]) and torch.allclose(u2(x)[0], umid.eval()(x)[0])


@pytest.mark.parametrize("mixer", ["delta", "optimal"])
def test_reset_on_eot_isolates_documents(mixer):
    torch.manual_seed(0)
    cfg = UmidConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, head_dim=16, chunk_size=8, mixer=mixer,
                     reset_on_eot=True)
    model = UmidLM(cfg).eval()
    eot, tail, doc2 = 63, torch.randint(0, 60, (1, 3)), torch.randint(0, 60, (1, 13))
    seqs = [torch.cat([torch.randint(0, 60, (1, n)), tail, torch.tensor([[eot]]), doc2], 1) for n in (5, 21)]
    outs = [model(s)[0][:, -doc2.size(1):] for s in seqs]
    # Ikkinchi hujjat birinchisining mazmuniga bog'liq emas (faqat oxirgi conv_size−1 token konvolyutsiyadan o'tadi)
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    # Tokenma-token (generatsiya rejimi) ham xuddi shunday
    full = model(seqs[1])[0]
    _, _, st = model(seqs[1][:, :1], return_states=True)
    steps = [model(seqs[1][:, :1])[0]]
    for t in range(1, seqs[1].size(1)):
        lg, _, st = model(seqs[1][:, t:t + 1], states=st, return_states=True)
        steps.append(lg)
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-4)


@pytest.mark.parametrize("doc_mask", [False, True])
def test_gpt_cache_matches_full_forward(doc_mask):
    from umid.baseline import GPT, GPTConfig
    torch.manual_seed(0)
    gpt = GPT(GPTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, doc_mask=doc_mask)).eval()
    seq = torch.cat([torch.randint(0, 60, (1, 7)), torch.tensor([[63]]), torch.randint(0, 60, (1, 9))], 1)
    full = gpt(seq)[0]
    # bo'laklab: 5 token, keyin 6 token (ichida EOT), keyin tokenma-token
    lg1, _, st = gpt(seq[:, :5], return_states=True)
    lg2, _, st = gpt(seq[:, 5:11], states=st, return_states=True)
    steps = [lg1, lg2]
    for t in range(11, seq.size(1)):
        lg, _, st = gpt(seq[:, t:t + 1], states=st, return_states=True)
        steps.append(lg)
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
    if doc_mask:   # EOT dan keyingi qism birinchi hujjatga bog'liq emas
        other = seq.clone()
        other[:, :7] = torch.randint(0, 60, (1, 7))
        assert torch.allclose(gpt(other)[0][:, 7:], full[:, 7:], atol=1e-5)
        assert st[0][0].shape[2] == seq.size(1) - 7   # kesh EOT dan boshlab saqlanadi
