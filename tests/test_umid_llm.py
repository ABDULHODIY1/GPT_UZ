"""umid_llm.py (hammasi bitta faylda): model aniqligi va repo'dagi Tanlov algoritmi bilan mosligi."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("umid_llm", Path(__file__).parent.parent / "umid_llm.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_single_file_model_parallel_equals_step_by_step():
    torch.manual_seed(0)
    m = u.UmidLM(u.ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, select_k=5, select_chunk=7)).eval()
    for blk in m.blocks:
        torch.nn.init.normal_(blk.mem.s_proj.weight, std=1.0)
    idx = torch.randint(0, 63, (1, 40))
    idx[0, 17] = 63                                   # EOT
    with torch.no_grad():
        full = m(idx)[0]
        steps, st = [], None
        for t in range(40):
            lg, _, st = m(idx[:, t:t + 1], states=st)
            steps.append(lg)
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-4, atol=1e-5)


def test_single_file_algorithm_matches_repo():
    from umid.tanlov import selective_read
    torch.manual_seed(1)
    q, k = (F.normalize(torch.randn(2, 3, 30, 8), dim=-1) for _ in range(2))
    v, s = torch.randn(2, 3, 30, 8), torch.randn(2, 3, 30)
    seg = torch.zeros(2, 30, dtype=torch.long)
    seg[1, 12:] = 1
    args = (q, k, v, s, torch.arange(30).expand(2, 30), seg, 5, torch.tensor([4., 8., 2.]),
            torch.tensor([-1., -2., 0.]), torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(u.selective_read(*args)[0], selective_read(*args)[0])


def test_single_file_cleaning_and_tokenizer():
    text = "O‘zbekiston — <ref>x</ref> [[Osiyo|Markaziy Osiyodagi]] davlat, poytaxti Toshkent shahri hisoblanadi.\n== Manbalar ==\nmanba"
    out = u.clean_doc(text)
    assert out == "O'zbekiston — Markaziy Osiyodagi davlat, poytaxti Toshkent shahri hisoblanadi."
    tok = u.Tokenizer.train(out * 20, 300)
    assert tok.decode(tok.encode(out)) == out
