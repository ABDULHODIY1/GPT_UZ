import torch

from umid.data import StreamBatcher, split_tokens


def test_split_is_disjoint_and_spread_over_corpus():
    tokens = torch.arange(100_000)
    train, val = split_tokens(tokens, 0.1, block=4096)
    assert len(train) + len(val) == len(tokens)
    assert not set(train.tolist()) & set(val.tolist())
    assert val.min() < 30_000 and val.max() > 60_000   # korpusning turli joylaridan


def test_split_small_corpus_falls_back_to_tail():
    train, val = split_tokens(torch.arange(1000), 0.1)
    assert torch.equal(val, torch.arange(900, 1000))


def test_stream_batcher_is_contiguous_and_resets():
    tokens = torch.arange(1001)
    sb = StreamBatcher(tokens, batch_size=2, seq_len=100)
    x1, y1, r1 = sb.next()
    x2, _, r2 = sb.next()
    assert r1 and not r2
    assert torch.equal(y1, x1 + 1) and torch.equal(x2[:, 0], x1[:, -1] + 1)
    for _ in range(3):
        sb.next()
    assert sb.next()[2]   # oqim oxiri → qaytadan, holatni tozalash kerak


QA = "Savol: Salom\nJavob: Assalomu alaykum!\n\nSavol: Isming nima?\nJavob: Mening ismim Umid.\n\nSavol: Rahmat\nJavob: Arzimaydi!\n"


def test_repeated_source_is_reshuffled_each_time(tmp_path):
    from umid.data import read_corpus
    from umid.tokenizer import EOT
    (tmp_path / "qa.txt").write_text(QA, encoding="utf-8")
    blocks = read_corpus(f"{tmp_path / 'qa.txt'}*4").split(EOT)
    assert len(blocks) == 12 and all(b.startswith("Savol:") for b in blocks)
    orders = {tuple(blocks[i:i + 3]) for i in range(0, 12, 3)}
    assert len(orders) > 1   # har takrorda boshqa tartib


def test_qa_batch_masks_everything_but_answers(tmp_path):
    from umid.data import encode_qa, qa_batch, read_qa_pairs
    from umid.tokenizer import BPETokenizer
    (tmp_path / "qa.txt").write_text(QA, encoding="utf-8")
    pairs = read_qa_pairs(tmp_path)
    assert pairs[1] == ("Isming nima?", "Mening ismim Umid.")
    tok = BPETokenizer.train(QA * 3, 300)
    x, y = qa_batch(encode_qa(tok, pairs), 3, 32, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (3, 32)
    targets = [t for t in y[0].tolist() if t != -100]
    text = tok.decode(targets)
    assert "Savol" not in text and "Javob" not in text   # savol va "Javob:" uchun loss yo'q
    assert tok.eot_id in targets                          # javob oxirida to'xtashni o'rganadi
    assert torch.equal(x[:, 1:][y[:, :-1] != -100], y[:, :-1][y[:, :-1] != -100])
