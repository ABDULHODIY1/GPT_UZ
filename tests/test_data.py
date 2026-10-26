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
