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


def test_dialog_test_values_never_seen_in_training():
    import random
    from umid.dialog import CITIES, COLORS, DEF_WORDS, NAMES, SYLLABLES, make_dialog
    for pools in (NAMES, CITIES, COLORS, SYLLABLES):
        assert not set(pools["train"]) & set(pools["test"])
    for a, b in zip(DEF_WORDS["train"], DEF_WORDS["test"]):
        assert not set(a) & set(b)
    fillers = [("Salom", "Assalomu alaykum!"), ("Rahmat", "Arzimaydi!"), ("Xayr", "Xayr!")]
    train_text = " ".join(q + " " + a for i in range(300)
                          for q, a in make_dialog(random.Random(i), "train", fillers)[0])
    for i in range(50):
        turns, checks = make_dialog(random.Random(10_000 + i), "test", fillers, n_fill=1)
        assert len(checks) >= 1 and all(0 <= t < len(turns) for t, _ in checks)
        for t, v in checks:
            assert v.lower() in turns[t][1].lower()          # to'g'ri javobda qiymat bor
            if v != "aytmagansiz" and not v.isdigit():
                assert v not in train_text                   # o'qitishda bu qiymat yo'q


def test_encode_dialog_masks_only_answers():
    from umid.data import encode_dialog
    from umid.tokenizer import BPETokenizer
    tok = BPETokenizer.train(QA * 3, 300)
    ids, mask = encode_dialog(tok, [("Salom", "Assalomu alaykum!"), ("Rahmat", "Arzimaydi!")])
    assert ids[-1] == tok.eot_id and mask[-1]
    assert tok.decode(ids[mask].tolist()).replace("<|endoftext|>", "") == " Assalomu alaykum!\n Arzimaydi!\n"


WIKI = """'''Samarqand''' — O‘zbekistondagi eng qadimiy shaharlardan biri. U [[Zarafshon|Zarafshon vodiysida]] joylashgan va <ref>manba</ref> boy tarixga ega.
{{Infobox shahar | aholi = 551 700}}
== Tarixi ==
Shahar Buyuk Ipak yo‘li ustida joylashgan bo‘lib, ko‘p asrlar davomida savdo markazi bo‘lgan. https://example.com/samarqand
| 1991 | 2000 | 2010 | 2020 |
Qisqa.
== Manbalar ==
Bu qator manbalar bo'limida, olib tashlanishi kerak va u yetarlicha uzun matn.
"""


def test_clean_doc_removes_markup_and_reference_sections():
    from umid.clean import clean_doc
    out = clean_doc(WIKI)
    assert "O'zbekistondagi" in out and "yo'li" in out             # tutuq belgilari bir xil
    assert "Zarafshon vodiysida" in out and "[[" not in out          # havola matni qoladi
    assert out.startswith("Samarqand — ")
    for junk in ("<ref>", "manba boy", "{{", "Infobox", "== Tarixi", "https://", "| 1991", "Qisqa.", "manbalar bo'limida"):
        assert junk not in out


def test_deduper_drops_repeated_documents_and_paragraphs():
    from umid.clean import Deduper
    d = Deduper()
    assert d("Birinchi abzas matni.\nUmumiy abzas.") == "Birinchi abzas matni.\nUmumiy abzas."
    assert d("Birinchi abzas matni.\nUmumiy abzas.") == ""           # hujjat takrori
    assert d("Ikkinchi hujjat.\nUmumiy abzas.") == "Ikkinchi hujjat."   # abzas takrori
