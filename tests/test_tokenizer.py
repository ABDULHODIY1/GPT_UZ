from umid.tokenizer import EOT, BPETokenizer

TEXT = (
    "Assalomu alaykum! Sizga qanday yordam bera olaman? O‘g‘il bola bo'lsa ham, qiz bola bo'lsa ham "
    "farzand – baxt. Kitob o‘qish foydali. 2024-yilda 12345 ta kitob o‘qildi. "
) * 20


def test_roundtrip_and_compression():
    tok = BPETokenizer.train(TEXT, 400)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8")) / 3
    assert max(ids) < tok.vocab_size


def test_unseen_text_emoji_and_specials():
    tok = BPETokenizer.train(TEXT, 300)
    s = f"Yangi so'z: Привет 😊 漢字{EOT}keyingi hujjat"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert ids.count(tok.eot_id) == 1
    # allow_special=False bo'lsa maxsus token oddiy matn sifatida kodlanadi
    plain = tok.encode(EOT, allow_special=False)
    assert tok.eot_id not in plain and tok.decode(plain) == EOT


def test_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 350)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    assert tok2.vocab_size == tok.vocab_size
