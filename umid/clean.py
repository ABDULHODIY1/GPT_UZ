"""Internetdan olingan o'zbekcha matnni tozalash (prepare_data.py uchun).

Har bir hujjat uchun:
  * Unicode NFC, tutuq belgilari bitta ko'rinishga: o‘ oʻ o’ o` → o' (loyiha va foydalanuvchi shunday yozadi);
  * URL, e-pochta, HTML teglari/entity'lari, vikimatn qoldiqlari ({{…}}, [[…]], == Sarlavha ==) olib tashlanadi;
  * "Manbalar", "Havolalar", "Adabiyotlar", "Izohlar", "Yana qarang" bo'limlari kesiladi;
  * asosan raqam/belgidan iborat qatorlar (jadval, ro'yxat qoldiqlari) va juda qisqa qatorlar tashlanadi;
  * kirillcha ulushi katta hujjatlar tashlanadi (kichik model bitta yozuvni yaxshiroq o'rganadi).
Takrorlar: `Deduper` — bir xil hujjat va bir xil abzaslar faqat bir marta olinadi.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_APOS = re.compile(r"[‘’ʻʼ`´′]")
_URL = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\w+")
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>[\s\S]*?</ref>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]{1,200}>")
_BOLD = re.compile(r"'{2,}")                  # vikimatndagi qiya va qalin yozuv belgilari
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}|\{\|[\s\S]*?\|\}")
_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
_SPACE = re.compile(r"[ \t ​]+")
_LETTER = re.compile(r"[^\W\d_]")
_CYR = re.compile(r"[Ѐ-ӿ]")
_STOP_SECTIONS = ("manbalar", "havolalar", "adabiyotlar", "izohlar", "yana qarang", "tashqi havolalar",
                  "manba", "references", "external links", "see also")


def normalize_apostrophes(text):
    return _APOS.sub("'", text)


def letter_ratio(s):
    return len(_LETTER.findall(s)) / max(len(s), 1)


def cyrillic_ratio(s):
    letters = _LETTER.findall(s)
    return sum(bool(_CYR.match(c)) for c in letters) / max(len(letters), 1)


def clean_doc(text, min_line=30, min_letters=0.6):
    """Bitta hujjatni tozalaydi. Bo'sh satr qaytarsa — hujjat tashlanadi."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = normalize_apostrophes(text)
    text = _REF.sub(" ", text)
    text = _TEMPLATE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _BOLD.sub("", text)
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        m = _HEADING.match(line)
        head = (m.group(1) if m else line).strip().lower().rstrip(":")
        if head in _STOP_SECTIONS:           # manbalar/havolalar bo'limidan keyingisi kerak emas
            break
        if m:
            continue                          # sarlavhalarning o'zi o'qitish uchun foydasiz
        line = _SPACE.sub(" ", line).strip()
        if len(line) >= min_line and letter_ratio(line) >= min_letters:
            out.append(line)
    return "\n".join(out)


class Deduper:
    """Bir xil hujjat va bir xil abzaslarni olib tashlaydi (xeshlar xotirada, matnning o'zi emas)."""

    def __init__(self):
        self.docs, self.paras = set(), set()

    @staticmethod
    def _key(s):
        return hashlib.blake2b(re.sub(r"\W+", "", s.lower()).encode(), digest_size=8).digest()

    def __call__(self, text):
        k = self._key(text[:2000])
        if k in self.docs:
            return ""
        self.docs.add(k)
        kept = []
        for para in text.split("\n"):
            pk = self._key(para)
            if pk not in self.paras:
                self.paras.add(pk)
                kept.append(para)
        return "\n".join(kept)
