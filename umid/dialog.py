"""Suhbatdan o'rganish uchun sintetik dialoglar: foydalanuvchi o'zi haqida yoki yangi tushuncha haqida
fakt aytadi, oradan boshqa savollar o'tadi, keyin model aytilgan faktni eslab javob berishi kerak.

Bu modelning og'irliklarida yo'q ma'lumotni (yangi ism, o'ylab topilgan so'z) suhbatning o'zidan
o'rganishini talab qiladi. Umid'da bu xotira matritsasida sodir bo'ladi: har bir token delta qoidasi
bilan xotirani yangilaydi (S ← S + β k (v − Sᵀk)ᵀ) — bu kalit-qiymat bog'lanishini o'rganishning bir
gradient qadamiga teng.

O'qitish ("train") va baholash ("test") uchun qiymatlar (ismlar, shaharlar, ranglar, ta'riflar, o'ylab
topilgan so'zlar) KESISHMAYDI: baholashda model ismni ham, so'zni ham o'qitishda hech qachon ko'rmagan.
"""

from __future__ import annotations

import random

NAMES = {
    "train": ["Aziz", "Bobur", "Dilnoza", "Gulnora", "Jasur", "Kamola", "Laylo", "Madina", "Nodir", "Otabek",
              "Sardor", "Shahzoda", "Temur", "Umida", "Zarina", "Farrux", "Islom", "Javohir", "Malika", "Nilufar",
              "Rustam", "Sevara", "Feruza", "Bekzod", "Dilshod", "Elyor", "Hasan", "Komil", "Lola", "Murod"],
    "test": ["Anvar", "Barno", "Diyor", "Hulkar", "Ozoda", "Sanjar", "Shoxrux", "Yulduz", "Zuhra", "Mirjalol",
             "Nargiza", "Ravshan", "Saida", "Ulugbek", "Xurshid"],
}
CITIES = {
    "train": ["Toshkent", "Samarqand", "Buxoro", "Xiva", "Namangan", "Andijon", "Qarshi", "Termiz", "Nukus",
              "Shahrisabz"],
    "test": ["Jizzax", "Guliston", "Navoiy", "Urganch", "Marg'ilon", "Chirchiq", "Angren", "Denov"],
}
COLORS = {"train": ["qizil", "ko'k", "yashil", "sariq", "oq", "qora"],
          "test": ["pushti", "binafsha", "jigarrang", "kulrang"]}
DEFINITIONS = {
    "train": ["Marsdagi qadimiy shahar", "ko'k rangli shirin meva", "tez yuguradigan kichik hayvon",
              "bolalar o'ynaydigan o'yin", "tog'dagi sovuq ko'l", "qadimiy musiqa asbobi", "yangi turdagi robot",
              "dengiz tubidagi o'simlik", "kuzda pishadigan nok navi", "uchar mashina", "qishloqdagi bayram",
              "issiq ichimlik"],
    "test": ["Oydagi kichik qishloq", "qizil rangli nordon meva", "suvda yashaydigan qush", "eski kitob nomi",
             "yozda gullaydigan daraxt", "temirdan yasalgan ko'prik"],
}
# O'ylab topilgan so'zlar: train va test turli bo'g'inlardan — testdagi so'zlar o'qitishda hech qachon uchramaydi
SYLLABLES = {"train": ["ka", "ri", "zo", "mu", "le", "ta", "vi", "no", "sa", "du", "pe", "lo"],
             "test": ["xe", "gu", "fi", "ja", "bo", "qi", "yu", "he"]}

AGE_RANGE = {"train": (6, 60), "test": (61, 95)}
ROOM_RANGE = {"train": (100, 499), "test": (500, 999)}

ACKS = ["Yaxshi, eslab qoldim.", "Tushundim.", "Eslab qoldim.", "Yaxshi."]
UNKNOWN = "Siz buni menga hali aytmagansiz."

# (yozuv nomi, aytish shablonlari, so'rash shablonlari, javob shablonlari)
FACTS = [
    ("ism", ["Mening ismim {v}.", "Ismim {v}."], ["Mening ismim nima?", "Ismim nima edi?"],
     ["Sizning ismingiz {v}.", "Ismingiz {v}."]),
    ("shahar", ["Men {v} shahrida yashayman."], ["Men qayerda yashayman?", "Men qaysi shaharda yashayman?"],
     ["Siz {v} shahrida yashaysiz."]),
    ("yosh", ["Men {v} yoshdaman."], ["Men necha yoshdaman?"], ["Siz {v} yoshdasiz."]),
    ("do'st", ["Do'stimning ismi {v}."], ["Do'stimning ismi nima?"], ["Do'stingizning ismi {v}."]),
    ("rang", ["Mening sevimli rangim {v}."], ["Sevimli rangim qaysi?"], ["Sizning sevimli rangingiz {v}."]),
    ("mushuk", ["Mushugimning ismi {v}."], ["Mushugimning ismi nima?"], ["Mushugingizning ismi {v}."]),
    ("xona", ["Xonam raqami {v}."], ["Xonam raqami nechchi?"], ["Xonangiz raqami {v}."]),
    ("so'z", ["Bilib qo'y: {w} — bu {v}."], ["{w} nima?"], ["{w} — bu {v}."]),
]


def new_word(rng, split):
    syl = SYLLABLES[split]
    return "".join(rng.choice(syl) for _ in range(rng.randint(2, 3))).capitalize()


def fact_value(kind, rng, split):
    if kind in ("ism", "do'st"):
        return rng.choice(NAMES[split])
    if kind == "mushuk":
        return new_word(rng, split)
    if kind == "shahar":
        return rng.choice(CITIES[split])
    if kind == "yosh":
        return str(rng.randint(*AGE_RANGE[split]))
    if kind == "rang":
        return rng.choice(COLORS[split])
    if kind == "xona":
        return str(rng.randint(*ROOM_RANGE[split]))
    return rng.choice(DEFINITIONS[split])


def make_dialog(rng, split, fillers, n_facts=None, n_fill=None, unknown_p=0.15):
    """Bitta dialog: [(savol, javob), ...] va tekshiruv ro'yxati [(navbat raqami, kutilgan qiymat)].
    fillers — oraliq savol-javoblar (bilimga oid, masalan chat-data/savol-javob dan)."""
    kinds = rng.sample(FACTS, n_facts or rng.randint(1, 3))
    told, turns = [], []
    for kind, say, ask, ans in kinds:
        v = fact_value(kind, rng, split)
        w = new_word(rng, split) if kind == "so'z" else None
        turns.append((rng.choice(say).format(v=v, w=w), rng.choice(ACKS)))
        told.append((ask, ans, v, w))
    turns += rng.sample(fillers, min(n_fill if n_fill is not None else rng.randint(0, 3), len(fillers)))
    rng.shuffle(told)
    checks = []
    for ask, ans, v, w in told:
        checks.append((len(turns), v))
        turns.append((rng.choice(ask).format(w=w), rng.choice(ans).format(v=v, w=w)))
    if rng.random() < unknown_p:   # aytilmagan narsani so'rasa — o'ylab topmaydi
        untold = [f for f in FACTS if f[0] not in {k[0] for k in kinds} and f[0] != "so'z"]
        if untold:
            checks.append((len(turns), "aytmagansiz"))
            turns.append((rng.choice(rng.choice(untold)[2]), UNKNOWN))
    return turns, checks
