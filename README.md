# GPT_UZ — Umid: Transformer attention'siz o'zbekcha til modeli

**Umid** — PyTorch'da noldan yozilgan yangi turdagi til modeli (LLM). Unda Transformer'ning
softmax attention'i yo'q. Uning o'rniga har bir head'da qat'iy o'lchamli **assotsiativ xotira**
bor: **Gated Delta Memory**. Bu xotira har bir token bilan "xato bo'yicha tuzatish" (delta rule)
qoidasi asosida yangilanadi.

```
pip install -r requirements.txt
python train.py --reset-on-eot --persistent-heads 3 --out checkpoints/umid_base   # 1) til (pretraining)
python sft.py --init checkpoints/umid_base/model.pt --out checkpoints/umid        # 2) savol-javob (SFT)
python chat.py --ckpt checkpoints/umid/model.pt                                   # suhbat
```

## Transformer attention'ning qaysi kamchiliklari hal qilindi

| Transformer muammosi | Umid'da | Qanday | Isbot |
|---|---|---|---|
| O(T²) hisoblash: matn 2x uzaysa, attention 4x sekinlashadi | O(T) | chunk'li parallel delta algoritmi | `bench_speed` |
| KV-cache har token bilan o'sadi (16K kontekstda 128 MB) | **0.29 MB, o'zgarmas** | xotira — qat'iy o'lchamli matritsa S ∈ R^{Dk×Dv} | `bench_speed`, `test_state_size_does_not_grow_with_context` |
| Uzun kontekstda har bir yangi token sekinlashadi | o'zgarmas tezlik | generatsiyada 1 qadam = 1 matritsa yangilanishi | `bench_speed` |
| O'qitilganidan uzun matnda sifat tushadi (pozitsiya embedding) | pozitsiya embedding yo'q | tartibni unutish darvozasi α va qisqa konvolyutsiya beradi | `bench_extrapolation` |
| Holatni kuzata olmaydi (parity, hisoblagich) | kuzata oladi | β ∈ (0, 2) → (I − β k kᵀ) aks ettirish bo'la oladi | `bench_parity` |
| Softmax "1 ga yig'indi" majburiyati → attention sink, keraksiz tokenlarga e'tibor | softmax yo'q | yozish kuchi β va unutish α har token uchun alohida, chiqishda sigmoid-darvoza | dizayn |
| Oddiy linear attention'da xotira "ifloslanadi" (qiymatlar qo'shilib ketadi) | almashtiradi | delta rule: S ← S + β·k·(v − Sᵀk)ᵀ | `test_delta_rule_overwrites_instead_of_accumulating`, `bench_recall` |

## Arxitektura

Har bir blok (pre-norm):

```
x = x + GatedDeltaMemory(RMSNorm(x))      # attention o'rniga
x = x + SwiGLU(RMSNorm(x))
```

**Gated Delta Memory** (har bir head uchun, `umid/delta.py`):

```
q, k, v = ShortConv(W·x)              # 4 tokenli kauzal konvolyutsiya + SiLU
q, k    = L2norm(q), L2norm(k)
β_t     = 2·σ(W_β x)                  # yozish kuchi ∈ (0, 2)
α_t     = exp(−A · softplus(W_α x + b))  # unutish ∈ (0, 1], har head o'z tezligida

S_t = α_t · (I − β_t k_t k_tᵀ) · S_{t−1} + β_t · k_t v_tᵀ     # xotirani yangilash
o_t = RMSNorm(S_tᵀ q_t) ⊙ SiLU(W_g x)                         # o'qish + chiqish darvozasi
```

- **Delta rule**: model avval xotiradan `k` kalit bo'yicha nima "eslashini" (`Sᵀk`) hisoblaydi,
  so'ng faqat xatoni (`v − Sᵀk`) yozadi. Shuning uchun bir kalitga yangi qiymat yozilsa, eskisi
  almashtiriladi.
- **β ∈ (0, 2)**: β > 1 bo'lganda `(I − β k kᵀ)` manfiy xos qiymatga ega bo'ladi. Bu xotiradagi
  ishorani almashtira oladi, shuning uchun model juft/toq kabi holatlarni kuzata oladi.
- **O'qitish**: `delta_rule_chunked` ketma-ketlikni 64 tokenli chunk'larga bo'ladi. Chunk ichida
  rekurrensiya uchburchak tenglamalar sistemasi orqali parallel yechiladi (WY/UT ko'rinishi),
  chunk'lar orasida faqat S uzatiladi. Natija token-ma-token rekurrensiya bilan float64 da
  1e-15 aniqlikda mos keladi (`tests/test_delta.py`).
- **Generatsiya**: `delta_rule_step`, ya'ni har token uchun bitta yangilanish. Butun suhbat
  `S` matritsalarida saqlanadi: tiny modelda jami 0.29 MB.

**Fikrlash rejimi — xotiradan qayta so'rash** (`n_reads=2`, ixtiyoriy):

```
o¹_t = S_tᵀ q_t                        # birinchi o'qish
q²_t = L2norm(W_r · o¹_t)              # javob yangi so'rovga aylanadi (har head uchun Dv × Dk matritsa)
o²_t = S_tᵀ q²_t                       # o'sha xotiradan ikkinchi o'qish
o_t  = softmax(W_m x)₁·o¹_t + softmax(W_m x)₂·o²_t    # har token qaysi o'qish kerakligini tanlaydi
```

Bitta qatlam ichida "f(q) ni top, keyin f(f(q)) ni top" kabi ikki bosqichli qidiruv bajariladi.
Xotira hajmi o'zgarmaydi. Barcha o'qishlar chunk'ning bir xil (u, w, S) hisob-kitobidan
foydalanadi, shuning uchun algoritm aniqligicha qoladi (`test_multi_read_matches_recurrent`).
Bu rejim mantiqiy masalalarda kuchli, lekin 256 tokenli LM o'qitishda yomonroq. Shu sababli u
standart emas, `--n-reads 2` bilan yoqiladi ("Aql tadqiqoti" bo'limiga qarang).

Bu g'oyalarning ilmiy asoslari: DeltaNet (Yang va boshq., 2024), Gated DeltaNet (Yang, Kautz,
Hatamizadeh, ICLR 2025), manfiy xos qiymatlar orqali holat kuzatish (Grazzi va boshq., ICLR 2025),
Mamba-2 uslubidagi unutish darvozasi. Umid ularni bitta ixcham, tashqi kutubxonasiz
(faqat PyTorch) modelga birlashtiradi. Unda o'zbek tiliga mos tokenizer, o'qitish va suhbat
vositalari ham bor.

## Loyiha tuzilishi

```
umid/
  delta.py      # yadro: rekurrent, chunk'li va bir qadamli Gated Delta algoritmi (+ qayta so'rash)
  model.py      # UmidLM: qatlamlar, generatsiya, saqlash/yuklash
  optimal.py    # tadqiqot: aniq eng kichik kvadratlar xotirasi (Gᵀ(H + Λ)⁻¹q)
  tokenizer.py  # bayt darajasidagi BPE (o‘, g‘, ' belgilari so'z ichida qoladi)
  data.py       # korpus (tozalash, takrorlash), batch'lar, savol-javob/dialog kodlash, baholash
  dialog.py     # suhbatdan o'rganish uchun sintetik dialoglar (train/test qiymatlari kesishmaydi)
  baseline.py   # taqqoslash uchun klassik Transformer (RoPE + softmax attention, ixtiyoriy Canon, hujjat maskasi)
train.py        # o'qitish (pretraining); --arch gpt — Transformer'ni aynan shu sharoitda o'qitish
sft.py          # savol-javob fine-tuning: loss faqat javob tokenlarida
chat.py         # suhbat / generatsiya
chat-data/      # Data.txt (umumiy matn) va savol-javob/ (731 juftlik, shundan 431 tasi qayta ifoda)
tests/          # pytest testlari
benchmarks/     # Umid va Transformer'ni solishtirish (natijalar: benchmarks/results/*.json)
  qa_eval.py      # o'qitishda bo'lmagan 60 savol (qa_heldout.txt, qa_heldout2.txt)
  qa_nll.py       # javob NLL'i: nazorat savoli va asl savol bilan
  qa_diagnose.py  # bilim (300 javob ichida o'rin), so'z ta'siri (occlusion), savol tasviri
  icl_eval.py     # suhbatdan o'rganish: aytilgan yangi faktni masofa bo'yicha eslash
```

## O'qitish

```bash
python train.py                                   # tiny (4.5M): tozalangan Data.txt + savol-javoblar ×8
python train.py --reset-on-eot --persistent-heads 3   # suhbat modeli uchun tavsiya ("Savol-javob sifati")
python train.py --arch gpt --reset-on-eot         # taqqoslash uchun Transformer, aynan shu sharoitda
python sft.py --init checkpoints/umid/model.pt --out checkpoints/umid_chat   # savol-javob fine-tuning
python train.py --preset small --steps 5000       # 6 qatlam, d=384
python train.py --data korpus_papkasi/ --preset base --seq-len 1024 --compile   # GPU uchun
python train.py --stream                          # xotira holatini oynadan oynaga uzatib o'qitish
python train.py --out checkpoints/umid --resume   # to'xtagan joydan davom ettirish
python train.py --n-reads 2                       # fikrlash rejimi (xotiradan qayta so'rash)
```

- Presetlar: `tiny` (d=256, 4 qatlam), `small` (384, 6), `base` (768, 12), `large` (1024, 24).
- CUDA'da avtomatik bf16; `--compile` orqali `torch.compile` yoqiladi.
- `--stream` rejimida har bir batch qatori matnning ketma-ket bo'lagini oladi, xotira esa
  qadamdan qadamga uzatiladi. Bunda model `--seq-len` dan uzoqroq bog'lanishlarni o'rganadi,
  narx esa oshmaydi.
- Eng yaxshi model `checkpoints/umid/model.pt` ga saqlanadi (tokenizer ichida).
- `--resume` bilan `--steps` ni oshirsangiz, lr jadvali yangi uzunlikka cho'ziladi va lr vaqtincha
  ko'tariladi.
- `--data` ro'yxat qabul qiladi: `!` — manbani tozalash (yopishgan so'zlar, manbalar ro'yxati, URL,
  inglizcha bo'laklar), `*N` — N marta takrorlash (takrorda savol-javob bloklari aralashtiriladi va
  EOT bilan ajratiladi). Standart: `chat-data/Data.txt!,chat-data/savol-javob*8`.
- `--reset-on-eot`: EOT da xotira (va qisqa konvolyutsiya chegarasi) tozalanadi, hujjatlar bir-biriga
  aralashmaydi. `--persistent-heads N`: N ta head unutmaydi (α = 1). `--canon ACD`: Canon qatlamlari.
- Standart sozlamalar (`tiny`, 600 qadam) bilan o'qitish 4 yadroli CPU'da ~15 daqiqa oladi. Val loss
  ma'lumotga bog'liq: eski `Data.txt` da 4.04–4.11 (3 seed), hozirgi tozalangan matn + aralashtirilgan
  savol-javoblarda ~4.18–4.27 (bu raqamlar bir-biri bilan solishtirilmaydi). 1200 qadamda model
  300-qadamdan keyin yodlay boshlaydi va val loss oshadi, shuning uchun qadamlar soni cheklangan.

## Suhbat

```bash
python chat.py --ckpt checkpoints/umid/model.pt
python chat.py --ckpt checkpoints/umid/model.pt --prompt "Bir kuni" --max-new-tokens 200
```

Suhbat davomida model butun suhbatni o'zgarmas hajmli xotirada eslaydi (`/xotira` buyrug'i
hajmini ko'rsatadi). `/yangi` xotirani tozalaydi, `/chiqish` dasturdan chiqadi. Format `sft.py` bilan
bir xil: `Savol: …\nJavob: …\n`. Takrorlash jarimasi faqat javobga qo'llanadi. Model suhbatda siz
aytgan yangi faktlarni ishlatishi uchun alohida o'qitish kerak. Bu hali ishlamaydi, "Suhbatdan
o'rganish" bo'limiga qarang.

## Python'dan foydalanish

```python
import torch
from umid import BPETokenizer, UmidLM

model, ckpt = UmidLM.load("checkpoints/umid/model.pt")
tok = BPETokenizer.from_dict(ckpt["tokenizer"])

ids = torch.tensor([tok.encode("Bir kuni")])
new, state = model.generate(ids, max_new_tokens=60, temperature=0.8)
print(tok.decode(new[0].tolist()))

# davom ettirish: holat oldingi matnni eslaydi, qaytadan o'qish shart emas
more, state = model.generate(torch.tensor([tok.encode(" Keyin")]), 40, states=state)
print(UmidLM.state_bytes(state), "bayt xotira")   # matn qancha uzun bo'lmasin, o'zgarmaydi
```

## Testlar

```bash
python -m pytest tests
```

Testlar quyidagilarni tekshiradi:
- chunk'li algoritm = rekurrent etalon (natija va gradientlar);
- kauzallik;
- prefill + token-ma-token generatsiya = to'liq forward;
- xotira hajmi kontekstga bog'liq emas;
- 20 000 tokenda holat portlamaydi;
- model kichik batch'ni yodlay oladi;
- tokenizer matnni aynan qaytaradi;
- saqlash/yuklash.

## Benchmark natijalari

Barcha o'lchovlar CPU'da (4 yadro). Ikkala modelda sharoit bir xil: o'lcham, tokenizer, optimizer,
qadamlar soni. Taqqoslanadigan model — zamonaviy Transformer (Llama uslubi: RoPE, SwiGLU,
RMSNorm, `F.scaled_dot_product_attention`). Har bir jadvalni bitta buyruq bilan qayta hosil
qilish mumkin.

### 1. Uzun kontekst — ekstrapolyatsiya (`python -m benchmarks.bench_extrapolation`)

`chat-data/Data.txt` ning o'zida o'qitildi: ikkala model ham 4.3–4.5M parametr, 128 tokenli
oynalar, 800 qadam. Baholash ko'rilmagan matnda (korpusning turli joylaridan olingan 16K
token), uzunroq oynalarda. O'lchov — val loss (nat/token, past = yaxshi), eng yaxshi checkpoint.

| kontekst | **Umid** | Transformer (lr 2e-3) | Transformer (lr 1e-3, eng yaxshi) |
|---:|---:|---:|---:|
| 128 (o'qitilgan) | **4.556** | 5.043 | 4.678 |
| 256 | **4.502** | 5.050 | 4.679 |
| 512 | **4.457** | 5.099 | 4.718 |
| 1 024 | **4.399** | 5.176 | 4.774 |
| 2 048 (16× uzun) | **4.274** | 5.261 | 4.818 |

Transformer o'qitilganidan uzun matnda yomonlashadi: RoPE pozitsiyalari unga notanish.
Umid esa uzun kontekstdan **foyda oladi**: 2048 tokenda loss 128 tokendagidan 0.28 nat past.
Transformer'ga ikkita lr berildi va eng yaxshisi ko'rsatildi. Umid o'qitilgan uzunlikda ham
oldinda.

### 2. Holatni kuzatish — parity (`python -m benchmarks.bench_parity`)

Bitlar ketma-ketligi beriladi, model har bir pozitsiyada "1" lar soni juft yoki toq ekanini
aytishi kerak. O'qitish 40 uzunlikda, sinov undan uzunroq ketma-ketliklarda. Jadvalda oxirgi
token aniqligi (3 seed'dan eng yaxshisi). 50% = tasodifiy taxmin.

| uzunlik | **Umid** (β∈(0,2)) | Umid, β∈(0,1) (ablation) | Transformer |
|---:|---:|---:|---:|
| 40 (o'qitilgan) | **100%** | 91.4% | 52.3% |
| 80 | **100%** | 52.7% | 52.7% |
| 160 | **98.8%** | 52.3% | 52.3% |
| 320 | 74.2% | 50.8% | 49.2% |
| 640 | 53.1% | 52.0% | 52.3% |

Transformer o'qitilgan uzunlikda ham masalani yecha olmadi. β∈(0,1) li variant (oddiy
DeltaNet/Mamba kabi) o'qitilgan uzunlikni yodlaydi, lekin umumlashtira olmaydi. Umid 4 barobar
uzun ketma-ketlikkacha deyarli xatosiz ishlaydi. Natija uchala seed'da bir xil.

### 3. Assotsiativ eslab qolish — MQAR (`python -m benchmarks.bench_recall`)

Kontekstda `kalit qiymat` juftliklari, keyin kalitlar bo'yicha so'rovlar beriladi. Bu
Transformer'ning kuchli tomoni, qat'iy xotirali modellar uchun esa eng qiyin test. O'qitish
16 juftlikda; 32 va 48 juftlik o'qitishda ko'rilmagan.

| juftliklar | **Umid** (2000 qadam, lr 2e-3) | Transformer (2000 qadam, lr 2e-3) | Transformer (6000 qadam, lr 1e-3) |
|---:|---:|---:|---:|
| 16 | **100%** | 12.3% | **100%** |
| 32 | 99.7% | 7.7% | **99.9%** |
| 48 | **96.6%** | 6.3% | 95.8% |

Xulosa: **durrang**. To'g'ri lr bilan Transformer ham masalani yechadi. Lekin Umid qat'iy
o'lchamli xotira bilan Transformer darajasida eslab qoladi. Bunga delta rule'ning eski qiymatni
almashtirishi sabab. Transformer lr'ga sezgirroq: 2e-3 da u o'rgana olmadi.

### 4. Tezlik va xotira (`python -m benchmarks.bench_speed`)

Har ikki model tiny o'lchamda (d=256, 4 qatlam). Kontekst o'qilgandan keyin har bir yangi
token uchun:

| kontekst | Umid ms/token | Umid xotirasi | Transformer ms/token | Transformer KV-cache |
|---:|---:|---:|---:|---:|
| 256 | 4.8 | 0.29 MB | 2.1 | 2 MB |
| 1 024 | 4.9 | 0.29 MB | 2.6 | 8 MB |
| 4 096 | 3.1 | 0.29 MB | 5.1 | 32 MB |
| 16 384 | **4.0** | **0.29 MB** | 22.1 | 128 MB |

O'qitish (forward + backward, batch'da 8192 token):

| T | Umid token/s | Transformer token/s |
|---:|---:|---:|
| 256 | 4 368 | 8 064 |
| 1 024 | 3 936 | 6 066 |
| 4 096 | 2 850 | 4 639 |
| 8 192 | 3 436 | 3 387 |

Generatsiyada Umid'ning xotirasi va tezligi kontekstga bog'liq emas. 16K kontekstda u 5.5×
tezroq, xotirasi esa 440× kam. Transformer'da ikkalasi ham chiziqli o'sadi.

## Aql tadqiqoti: bir xil parametrda Transformer'dan aqlliroq qatlam izlash

Maqsad: teng parametr sonida Transformer'dan ancha "aqlliroq" qatlam topish. Qoidalar:
- har bir taqqoslashda parametrlar MLP o'lchami orqali ±1% aniqlikda tenglashtirildi
  (`benchmarks/bench_reasoning.py`);
- Transformer'ga bir nechta lr va ko'proq qadam berildi;
- da'volar faqat bir nechta seed'da takrorlangandagina qabul qilindi;
- barcha raqamlar `benchmarks/results/` da.

**Qisqa javob: teng parametrda umumiy "10x aqlliroq" model topilmadi.** Bunga erishilgan deb
da'vo qilinmaydi. Quyida aniq, takrorlangan yutuqlar va ishlamagan g'oyalar keltirilgan.

### Tasdiqlangan yutuqlar

**1. Mantiqiy zanjir (`--task hopmix`, ~430K parametr).** Kontekstda `x f(x)` juftliklari bor,
so'ng "1 bosqich: f(q)?" yoki "2 bosqich: f(f(q))?" so'raladi.

| model | 1 bosqich | 2 bosqich |
|---|---:|---:|
| Transformer (3 xil lr, 6000 qadamgacha, 4 urinish) | 18–21% | 18–20% |
| Umid (bitta o'qish) | 56% | 19% |
| **Umid, fikrlash rejimi** (`n_reads=2`), seed 0 / 1 / 2 | **80% / 84% / 74%** | **47%** / 21% / **40%** |

- Bitta qidiruvda fikrlash rejimi Transformer'dan **~4x** ustun, 3 seed'ning uchalasida.
- Ikki bosqichli kompozitsiyani u **3 seed'dan 2 tasida** o'rgandi (~2.2x).
- Transformer va bitta o'qishli Umid kompozitsiyani hech qachon o'rganmadi.
- Sof 2 bosqichli masalada (aralashtirmasdan) 6000 qadamda ham hech bir model o'rganmadi
  (~20%).

**2. Kattaroq masalaga umumlashtirish — eslab qolish (`--task mqar`, ~115K parametr).** O'qitish
32 juftlikda, sinov undan ko'p juftlikda. Har ikki model ham 3 seed'dan 2 tasida o'rgandi,
uchinchisida platoda qoldi. Jadvalda o'rgangan seed'lar:

| | 32 (o'qitilgan) | 48 | 64 |
|---|---:|---:|---:|
| Transformer (2 seed) | 100% / 100% | 84.6% / 83.8% | 51.3% / 40.8% |
| **Umid** (2 seed) | 99.0% / 99.1% | 95.6% / 94.3% | **88.6% / 85.5%** |

O'qitilgan o'lchamda farq yo'q. 2 barobar katta masalada Umid **~1.9x** yaxshiroq.

**3. O'zbekcha matn.** Teng o'lcham, 3 ta Umid seed va 2 ta Transformer seed o'rtachasi. Umid
qisqa kontekstda ~1.1x, 2048 tokenda **~1.7x** past perplexity beradi (yuqoridagi
1-jadval).

### Aralash natijalar

- **Fikrlash rejimi o'zbekcha matnda** (`bench_lm_variants.py`, juftlangan seed'lar):
  - 128 tokenli o'qitishda val loss'ni 3 seed'ning uchalasida pasaytirdi (o'rtacha −0.14 nat);
  - standart 256 tokenli `train.py` o'qitishida esa uchalasida oshirdi (o'rtacha +0.16 nat,
    `product_v1_vs_fikr.json`).

  Shuning uchun u standart emas. `read_mix_init` farqni +0.26 dan +0.11 gacha kamaytiradi, lekin
  yo'qotmaydi.
- **Kanal bo'yicha unutish** (`decay="channel"`, Kimi Delta Attention g'oyasi):
  - LM da kichik foyda bor (−0.09 va −0.002, 2 seed);
  - lekin eslab qolishni 100% dan 45% gacha va parity umumlashtirishini sezilarli tushirdi;
  - shuning uchun standart emas.
- **Ko'p qadamli yozish** (`n_writes=2`, Householder ko'paytmasi):
  - S3 almashtirishlar guruhida 41.5% → 50.6% (Transformer 27.9%), 1 seed;
  - eslab qolishda o'qitish platoda qolib ketdi.

### Ishlamagan g'oyalar (salbiy natijalar)

- **Optimal xotira** (`mixer="optimal"`, `umid/optimal.py`).
  - G'oya: xotira har tokenda regulyarizatsiyalangan eng kichik kvadratlar masalasini aniq
    yechadi: `o = Gᵀ(H + Λ)⁻¹q`.
  - Mexanizm ishlaydi: kuchli korrelyatsiyalangan 8 kalitda eslash xatosi 0.00003, delta rule'da
    3.63.
  - Lekin vazifalarda foyda bermadi: eslab qolishda 64 juftlikda 81.4% (oddiy Umid 85–89%),
    S3 da 41.9%.
  - Ustiga 3–10x sekinroq.
- **Aylanadigan koordinata tizimidagi optimal xotira** (`frame=True`, ushbu loyihada taklif
  qilingan birikma). S3 da 36.7%, oddiy Umid'dan yomon.
- **Fikrlash halqasi** (`n_loops=2`, parametrlarni qayta ishlatish). Hech bir vazifada aniq
  foyda bermadi.
- **3 marta qayta so'rash**: 2 martadan yomon.
- **Kattaroq qiymat o'lchami** (`expand_v=2`): +17% parametr evaziga deyarli foyda yo'q.
- **"Qattiq" darvozalar** (β va α chegaraga aniq yetadi): parity'da seed'ga qarab yaxshi yoki
  yomon, ishonchsiz.

### Xulosa

Teng o'lchamda Umid ba'zi qobiliyatlarda Transformer'dan bir necha barobar ustun:
- holatni kuzatish;
- bitta qidiruv;
- katta masalaga umumlashtirish;
- uzun kontekst.

Umumiy aqlda 10x farq esa bu tadqiqotda kuzatilmadi. Fikrlash rejimi eng istiqbolli yo'nalish:
u ko'p bosqichli mulohazani bitta qatlamga olib kiradi. Lekin uni LM o'qitishda ham foydali
qilish ochiq vazifa bo'lib qoladi.

## Savol-javob sifati: Umid va Transformer

**Usul.**
- Ma'lumot: tozalangan `Data.txt` va 731 ta savol-javob (300 asl, 431 qayta ifoda). Savol-javoblar
  ×4 takrorlanadi va har takrorda aralashtiriladi.
- Bosqichlar: `train.py` (600 qadam), keyin `sft.py` (300 qadam, loss faqat javobda).
- Transformer aynan shu retsept, tokenizer va o'lchamda o'qitiladi (`--arch gpt`, 4.26M; Umid 4.55M).
- Baholash: `qa_eval.py`, o'qitishda bo'lmagan 60 ta savol, greedy. 2-to'plam qayta ifoda
  ma'lumoti yozilishidan oldin commit qilingan. Nazorat savollari o'qitishga aynan tushmagani
  tekshirilgan.

**Natijalar** (to'g'ri javoblar ulushi, har raqam alohida seed):

| model (tiny) | takrorlash jarimasi 1.15 | jarimasiz |
|---|---|---|
| Umid, eski ma'lumot (tartiblangan savol-javob ×8, SFT'siz) | 5–10% | — |
| Umid + qayta ifodalar + SFT | 23% | — |
| + EOT da xotirani tozalash | 32% | 35% |
| + 2 ta doimiy head | 43, 38% | — |
| + 3 ta doimiy head | 45, 47, 37% | 53, 50, 47% |
| + barcha head doimiy | 50, 38% | 57, 48% |
| + 3 doimiy head + Canon | 53, 35% | 57, 47% |
| Transformer (hujjat maskasiz) | 45% | — |
| **Transformer** (hujjat maskasi) | **52, 52, 55%** | **57, 55, 57%** |
| Transformer + Canon | 53% | 58% |

**Topilmalar** (`qa_diagnose.py`):
- Asosiy muammo — tez unutish. O'qitilgan Umid savolning birinchi so'zini javob boshlanguncha
  4–78% saqlab qolardi (birinchi qatlamda 4–30%). Doimiy head'lar (`persistent_heads`, faqat EOT da
  tozalanadi) buni tuzatdi: Umid savolning bosh so'ziga Transformer'dan 2.3× kuchliroq tayanadi.
- Bilimda farq kichik. 300 ta javob ichida to'g'risining o'rni:
  - Transformer: top-1 41%, top-5 53%, median 4;
  - Umid (3 doimiy head): top-1 39%, top-5 56%, median 2;
  - Umid + Canon: top-1 44%, top-5 58%, median 2.
- Takrorlash jarimasi savol so'zlarini ham jazolaydi va Umid'ga ko'proq zarar qiladi. Shuning uchun
  `chat.py` jarimani faqat javobga qo'llaydi.
- **Xulosa (halol):**
  - Oddiy savol-javobda Transformer hali ~5 ball oldinda (jarimasiz o'rtacha ~56% ga qarshi ~51%).
  - Umid natijasi seed'ga kuchli bog'liq (35–57%), Transformer'niki barqaror.
  - Canon Umid'da bitta seed'da +8 ball berdi, ikkinchisida tasdiqlanmadi.

**Sinab ko'rilgan, foyda bermagan g'oyalar** (kodi olib tashlangan yoki standart emas):
- "Just read twice" (savolni ikki marta o'qish): Umid 45→42%, Transformer 52→48%;
- beam search;
- lr 3e-3 (30%);
- 2× uzoq o'qitish (yodlash, 22%);
- o'chirish darvozasini ajratish (`decouple_erase`, 32–40%);
- fikrlash rejimi (33%).

## Suhbatdan o'rganish (davom etayotgan ish)

Maqsad: model suhbatda aytilgan yangi faktni eslab qolsin. Fakt — o'qitishda hech uchramagan ism,
shahar, raqam yoki o'ylab topilgan so'z va uning ta'rifi. Model uni oradagi boshqa savollardan
keyin ishlatishi kerak. Umid'da bu xotira matritsasida bo'ladi: delta qoidasi kalit-qiymat
bog'lanishini o'rganishning bir gradient qadamiga teng.

- Vositalar:
  - `umid/dialog.py`: train va test qiymatlari kesishmaydi;
  - `sft.py --dialog-frac 0.5` (har qadamda yangi dialoglar);
  - `benchmarks/icl_eval.py`: masofa 0…40 oraliq savol, ~70…1400 token.
- Hozirgi holat — **ishlamaydi**:
  - dialog bilan fine-tuning qilingan ikkala model ham kontekstdan ko'chirmaydi (0–6%);
  - ular o'qitishda ko'p uchragan qiymatni taxmin qiladi;
  - faqat "aytmagansiz" holatini to'g'ri o'rganadi.
- Keyingi qadam:
  - har qadamda yangi dialoglar yaratish (yodlab bo'lmaydi);
  - dialog bosqichini uzaytirish (ko'chirish mexanizmini noldan o'rganish uchun 600 qadam kam).

## Halol cheklovlar

- **Oddiy savol-javobda Transformer hali oldinda** (yuqoridagi bo'lim): o'rtacha ~5 ball, Umid'da
  seed'lar orasidagi tebranish katta. Suhbatdan o'rganish hali ishlamaydi.

- **Qisqa matnlarda o'qitish CPU'da sekinroq**: T ≤ 4096 da Transformer 1.6–1.9× tezroq. Sababi —
  attention uchun PyTorch'da tayyor optimallashtirilgan yadro (`scaled_dot_product_attention`)
  bor, Umid'ning delta algoritmi esa oddiy PyTorch operatsiyalarida yozilgan. GPU'da Triton
  yadrosi (masalan, `flash-linear-attention` dagi `chunk_gated_delta_rule`) bu farqni yo'qotadi.
  Hozircha `--compile` qisman yordam beradi. T = 8192 da tezlik tenglashadi, undan uzunda Umid
  tezroq.
- **Parity 4× uzunlikkacha deyarli mukammal**, 8× da 74%, 16× da tasodifiy. β aniq 2 ga yetmaydi,
  shuning uchun aks ettirish juda uzun ketma-ketlikda asta-sekin "so'nadi". Transformer va
  β∈(0,1) variant esa 2× da ham tasodifiy.
- **Xotira hajmi cheklangan**: har bir head'da Dk×Dv son bor. Juda ko'p faktni so'zma-so'z eslab
  qolish (masalan, 10 000 ta telefon raqam) Transformer'ning to'liq KV-cache'idan yomonroq
  bo'ladi. Buning evaziga xotira va tezlik o'zgarmas.
- **Ma'lumot kichik**: `Data.txt` atigi ~200K token. Bunday hajmda har qanday model (Transformer
  ham) faqat jumla bo'laklarini o'rganadi va ma'noli suhbat qura olmaydi. Haqiqiy sifat uchun
  yuzlab million token o'zbekcha matn kerak (masalan, CulturaX/mC4 ning `uz` qismi,
  Vikipediya). Arxitektura va vositalar bunga tayyor: `python train.py --data katta_korpus/
  --preset base`.
- Bu natijalar kichik modellarda (≤4.7M parametr) olingan. Milliardlab parametrli darajada
  Gated DeltaNet oilasi modellari Transformer bilan raqobatlashishi ilmiy maqolalarda
  ko'rsatilgan. Lekin bu loyihada bunday masshtab sinab ko'rilmagan.
- **100M vs 100M solishtirish o'tkazilmagan.** Uchta sabab bor:
  - u GPU talab qiladi;
  - ~2 milliard token o'zbekcha matn kerak;
  - ishlab chiqish muhitida faqat 4 ta CPU yadrosi bor edi va Hugging Face'ga kirish yopiq edi.

  Kod bunga tayyor. Masalan:

  ```
  python train.py --data korpus/ --preset large --seq-len 1024 --compile
  ```

  Transformer bilan taqqoslash uchun `umid/baseline.py` dan foydalanish mumkin.

---

## Eski kod (GPT-J fine-tune)

`Fine-Tune.py` — oldingi yondashuv: tayyor GPT-J-6B (Transformer) modelini fine-tune qilish.
Model hajmi ~20 GB, kamida 48 GB RAM kerak. Umid esa noldan o'qitiladi va oddiy kompyuterda ishlaydi.

<details>
<summary>Eski README</summary>

```
pip3 install torch transformers
```

~~~python
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
def load_fine_tuned_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=500, temperature=1.5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
config.max_position_embeddings = 3048
model = GPT2LMHeadModel(config)
model_path = "path/to/path"
fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_path)

while True:
    prompt = input("You:").lower()
    if prompt.lower() == "exit":
        print("AI is Sleeping. Goodbye!")
        break
    else:
        try:
            print(f"Your Question: {prompt}")
            response = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt, temperature=1.0)
            print("AI:", response)
        except IndexError as err:
            print("Empty question Error")
~~~

</details>

Muallif: Muhiddinov Abdulhodiy
