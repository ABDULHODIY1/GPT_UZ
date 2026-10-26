# GPT_UZ — Umid: Transformer attention'siz o'zbekcha til modeli

**Umid** — PyTorch'da noldan yozilgan yangi turdagi til modeli (LLM). Unda Transformer'ning
softmax attention'i yo'q. Uning o'rniga har bir head'da qat'iy o'lchamli **assotsiativ xotira**
bor: **Gated Delta Memory**. Bu xotira har bir token bilan "xato bo'yicha tuzatish" (delta rule)
qoidasi asosida yangilanadi.

```
pip install -r requirements.txt
python train.py                                  # chat-data/Data.txt da o'qitish
python chat.py --ckpt checkpoints/umid/model.pt  # suhbat
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

Bu g'oyalarning ilmiy asoslari: DeltaNet (Yang va boshq., 2024), Gated DeltaNet (Yang, Kautz,
Hatamizadeh, ICLR 2025), manfiy xos qiymatlar orqali holat kuzatish (Grazzi va boshq., ICLR 2025),
Mamba-2 uslubidagi unutish darvozasi. Umid ularni bitta ixcham, tashqi kutubxonasiz
(faqat PyTorch) modelga birlashtiradi. Unda o'zbek tiliga mos tokenizer, o'qitish va suhbat
vositalari ham bor.

## Loyiha tuzilishi

```
umid/
  delta.py      # yadro: rekurrent, chunk'li va bir qadamli Gated Delta algoritmi
  model.py      # UmidLM: qatlamlar, generatsiya, saqlash/yuklash
  tokenizer.py  # bayt darajasidagi BPE (o‘, g‘, ' belgilari so'z ichida qoladi)
  data.py       # korpus, batch'lar, baholash
  baseline.py   # taqqoslash uchun klassik Transformer (RoPE + softmax attention)
train.py        # o'qitish
chat.py         # suhbat / generatsiya
tests/          # pytest testlari
benchmarks/     # Umid va Transformer'ni solishtirish (natijalar: benchmarks/results/*.json)
```

## O'qitish

```bash
python train.py                                   # tiny (4.5M), chat-data/Data.txt
python train.py --preset small --steps 5000       # 6 qatlam, d=384
python train.py --data korpus_papkasi/ --preset base --seq-len 1024 --compile   # GPU uchun
python train.py --stream                          # xotira holatini oynadan oynaga uzatib o'qitish
python train.py --out checkpoints/umid --resume   # to'xtagan joydan davom ettirish
```

- Presetlar: `tiny` (d=256, 4 qatlam), `small` (384, 6), `base` (768, 12), `large` (1024, 24).
- CUDA'da avtomatik bf16; `--compile` orqali `torch.compile` yoqiladi.
- `--stream` rejimida har bir batch qatori matnning ketma-ket bo'lagini oladi, xotira esa
  qadamdan qadamga uzatiladi. Bunda model `--seq-len` dan uzoqroq bog'lanishlarni o'rganadi,
  narx esa oshmaydi.
- Eng yaxshi model `checkpoints/umid/model.pt` ga saqlanadi (tokenizer ichida).

## Suhbat

```bash
python chat.py --ckpt checkpoints/umid/model.pt
python chat.py --ckpt checkpoints/umid/model.pt --prompt "Bir kuni" --max-new-tokens 200
```

Suhbat davomida model butun suhbatni o'zgarmas hajmli xotirada eslaydi (`/xotira` buyrug'i
hajmini ko'rsatadi). `/yangi` xotirani tozalaydi, `/chiqish` dasturdan chiqadi.

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

## Halol cheklovlar

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
- Bu natijalar kichik modellarda (≤4.5M parametr) olingan. Milliardlab parametrli darajada
  Gated DeltaNet oilasi modellari Transformer bilan raqobatlashishi ilmiy maqolalarda
  ko'rsatilgan. Lekin bu loyihada bunday masshtab sinab ko'rilmagan.

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
