# Reja: Transformer'dan o'tib ketadigan yangi algoritm

## Qayerdan boshlaymiz (o'lchangan faktlar)

- **Oddiy savol-javob** (60 nazorat savoli): Transformer ~56%, Umid ~51%.
- **Umid seed'ga sezgir**: 35–57%. Transformer barqaror: 52–57%.
- **Diagnostika** (`benchmarks/qa_diagnose.py`):
  - Umid savolning boshini unutardi; doimiy head'lar buni qisman tuzatdi.
  - Bilimda farq kichik: to'g'ri javobning median o'rni Umid'da 2, Transformer'da 4.
  - Farq aniq tanlovda: savoldagi eng muhim so'zni (masalan "Ibn Sino", "Everest") javob paytida
    aniq topish.
- **Sabab — xotiraning tuzilishi.**
  - Delta xotira hamma narsani bitta matritsaga ustma-ust yozadi, o'qish chiziqli: muhim so'z
    boshqalar bilan aralashadi.
  - Transformer attention har bir oldingi tokenni alohida saqlaydi va keraklisini aniq tanlaydi.
    Lekin uning narxi o'sib boruvchi KV-cache, O(T²) va softmax "1 ga yig'indi" majburiyati.

## Yangi algoritm: Tanlov xotirasi (eski delta qoidasidan foydalanmaydi)

G'oya: xotira matritsa emas, **K ta eng muhim tokenning o'zi**. Model qaysi token muhimligini o'zi
o'rganadi.

1. **Muhimlik.** Har token o'zining muhimligini baholaydi: `s_t = w_s · x_t` (o'rganiladi).
2. **Tanlab saqlash.** Har head'da K ta joy bor. Yangi token kirganda eng muhim K ta token qoladi,
   eng kami chiqariladi. Xotira hajmi **o'zgarmas** (K), KV-cache o'smaydi.
   - Chiqarilgan token hech qachon qaytmaydi: undan muhimroq K ta token doim bor.
   - Shuning uchun o'qitishda (parallel, chunk'larda) va generatsiyada natija **aynan bir xil**.
3. **Aniq o'qish, softmax'siz.** Har saqlangan token alohida tanlanadi:
   `w_tj = σ(τ·q̂_t·k̂_j + b + log σ(s_j) − λ·log(1 + t − j))`.
   - Sigmoid: "1 ga yig'indi" majburiyati va attention sink yo'q.
   - Muhimlik og'irlikka qo'shiladi: gradient muhim tokenlarni topishni o'rgatadi.
   - Masofa jarimasi logarifmik (pozitsiya embedding yo'q): har head "yaqin" yoki "uzoq" bo'lishni
     o'zi tanlaydi va uzunroq matnga ekstrapolyatsiya qiladi.
4. **EOT chegarasi.** Yangi hujjatda xotira bo'shaydi, hujjatlar aralashmaydi.
5. **Boshlang'ich holat.** Hamma muhimlik teng bo'lgani uchun model oxirgi K ta tokenga qaraydigan
   "sirpanuvchi oyna" bo'lib boshlanadi. O'qish davomida muhim tokenlarni uzoqroq ushlashni o'rganadi.

Murakkablik:
- generatsiyada har token uchun O(K) vaqt va O(K) xotira, kontekstga bog'liq emas;
- o'qitishda chunk ichida O(C·(C+K)).

Eng yaqin ma'lum ishlar: TOVA va H2O (o'qitilgan Transformer'da qo'lda yozilgan qoida bilan token
chiqarish), sigmoid attention (Ramapuram va b., 2024), o'rganiladigan token chiqarish bilan gibrid
modellar (2025). Farqimiz:
- muhimlik oxiridan-oxirigacha o'qitiladi va o'qishga ham ta'sir qiladi;
- o'qitish va generatsiya aynan bir xil;
- softmax yo'q;
- logarifmik masofa jarimasi bor;
- alohida chiziqli xotirasiz, sof tanlov.

## Sinov rejasi (har bosqich Transformer bilan bir xil sharoitda)

1. **To'g'rilik testlari:**
   - parallel = tokenma-token = bo'laklab;
   - kauzallik;
   - xotira hajmi o'zgarmasligi;
   - naiv etalon bilan moslik.
2. **Sintetik:** MQAR (eslab qolish) va uzunroq masalaga umumlashtirish.
3. **Asosiy maqsad — savol-javob.**
   - Retsept `train.py` + `sft.py`, 3 ta seed.
   - O'lchovlar: `qa_eval` (jarima 1.15 va jarimasiz), `qa_diagnose`, val loss (256 va 1024).
   - Transformer'ning 3 seed natijasi tayyor.
4. **Tezlik va xotira:** uzun kontekstda generatsiya.
5. **Natija bo'yicha qaror.**
   - Agar ko'p seed'da Transformer'dan barqaror yuqori bo'lsa, suhbat modeli uchun standart bo'ladi.
   - Bo'lmasa, sababini o'lchab (K, muhimlik, masofa) takomillashtiramiz.
   - Natija qanday bo'lsa, shunday yoziladi.
