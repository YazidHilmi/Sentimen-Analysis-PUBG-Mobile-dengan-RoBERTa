# 🎮 PUBG Mobile Review Sentiment Analysis

Analisis sentimen terhadap ulasan pengguna aplikasi **PUBG Mobile** di Google Play Store, menggunakan pipeline NLP end-to-end mulai dari web scraping, preprocessing teks Bahasa Inggris, automatic labeling, penanganan data imbalance, hingga fine-tuning model **RoBERTa** untuk klasifikasi sentimen (Positive / Neutral / Negative).

## 📌 Ringkasan Proyek

Proyek ini bertujuan untuk memahami persepsi pengguna terhadap aplikasi PUBG Mobile melalui analisis ulasan yang di-scrape langsung dari Google Play Store. Karena dataset asli tidak memiliki label sentimen, proyek ini menerapkan pendekatan **automatic labeling** menggunakan model RoBERTa pretrained, dilanjutkan dengan fine-tuning model klasifikasi sentimen sendiri di atas data yang telah dilabeli.

## 🗂️ Struktur Proyek

```
├── Scrapping_PUBG.ipynb                     # Notebook scraping data ulasan dari Google Play Store
├── Analisis_Sentiment_PUBG_Mobile.ipynb      # Notebook preprocessing, labeling, dan modeling
└── README.md
```

## ⚙️ Tech Stack

| Komponen | Library / Tools |
|---|---|
| Scraping | `google-play-scraper` |
| Data Processing | Pandas, NumPy |
| Text Preprocessing | `emoji`, `contractions`, `nltk`, `Sastrawi` |
| Feature Extraction | TF-IDF (`scikit-learn`), FastText (`gensim`) |
| Imbalance Handling | Oversampling manual, SMOTE (`imbalanced-learn`) |
| Automatic Labeling | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) |
| Modeling | RoBERTa fine-tuning (`transformers`, PyTorch) |
| Visualisasi | Matplotlib, Seaborn, WordCloud |

## 🔄 Pipeline

### 1. Data Scraping (`Scrapping_PUBG.ipynb`)
- Mengambil **15.000 ulasan** aplikasi PUBG Mobile (`com.tencent.ig`) dari Google Play Store menggunakan `google-play-scraper`, diurutkan berdasarkan ulasan terbaru.
- Melakukan pengecekan awal (missing value, duplikat, info dataset).
- Menyeleksi fitur relevan (`reviewId`, `content`, `score`) dan menyimpannya sebagai `scrapping_pubg.csv`.

### 2. Data Understanding
- Eksplorasi dataset: pengecekan nilai kosong & duplikat.
- Visualisasi WordCloud dan kata-kata yang paling sering muncul.
- Analisis distribusi skor ulasan (1–5) serta kata-kata dominan pada rating rendah, netral, dan tinggi.

### 3. Data Preparation
- **Text cleaning**: lowercase, hapus newline/URL/username, ekstraksi teks dari hashtag, konversi emoji ke teks, ekspansi kontraksi, hapus karakter non-huruf, normalisasi karakter berulang.
- **Slangword normalization** menggunakan kamus slang custom.
- **Tokenizing**, **stopword removal**, dan **stemming** (Sastrawi/NLTK).

### 4. Automatic Labeling
- Karena data hasil scraping tidak memiliki label sentimen, pelabelan dilakukan otomatis menggunakan model pretrained **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`), menghasilkan label `positive`, `neutral`, dan `negative`.

### 5. Handling Data Imbalance
Dua pendekatan dieksplorasi untuk mengatasi ketidakseimbangan kelas:
- **Oversampling manual** dengan feature extraction **TF-IDF**.
- **SMOTE** dengan feature extraction **FastText** (word embedding custom-trained).

### 6. Modeling
- Fine-tuning model **`roberta-base`** untuk klasifikasi sentimen 3 kelas menggunakan `transformers.Trainer` dengan **weighted loss** (menangani class imbalance).
- Training selama 3 epoch dengan learning rate `2e-5`, batch size 8.

## 📊 Hasil Model

Hasil fine-tuning RoBERTa pada data validasi:

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score |
|---|---|---|---|---|
| 1 | 0.4721 | 0.5396 | 91.04% | 91.03% |
| 2 | 0.3114 | 0.4709 | 89.67% | 89.60% |
| 3 | 0.1900 | 0.4952 | **91.21%** | **91.21%** |

Model mencapai performa terbaik pada epoch ke-3 dengan **akurasi dan F1-Score ±91%**.

## 🚀 Cara Menjalankan

1. Clone repository dan buka notebook di Google Colab (disarankan menggunakan GPU runtime, misal T4).
2. Jalankan `Scrapping_PUBG.ipynb` terlebih dahulu untuk menghasilkan `scrapping_pubg.csv`.
3. Jalankan `Analisis_Sentiment_PUBG_Mobile.ipynb` secara berurutan dari Tahap 1 hingga Tahap 8 untuk melakukan preprocessing, labeling, dan training model.

```bash
pip install emoji gensim sastrawi gdown contractions transformers google-play-scraper imbalanced-learn
```

## 👤 Author

Yazid Hilmi — [GitHub](https://github.com/YazidHilmi) | [LinkedIn](https://www.linkedin.com/in/yazidhilmi/)
