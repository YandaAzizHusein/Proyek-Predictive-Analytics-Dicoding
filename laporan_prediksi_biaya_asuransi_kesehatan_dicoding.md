# Laporan Proyek Machine Learning - Prediksi Biaya Asuransi Kesehatan

## Domain Proyek

Kesehatan merupakan sektor dengan pengeluaran yang sangat besar secara global. Pada tahun 2021 saja, total belanja kesehatan dunia mencapai $9,8 triliun atau sekitar 10,3% dari PDB global. Namun, peningkatan belanja tersebut tidak selalu diiringi perbaikan signifikan pada outcome kesehatan masyarakat. Salah satu penyebab utama membengkaknya biaya kesehatan adalah faktor gaya hidup tidak sehat, seperti obesitas dan merokok, yang telah terbukti meningkatkan biaya medis tahunan secara signifikan.

Asuransi kesehatan menjadi krusial sebagai mekanisme pembagian risiko finansial akibat biaya medis yang tinggi. Perusahaan asuransi membutuhkan prediksi biaya klaim yang akurat untuk menetapkan premi yang adil dan mengelola risiko. Model prediksi yang baik dapat memperkecil selisih antara prediksi dan realisasi klaim, sekaligus menjadi dasar strategi bisnis yang proaktif.

**Mengapa masalah ini penting?**  
- Biaya kesehatan yang tidak terprediksi dapat menimbulkan kerugian besar bagi asuransi dan memberatkan peserta.
- Faktor risiko utama perlu diidentifikasi agar bisa dilakukan intervensi (misal: program berhenti merokok, edukasi BMI).

**Referensi:**  
- [CDC - Adult Obesity Facts](https://www.cdc.gov/obesity/adult-obesity-facts/index.html)
- [CDC - Smoking & Tobacco Use](https://www.cdc.gov/nccdphp/priorities/tobacco-use.html)
- [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## Business Understanding

### Problem Statements

1. **Bagaimana memprediksi biaya klaim asuransi kesehatan per individu dengan akurasi tinggi berdasarkan data profil pelanggan?**
2. **Faktor-faktor apa saja yang paling berpengaruh terhadap besarnya biaya asuransi kesehatan?**

### Goals

1. Membangun model regresi prediktif untuk memperkirakan biaya asuransi kesehatan per individu dengan tingkat error serendah mungkin.
2. Mengidentifikasi dan mengurutkan faktor risiko utama yang mempengaruhi besarnya biaya klaim.

### Solution Statements

- Menggunakan pendekatan machine learning regresi dengan dua model: **Linear Regression** (baseline, interpretasi mudah) dan **Random Forest Regressor** (untuk menangkap hubungan non-linear).
- Membandingkan kinerja kedua model menggunakan metrik RMSE, MAE, dan R² pada data uji.
- Mengidentifikasi feature importance dan membahas implikasi bisnis dari insight yang diperoleh.

---

## Data Understanding

**Dataset:**  
- [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Jumlah sampel: 1338 entri (1 entri = 1 orang tertanggung asuransi)
- Jumlah fitur: 6 fitur prediktor + 1 target (charges)

**Fitur pada dataset:**
- `age`: Usia (tahun, numerik)
- `sex`: Jenis kelamin (`male`, `female`, kategorikal)
- `bmi`: Body Mass Index (kg/m², numerik)
- `children`: Jumlah anak tanggungan (numerik)
- `smoker`: Status perokok (`yes`/`no`, kategorikal)
- `region`: Wilayah tinggal (`northeast`, `northwest`, `southeast`, `southwest`, kategorikal)
- `charges`: Biaya asuransi kesehatan individu (USD, target)

**Kondisi Data:**
- Tidak ada missing value.
- Tidak ada duplikasi.
- Rentang usia: 18–64 tahun, BMI: 15,96–53,13, children: 0–5, proporsi perokok: ~20%.

### Exploratory Data Analysis (EDA)

1. **Distribusi Charges**  
   Distribusi charges right-skewed dengan banyak outlier (biaya sangat tinggi) pada kelompok risiko tinggi.  
   ![Distribusi Charges](#)  
   *Mayoritas biaya < $20.000/tahun, sedikit kasus ekstrem di atas $50.000.*

2. **Korelasi Fitur Numerik**
   - `age` memiliki korelasi positif sedang terhadap `charges` (r ≈ 0,30)
   - `bmi` korelasi positif lemah (r ≈ 0,20)
   - `children` korelasi hampir nol (r ≈ 0,07)

   ![Heatmap Korelasi](#)

3. **Analisis Fitur Kategorikal**
   - **Smoker:** Rata-rata biaya perokok hampir 4x non-perokok ($32.050 vs $8.434)
   - **Sex:** Perbedaan rata-rata kecil (pria sedikit lebih tinggi)
   - **Region:** Perbedaan kecil, southeast tertinggi

   ![Boxplot Charges vs Smoker](#) ![Boxplot Charges vs Sex](#) ![Boxplot Charges vs Region](#)

**Insight Utama:**  
Status perokok adalah *prediktor terkuat* biaya asuransi, diikuti oleh usia dan BMI. Anak, jenis kelamin, dan region berpengaruh sangat kecil.

---

## Data Preparation

1. **Encoding Fitur Kategorikal**  
   - *One-Hot Encoding* pada `sex` (drop: female), `smoker` (drop: no), dan `region` (drop: northeast).
   - Hasil: Semua fitur jadi numerik, siap untuk ML.

2. **Train-Test Split**  
   - Data dibagi: 80% train, 20% test, random_state=42.

3. **Handling Outliers**  
   - Outlier tidak dihapus (penting untuk prediksi risiko tinggi).
   - Tidak melakukan scaling (algoritma tree-based tidak memerlukan, rentang numerik masih homogen).

4. **Fitur Interaksi**  
   - Tidak ditambahkan manual, karena Random Forest dapat menangkap interaksi secara otomatis.

**Alasan tiap tahapan:**  
- Encoding diperlukan agar data kategorikal bisa digunakan algoritma ML.
- Train-test split untuk validasi objektif kinerja model.
- Outlier dipertahankan agar model belajar kasus ekstrim yang krusial di dunia asuransi.

---

## Modeling

### Model yang digunakan

1. **Linear Regression**  
   - Baseline, interpretasi mudah, cepat.
   - Kelebihan: interpretatif, sederhana, mudah dikomunikasikan ke bisnis.
   - Kekurangan: hanya menangkap hubungan linear, sensitif outlier.

2. **Random Forest Regressor**  
   - Menangkap hubungan non-linear & interaksi, robust pada outlier.
   - Kelebihan: akurasi lebih tinggi untuk data non-linear, otomatis feature selection.
   - Kekurangan: lebih kompleks, interpretasi sulit.

### Parameter dan Proses

- Linear Regression: default (tanpa regularisasi)
- Random Forest: n_estimators=100, random_state=42, parameter lain default.

### **Improvement**
- Random Forest dipilih untuk improvement karena lebih robust terhadap outlier & non-linearitas.

### **Pemilihan Model Terbaik**
- Model terbaik dipilih berdasarkan error terendah (RMSE, MAE) dan skor R² tertinggi pada data uji.

---

## Evaluation

### Metrik Evaluasi

- **MAE (Mean Absolute Error):** rerata absolut selisih prediksi vs aktual.
- **RMSE (Root Mean Squared Error):** penalti lebih besar untuk error besar/outlier.
- **R² (R-Squared):** proporsi variansi target yang dijelaskan model.

**Formula:**
- \( MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| \)
- \( RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} \)
- \( R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} \)

### **Hasil Evaluasi (Contoh Output)**

| Model              | MAE     | RMSE    | R²    |
|--------------------|---------|---------|-------|
| Linear Regression  | 4,002   | 5,775   | 0.74  |
| Random Forest      | 2,409   | 3,740   | 0.86  |

*(Silakan sesuaikan dengan hasil aktual notebook Anda)*

- **Random Forest** menunjukkan MAE & RMSE lebih rendah, serta R² lebih tinggi, sehingga dipilih sebagai model terbaik.
- Feature importance (Random Forest): `smoker_yes` > `age` > `bmi` > fitur lainnya.

### **Insight dan Implikasi Bisnis**

- **Merokok** adalah *driver* utama biaya asuransi—premi diferensial bagi perokok sangat dianjurkan.
- **BMI dan usia** juga berpengaruh signifikan; program wellness dapat difokuskan pada kelompok ini.
- Region dan gender hanya sedikit berpengaruh dalam dataset ini.

---

## Kesimpulan

Model prediksi biaya asuransi kesehatan berhasil dibangun menggunakan Linear Regression (baseline) dan Random Forest (model terbaik). Model Random Forest memberikan prediksi paling akurat dan berhasil mengidentifikasi *faktor risiko utama* yang mempengaruhi besarnya biaya klaim, yaitu status perokok, usia, dan BMI.

Model ini dapat diintegrasikan dalam sistem underwriting asuransi untuk penetapan premi dan program intervensi kesehatan yang lebih tepat sasaran.

---

## Referensi

- Kaggle: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- CDC: [Adult Obesity Facts](https://www.cdc.gov/obesity/adult-obesity-facts/index.html)
- CDC: [Smoking & Tobacco Use](https://www.cdc.gov/nccdphp/priorities/tobacco-use.html)
- Dataquest: [Predicting Insurance Costs with Linear Regression](https://www.dataquest.io/blog/predicting-insurance-costs-with-linear-regression/)
- Dicoding Submission Guidance: [contoh laporan mlt](https://github.com/dicodingacademy/contoh-laporan-mlt)

---

*Silakan tambahkan gambar hasil visualisasi/plot dari notebook Anda ke dalam laporan ini (paste link gambar atau lampirkan secara lokal saat submit). Semua insight, narasi, dan formula telah disesuaikan dengan rubrik/kriteria bintang 5 Dicoding. Laporan siap di-export ke format .md atau .txt dan dikirim bersama notebook & file .py Anda!*

