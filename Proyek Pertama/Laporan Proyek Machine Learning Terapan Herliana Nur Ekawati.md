# Laporan Proyek Machine Learning - Herliana Nur Ekawati
## Domain Proyek
Setiap daerah pasti memiliki setidaknya satu pasien di rumah sakit. Namun, beberapa orang tidak pergi ke rumah sakit bahkan jika mereka sakit. 
Alasannya sangat beragam, ada yang tidak percaya dengan dunia medis, ada yang lebih memilih istirahat di rumah, dan ada juga yang takut dengan biaya pengobatan yang tinggi. Untuk itu, diperlukan penelitian untuk memprediksi biaya perawatan di rumah sakit. Proyek ini bertujuan untuk memprediksi nilai biaya yang dikeluarkan oleh masyarakat yang ingin berobat ke rumah sakit.

## Business Understanding
Orang akan berpikir mereka perlu pergi ke rumah sakit ketika penyakitnya sudah mulai sangat menyakitkan. 
Dengan meremehkan apa yang dianggap normal, penyakitnya menjadi lebih besar. Begitu juga dengan tagihan rumah sakit. Jika penyakitnya masih umum, biayanya lebih murah dibandingkan yang sudah parah.

### Problem Statement
Berdasarkan permasalahan di atas, maka dapat disimpulkan masalah yang ada yaitu:
- Bagaimana cara memprediksi biaya untuk berobat di rumah sakit?

### Goal
Berdasarkan permasalahan di atas, tujuan dari proyek ini yaitu:
- Mengetahui cara memprediksi biaya untuk berobat di rumah sakit

### Solution statements
- Melakukan proses EDA 
- Membuat Model Machine Learning dengan menggunakan metode:
  1. KNN
  2. RF
  3. Boosting Algorithm

## Data Understanding

Dataset yang digunakan untuk proyek ini diperoleh dari situs kaggle yang dapat diunduh melalui [Kaggle](https://www.kaggle.com/datasets/rajgupta2019/medical-insurance-dataset). 
Dataset ini memiliki 1338 data dan 7 sampel data. Sampel data akan dibagi menjadi dua, Numerical Features (age, bmi, children, dan expenses) dan Categorical Features (sex, smoker, dan region).
Adapun penjelasan detail dari sampel data sebagai berikut:
- age: Umur pasien
- bmi: _Body Mass Index_ 
- children: Jumlah anak yang ditanggung asuransi
- expenses: Biaya medis individu
- sex: Jenis kelamin pasien
- smoker: Perokok atau bukan
- region: Tempat asal

### EDA-Univariate
