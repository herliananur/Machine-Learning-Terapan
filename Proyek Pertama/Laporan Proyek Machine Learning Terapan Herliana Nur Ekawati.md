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
Berikut merupakan EDA-Univariate Analysis:

- Grafik sex, grafik dibawah ini menunjukkan bahwa laki-laki mendominasi.

![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/grafik%20sex.png)

- Grafik smoker, grafik dibawah ini menunjukkan bahwa pasien kebanyakan tidak merokok.
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/grafik%20smoker.png)

- Grafik region, grafik dibawah ini menujukkan bahwa pasien kebanyakan berasal dari southeast.
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/grafik%20region.png)


### EDA-Multivariate
Berikut merupakan EDA-Multivariate Analysis:
- Grafik rata-rata expenses relatif terhadap sex, grafik dibawah ini menunjukkan bahwa pasien laki-laki mendominasi
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/multivariate-sex.png)

- Grafik rata-rata expenses relatif terhadap smoker, grafik dibawah ini menunjukkan bahwa biaya yang dikeluarkan untuk berobat lebih banyak jika pasien merokok
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/multivariate-smoker.png)

- Grafik rata-rata expenses relatif terhadap region, grafik dibawah ini menunjukkan bahwa pasien terbanyak berasal dari southeast
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/multivariate-region.png)

Dari ketiga grafik diatas, dapat disimpulkan bahwa biaya pengobatan di rumah sakit akan lebih besar untuk perokok dibandingkan orang yang tidak merokok.

#### Korelasi Matriks
Berdasarkan matriks dibawah, Numerical Features memiliki korelasi yang rendah terhadap expenses.
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/korelasi%20matriks%20fitur%20numerik.png)


## Data Preparation
### Encoding
Beberapa cara untuk melakukan Encoding sebagai berikut:
1. Melakukan one-hot-encoding pada categorical features menggunakan get_dummies.
```sh
from sklearn.preprocessing import OneHotEncoder
hospital = pd.concat([hospital, pd.get_dummies(hospital['sex'], prefix='sex')], axis=1)
hospital = pd.concat([hospital, pd.get_dummies(hospital['smoker'], prefix='smoker')], axis=1)
hospital = pd.concat([hospital, pd.get_dummies(hospital['region'], prefix='region')], axis=1)
```
Korelasi matriks untuk seluruh fitur
Berdasarkan matriks dibawah, maka dapat disimpulkan bahwa smoker berkorelasi kuat terhadap expense

![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/korelasi%20matriks%20seluruh%20fitur.png)


2. Label Data
Kita akan mengahpus fitur sex, smoker, dan region terlebih dahulu karena telah melalui proses encoding
```sh
hospital.drop(['sex', 'smoker', 'region'], axis=1, inplace=True)
hospital
```

Kemudian membuat dataframe x yang menampung variabel
```sh
x = hospital.drop(['expenses'], axis=1)
x
```

Selanjutnya, buat dataframe y untuk menampung variabel
```sh
y = hospital['expenses']
y
```

3. Train-Test-Split
Membagi data sampel menjadi data train dan data test dengan ukuran 80% data train dan 20% data test.
```sh
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=200)
```
Total dari panjang x, data train, dan data test sebagai berikut:
```sh
Total # of sample in whole dataset: 1338
Total # of sample in train dataset: 1070
Total # of sample in test dataset: 268
```

4. Standarisasi
