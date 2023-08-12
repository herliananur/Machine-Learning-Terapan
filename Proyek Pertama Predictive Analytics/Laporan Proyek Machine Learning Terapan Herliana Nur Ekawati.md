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
Dataset ini memiliki 1338 data dan 7 sampel data. Sampel data akan dibagi menjadi dua, *Numerical Features* (age, bmi, children, dan expenses) dan *Categorical Features* (sex, smoker, dan region).
Adapun penjelasan detail dari sampel data sebagai berikut:
- age: Umur pasien
- bmi: _Body Mass Index_ 
- children: Jumlah anak yang ditanggung asuransi
- expenses: Biaya medis individu
- sex: Jenis kelamin pasien
- smoker: Perokok atau bukan
- region: Tempat asal

### EDA-Univariate
Berikut merupakan *EDA-Univariate Analysis*:

- Grafik sex, grafik dibawah ini menunjukkan bahwa laki-laki mendominasi.

![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 1. Grafik Sex

- Grafik smoker, grafik dibawah ini menunjukkan bahwa pasien kebanyakan tidak merokok.
![grafik smoker](https://user-images.githubusercontent.com/111114060/192172102-91b26b8f-05d2-436c-ae59-839721f27ed0.png)
Gambar 2. Grafik Smoker

- Grafik region, grafik dibawah ini menujukkan bahwa pasien kebanyakan berasal dari southeast.
![grafik region](https://user-images.githubusercontent.com/111114060/192172126-f032a6ff-2ebe-4af8-a65e-6c31b6f69607.png)
Gambar 3. Grafik Region

### EDA-Multivariate
Berikut merupakan *EDA-Multivariate Analysis*:
- Grafik rata-rata expenses relatif terhadap sex, grafik dibawah ini menunjukkan bahwa pasien laki-laki mendominasi
![multivariate-sex](https://user-images.githubusercontent.com/111114060/192172153-717b72a1-9b55-461b-9be4-f342437d788e.png)
Gambar 4. Grafik Rata-rata Expenses Relatif Terhadap Sex

- Grafik rata-rata expenses relatif terhadap smoker, grafik dibawah ini menunjukkan bahwa biaya yang dikeluarkan untuk berobat lebih banyak jika pasien merokok
![multivariate-smoker](https://user-images.githubusercontent.com/111114060/192172178-b6709729-1460-42bf-8179-5e3791fad005.png)
Gambar 5. Grafik Rata-rata Expenses Relatif Terhadap Smoker

- Grafik rata-rata expenses relatif terhadap region, grafik dibawah ini menunjukkan bahwa pasien terbanyak berasal dari southeast
![multivariate-region](https://user-images.githubusercontent.com/111114060/192172192-bc1d1470-b363-4a57-b85a-40dd93908d0f.png)
Gambar 6. Grafik Rata-rata Expenses Relatif Terhadap Region

Dari ketiga grafik diatas, dapat disimpulkan bahwa biaya pengobatan di rumah sakit akan lebih besar untuk perokok dibandingkan orang yang tidak merokok.

#### Korelasi Matriks
Berdasarkan matriks dibawah, *Numerical Features* memiliki korelasi yang rendah terhadap expenses.
![korelasi matriks fitur numerik](https://user-images.githubusercontent.com/111114060/192172220-79a78f73-7d35-4fd3-ae44-5348429cf4ed.png)
Gambar 7. Korelasi Matriks *Numerical Features*

## Data Preparation
### Encoding
Beberapa cara untuk melakukan Encoding sebagai berikut:
1. Melakukan one-hot-encoding pada *categorical features* menggunakan get_dummies.
```sh
hospital = pd.concat([hospital, pd.get_dummies(hospital['sex'], prefix='sex')], axis=1)
hospital = pd.concat([hospital, pd.get_dummies(hospital['smoker'], prefix='smoker')], axis=1)
hospital = pd.concat([hospital, pd.get_dummies(hospital['region'], prefix='region')], axis=1)
```
Korelasi matriks untuk seluruh fitur
Berdasarkan matriks dibawah, maka dapat disimpulkan bahwa smoker berkorelasi kuat terhadap expense
![korelasi matriks seluruh fitur](https://user-images.githubusercontent.com/111114060/192172625-e9626678-5d8d-4063-9ca6-9dff650c8420.png)
Gambar 8. Korelasi Matriks untuk Seluruh Fitur

2. Label Data
Kita akan menghapus fitur sex, smoker, dan region terlebih dahulu karena telah melalui proses encoding menggunakan fungsi drop.

Kemudian membuat dataframe x yang menampung variabel
```sh
x = hospital.drop(['expenses'], axis=1)
x
```

Selanjutnya, buat dataframe y untuk menampung variabel expenses

3. Train-Test-Split
Membagi data sampel menjadi data train dan data test dengan ukuran 80% data train dan 20% data test.
```sh
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=200)
```
Total dari panjang x, data train, dan data test sebagai berikut:
```sh
Total # of sample in whole dataset: 1338
Total # of sample in train dataset: 1070
Total # of sample in test dataset: 268
```

4. Standarisasi
Melakukan standarisasi menggunakan StandardScaler pada data. Kemudian mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.
```sh
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
```

Mengecek nilai mean dan standar deviasi pada setelah proses standarisasi
```sh
x_train[numerical_features].describe().round(4)
```

## Modeling
Model-model yang digunakan padal proyek ini adalah:
- **KNN** adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.
```sh
knn = KNeighborsRegressor(n_neighbors=10)
```
Di sini menggunakan parameter n_neighbors sebanyak 10, yang mana parameter tersebut akan mengambil 10 tetangga dengan jarak dekat. Selanjutnya parameter tersebut akan mengambil data pada 10 sampel tetangga untuk dimasukkan menjadi data baru.

- **Random forest** merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Ensemble merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. 
```sh
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
```
Di sini menggunakan parameter n_estimator sebanyak 50 yang mana parameter tersebut akan membuat sebanyak 50 cabang pohon, dengan kedalaman maksimal 16.


- **Algoritma Boosting** bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. 
```sh
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
```
Di sini menggunakan parameter learning_rate 0.05 yang mana parameter tersebut akan di latih sebanyak 0.05 dengan random_state sebanyak 55.

Tabel 1. Nilai *Train* dan *Test* dari KNN, RF, dan Boosting
** | **train** | **test**|
:-----:|:-----:|:-----:|
**KNN** | 30064.576543 | 39220.122427|
**RF** | 3791.412142 | 22034.478805 |
**Boosting** | 21482.520677 | 22608.716112|

Karena MSE RF lebih rendah daripada KNN dan Boosting, maka akan menggunakan model RF

## Evaluasi
Matriks valuasi yang akan digunakan adalah MSE (Mean Squared Error) dan R2 Square
MSE (Mean Squared Error) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut.
![mse](https://user-images.githubusercontent.com/111114060/192172988-a8427c11-74c6-4911-9fd1-4c2f6956bb6c.png)
Gambar 9. MSE

Berdasarkan hasil evauasi menggunakan matriks MSE, dapat disimpulkan bahwa model Random Forest memiliki MSE yang lebih kecil dibanding KNN dan Boosting.

Tabel 2. Nilai *Train* dan *Test* dari KNN, RF, dan Boosting
** | **train** | **test**|
:-----:|:-----:|:-----:|
**KNN** | 30064.576543 | 39220.122427|
**RF** | 3791.412142 | 22034.478805 |
**Boosting** | 21482.520677 | 22608.716112|

![grafik mse](https://user-images.githubusercontent.com/111114060/192173026-d3d13942-7e80-4f9c-bdb0-189035913024.png)
Gambar 10. Grafik MSE

R2 squared merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen.

![r2](https://user-images.githubusercontent.com/111114060/192173053-4684901e-bb96-4d94-9d50-5427ba70330e.png)
Gambar 11. *R2 Squared*

Tabel 3. Nilai prediksi
** | y_true |	prediksi_KNN | prediksi_RF | prediksi_Boosting |
:-----:|:-----:|:-----:|:-----:|:-----:|
992 | 10118.42 | 12328.2 | 10616.4 | 13157.5 |

Dapat dilihat pada tabel bahwa prediksi menggunakan Random Forest lebih mendekati nilai y.

Nilai R2 Square dari Random Forest lebih besar dari KNN dan Boosting, maka Random Forest sangat efektif dalam memprediksi nilai.
```sh
R2 score KNN :  0.7494154894917251
R2 score RF :  0.8592176988797552
R2 score Boosting :  0.7494154894917251
```
