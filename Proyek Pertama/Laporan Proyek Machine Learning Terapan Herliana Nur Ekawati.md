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
Melakukan standarisasi menggunakan StandardScaler pada data. Kemudian mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.
```sh
from sklearn.preprocessing import StandardScaler

numerical_features = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
```

Mengecek nilai mean dan standar deviasi pada setelah proses standarisasi
```sh
x_train[numerical_features].describe().round(4)
```

5. Modeling
Model-model yang digunakan padal proyek ini adalah:
- **KNN** adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.
```sh
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
```

- **Random forest** merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Ensemble merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. 
```sh
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(x_train, y_train)
```

- **Algoritma Boosting** bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. 
```sh
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(x_train, y_train)
```

** | **train** | **test**|
:-----:|:-----:|:-----:|
**KNN** | 30064.576543 | 39220.122427|
**RF** | 3791.412142 | 22034.478805 |
**Boosting** | 21482.520677 | 22608.716112|

Karena MSE RF lebih rendah daripada KNN dan Boosting, maka akan menggunakan model RF

6. Evaluasi
Matriks valuasi yang akan digunakan adalah MSE (Mean Squared Error) dan R2 Square
MSE (Mean Squared Error) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut.
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/mse.png)

Berdasarkan hasil evauasi menggunakan matriks MSE, dapat disimpulkan bahwa model Random Forest memiliki MSE yang lebih kecil dibanding KNN dan Boosting.
** | **train** | **test**|
:-----:|:-----:|:-----:|
**KNN** | 30064.576543 | 39220.122427|
**RF** | 3791.412142 | 22034.478805 |
**Boosting** | 21482.520677 | 22608.716112|

![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/grafik%20mse.png)

R2 squared merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen.
![This is an image](https://github.com/herliananur/Machine-Learning-Terapan/blob/main/Proyek%20Pertama/Gambar/r2.png)

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
