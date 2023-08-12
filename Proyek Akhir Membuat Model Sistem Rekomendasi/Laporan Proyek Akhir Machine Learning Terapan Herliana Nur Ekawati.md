# Laporan Proyek Machine Learning - Herliana Nur Ekawati

## Project Overview

Buku merupakan jendela dunia. Tanpa adanya buku, kita tidak akan mengerti apa-apa. Buku memiliki banyak jenisnya, yaitu Novel, Ensiklopedia, Biografi, Sains, dan lain-lain. Buku sebagai jembatan informasi dari penulis ke pembaca. Manfaat membaca buku yaitu menambah wawasan, menambah ilmu, dan sebagai pedoman hidup. Proyek ini membuat sistem rekomendasi buku sesuai dengan apa buku yang telah dibaca sebelumnya.

## Business Understanding

Setiap individu pasti memiliki minat baca yang berbeda. Contohnya perbedaan ketertarikan. Ada yang menyukai buku berdasarkan penulisnya, genrenya, maupun covernya. Proyek ini merupakan sistem rekomendasi berdasarkan ketertarikan pembaca.

### Problem Statements

Berdasarkan permasalahan di atas, maka dapat disimpulkan masalah yang ada yaitu:
- Bagaimana cara untuk bisa menemukan rekomendasi buku berdasarkan buku yang telah dibaca?
- Bagaimana cara untuk membuat sistem rekomendasi buku berdasarkan rating buku?

### Goals

Berdasarkan permasalahan di atas, tujuan dari proyek ini yaitu:
- Mengetahui cara untuk bisa menemukan rekomendasi buku berdasarkan buku yang telah dibaca
- Mengetahui cara untuk membuat sistem rekomendasi buku berdasarkan rating buku

### Solution statements
Proyek ini akan menggunakan 2 algoritma *Machine Learning* yaitu:
1. **Content Based Filtering** adalah bagian untuk merekomendasikan item yang mirip dengan masa lalu. Contohnya jika membaca buku dengan genre horor, maka algoritma ini akan merekomendasikan buku genre horor dengan judul buku yang berbeda.
2. **Collaborative Filtering** bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten.

## Data Understanding
Dataset yang digunakan untuk proyek ini diperoleh dati situs kaggle yang dapat diunduk melalui [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Data yang akan digunakan yaitu Books.csv dan Ratings.csv. 

Adapun penjelasan detail dari sampel data **Books.csv** sebagai berikut:
- ISBN: Nomor dari sebuah buku
- Book-Title: Judul buku
- Book-Author: Penulis buku
- Year-of-Publication: Tahun buku diterbitkan
- Publisher: Penerbit buku
- Image-URL-S: Tautan untuk gambar sampul buku berukuran kecil
- Image-URL-M: Tautan untuk gambar sampul buku berukuran sedang
- Image-URL-L: Tautan untuk gambar sampul buku berukuran besar

Adapun penjelasan detail dari sampel data **Ratings.csv** sebagai berikut:
- User-ID: ID Pengguna
- ISBN: Nomor dari sebuah buku
- Book-Rating: Rating(penilaian) buku dari pengguna

### Univariate Exploratory Data Analysis
**Book**

Tabel 1. Info buku

``` sh
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 271360 entries, 0 to 271359
Data columns (total 8 columns):
```
**#** | **Column** | **Non-Null Count** | **Dtype** |
:-----:|:-----:|:-----:| :-----:|
0 | ISBN | 39271360 non-null | object |
1 | Book-Title | 271360 non-null | object |
2 | Book-Author | 271359 non-null | object |
3 | Year-Of-Publication | 271360 non-null | object | 
4 | Publisher | 271358 non-null | object | 
5 | Image-URL-S | 271360 non-null | object |
6 | Image-URL-M | 271360 non-null | object |
7 | Image-URL-L | 271357 non-null | object |
```sh
dtypes: object(8)
memory usage: 16.6+ MB
```
Berdasarkan tabel diatas, book memiliki 271360 data entri. Terdapat 8 variabel berupa ISBN, Book-Title, Book-Author, Year-of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L.

```sh
Banyak buku:  242135
Banyak penulis:  102024
Judul Buku:  ['Classical Mythology' 'Clara Callan' 'Decision in Normandy' ...
 'Lily Dale : The True Story of the Town that Talks to the Dead'
 "Republic (World's Classics)"
 "A Guided Tour of Rene Descartes' Meditations on First Philosophy with Complete Translations of the Meditations by Ronald Rubin"]
Nama Penulis:  ['Mark P. O. Morford' 'Richard Bruce Wright' "Carlo D'Este" ...
 'David Biggs' 'Teri Sloat' 'Christopher  Biffle']
```
Terdapat 242135 buku dengan tahun terbit dan penulis yang berbeda.

**Rating**

Tabel 2. Info Rating

``` sh
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1149780 entries, 0 to 1149779
Data columns (total 3 columns):
```
**#** | **Column** | **Non-Null Count** | **Dtype** |
:-----:|:-----:|:-----:| :-----:|
0 | User-ID | 1149780 non-null | int64 
1 | ISBN | 1149780 non-null | object
2 | Book-Rating | 1149780 non-null | int64 
```sh
dtypes: int64(2), object(1)
memory usage: 26.3+ MB
```
Berdasarkan tabel diatas, book memiliki 1149780 data entri. Terdapat 3 variabel berupa User-ID, ISBN, dan Book-Rating.

```sh
Angka rating:  [ 0  5  3  6  8  7 10  9  4  1  2]
Banyak user:  105283
```
Terdapat 105283 data userid yang berbeda dengan rating dari 0 sampai 10.

## Data Preparation
Proyek ini menggunakan 5000 data books dan 1000 data ratings.
```sh
book = book[:5000]
rating = rating[:1000]
```

**Content Based Filtering** menggunakan 4 data preparation, antara lain:
- Drop kolom Na, untuk menghapus kolom yang berisi nilai Na/Null
- Drop baris duplikat, untuk menghapus baris duplikat supaya tidak ada tumpang tindih
- Mengubah dataframe book menjadi list, mengonversi beberapa fitur dataframe book menjadi list.
- Membuat dictionary, menambahkan key value untuk memanggil fitur yang telah diubah menjadi list

**Collaborative Filtering** menggunakan 4 data preparation, antara lain:
- Melakukan Encoding pada kolom user_id dan ISBN, outputnya sebagai berikut:
```sh
encoded userID :  {276725: 0, 276726: 1, 276727: 2, 276729: 3, 276733: 4, . . .
encoded angka ke userID:  {0: 276725, 1: 276726, 2: 276727, 3: 276729, 4: . . .
```
```sh
encoded ISBN :  {'034545104X': 0, '0155061224': 1, '0446520802': 2, '052165615X': 3, '0521795028': 4 . . .
encoded angka ke ISBN:  {0: '034545104X', 1: '0155061224', 2: '0446520802', 3: '052165615X', 4: . . .
```

- Mapping, Memetakan userid dan ISBN ke dataframe yang berhubungan.
- Cek data, untuk mengecek jumlah dari user, buku, dan mengubah nilai rating menjadi float.
- Melatih data, dengan mengacak data terlebih dahulu. Membagi data train dan test dengan komposisi 82%:18%.

## Modeling and Results

### Content Based Filtering
- Modelling menggunakan fungsi TF-IDF Vectorizer. Menggunakan satu data dari book_author untuk mendapatkan rekomendasi berdasarkan penulis.
- Lakukan fit dan transformasi kedalam bentuk matriks dengan hasil
```sh
(5000, 3538)
```
- Menghasilkan vektor tf-idf dalam bentuk matriks
```sh
matrix([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])
```

- Menghitung derajat kesamaan menggunakan Cosine  Similarity
```sh
array([[1., 0., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.],
       [0., 0., 0., ..., 0., 0., 1.]])
```

- Melihat matriks kesamaan dengan sampel buku 10 dan 20
```sh
Shape: (5000, 5000)
```

- Menguji dengan judul buku Classical Mythology dengan penulis Mark P.O. Morford

Tabel 3. Tampilan keterangan buku Classical Mythology

**#** | **book_title** | **book_author** | **book_year** | **book_ISBN** |
:-----:|:-----:|:-----:| :-----:| :-----:|
0 | Classical Mythology | Mark P.O. Monford | 2002 | 0195153448 |

- Mendapatkan rekomendasi yang mirip dengan buku Classical Mythology berdasarkan penulisnya

Tabel 4. Rekomendasi yang mirip

**#** | **book_title** | **book_author** | 
:-----:|:-----:|:-----:| 
0	| Fishboy: A Ghost's Story	| Mark Richard |
1	| The Diaries of Adam and Eve (Literary Classics) |	Mark Twain |
2	| Adventures of Huckleberry Finn (Signet Classic... |	Mark Twain |
3	| A Connecticut Yankee in King Arthur's Court (D...	| Mark Twain |
4	| Adventures of Huckleberry Finn	| Mark Twain |

Berdasarkan hasil rekomendasi, sistem mengambil kata kunci Mark untuk merekomendasikan kepada user.

### Collaborative Filtering
- Melakukan proses encoding terhadap user id dan buku.
- Latih dengen menggunakan kelas RecomenderNet dan melakukan proses embedding
- Mengkompile menggunakan BinaryCrossentrophy() untuk menghitung loss function, Adam sebagai optimizer, dan RMSE sebagai metrics evaluation.

Hasil rekomendasi buku sebagai berikut:

Tabel 5. Hasil rekomendasi buku

**Top 10 book recommendation** | 
:-----:|
Heart of Darkness (Wordsworth Collection) : Joseph Conrad |
Alice's Adventures in Wonderland and Through the Looking Glass : Lewis Carroll | 
The Lovely Bones: A Novel : Alice Sebold |
The Da Vinci Code : Dan Brown |
Politically Correct Bedtime Stories: Modern Tales for Our Life and Times : James Finn Garner |
Quidditch Through the Ages : J. K. Rowling | 
No Pasaran! El Videojuego : Chrstine Lehmann |
The Restaurant at the End of the Universe (Hitchhiker's Trilogy (Paperback)) : Douglas Adams |
Q : Luther Blissett |
La Sombra del Viento : Carlos Ruiz Zafon |


## Evaluation
Pada **Content Based Filtering** ini menggunakan metriks Precision, yang mana merupakan metriks yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat.
Pada hasil rekomendasi Content Based Filtering, dari 5 buku yang direkomendasikan, tidak ada satu pun rekomendasi memiliki penulis yang sesuai dengan buku yang sudah dibaca oleh user. Sehingga presisi nya adalah 0%
```sh
Accuracy = real_author/5*100
print("Accuracy of the model is {}%".format(Accuracy))
```
Output
```sh
Accuracy of the model is 0.0%
```

Pada **Collaborative Filtering** ini menggunakan metriks RMSE, yang mana besar kesalahan hasil prediksi, dimana semakin kecil (mendekati 0), maka  nilai RMSE hasil prediksi akan semakin akurat.

Berikut hasil nilai RMSE pada Collaborative Filtering

![download](https://user-images.githubusercontent.com/111114060/193612621-a1eb3116-6426-4179-aa3a-9d7f0fbf2bcb.png)

Gambar 1. Nilai RMSE pada Collaborative Filtering

Kesimpulan yang bisa diambil dari plot nilai RMSE yaitu nilai train dan test mengalami penurunan, yang berarti model yang telah dibuat cukup akurat.
