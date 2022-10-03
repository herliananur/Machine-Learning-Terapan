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
Proyek ini akan menggunakan 2 algoritma Machine Learning yaitu:
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
print('Banyak buku: ', len(book.book_title.unique()))
print('Banyak penulis: ', len(book.book_author.unique()))
print('Judul Buku: ', book.book_title.unique())
print('Nama Penulis: ', book.book_author.unique())
```
Output
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
print('Angka rating: ', rating.book_rating.unique())
print('Banyak user: ', len(rating.userid.unique()))
```
Output
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

- Mapping
```sh
rating['user'] = rating['userid'].map(user_to_user_encoded)
rating['book'] = rating['ISBN'].map(book_to_book_encoded)
```
Memetakan userid dan ISBN ke dataframe yang berhubungan.
- Cek data, untuk mengecek jumlah dari user, buku, dan mengubah nilai rating menjadi float.
- Melatih data, dengan mengacak data terlebih dahulu
```sh
rating = rating.sample(frac=1, random_state=42)
rating
```
Membagi data train dan test dengan komposisi 82%:18%.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
