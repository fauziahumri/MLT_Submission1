# Laporan Proyek 1 Machine Learning Terapan - Fauziah Umri
## Domain Proyek
Anggur merupakan salah satu buah yang banyak di konsumsi dan cukup populer di berbagai wilayah. Buah anggur biasanya di konsumsi secara langsung dan juga diolah menjadi suatu produk seperti makanan dan minuman yang difermentasi yang akan menjadi minuman beralkohol seperti _wine_. Biasanya jangka waktu yang dibutuhkan untuk fermentasi anggur untuk menjadi _wine_ bervariasi, ada yang membutuhkan waktu singkat dan adapula yang membutuhkan waktu yang lama.
Beberapa jenis-jenis _wine_ yaitu _Rose Wine, Sweet Wine, Sparkling Wine, Red Wine, White Wine,_ dan _Fortified Wine_. 

Anggur memiliki berbagai karakteristik seperti kepadatan, nilai pH, alkohol dan asam lainnya. Dalam perkembanganya _wine_ semakin bermacam varianya. Hal itu pula yang membuat _wine_ di bagi berdasarkan kualitasnya untuk memnetukan harga jual di pasaran. Kulaitas pada _wine_ dipengaruhi oleh bebrapa faktor, contohnya komposisi yang terdapat di dalamnya. Untuk menentukan kualitas _wine_ tentu harus ada ahli yang bertugas untuk mencicipi sampel dari minuman anggur tersebut [1].

Dari permasalah diatas yang akan dilakukan terhadap dataset _Red wine Quality_ adalah melakukan pengujian dengan menggunakan _Random Forest Classsifier_ yang akan dilakukan dengan _tools_ Python.

## Pendefinisian Bisnis

Suatu perusahaan ingin meningkatkan pengetahuan tentang kualitas _wine_ untuk memenuhi kebutuhan dan keinginan konsumen dan memberi petunjuk tentang kemungkinan, dan kesediaan konsumen untuk membeli anggur dengan campuran bahan-bahan tertentu serta memberikan keunggulan bagi produsen dibandingkan pesaing lainnya. Untuk meningkatkan kualitas teresebut perusahaan menggunakan teknologi machine learning untuk memprediksi kualitas wine tersebut. sehingga prediksi dilakukan dengan metode klasifikasi jenis _wine_ mulai dari kualitas yang rendah hingga kualitas yang paling tinggi.

## Masalah

Berdasarkan latar belakang yang telah diuraikan diatas, maka dapat dirumuskan rincian masalah apa saja yang dapat diselesaikan pada proyek ini :
* Bagaimana melakukan pra-pemrosesan data agar bisa digunakan pada model machine learning ?
* Bagaimana membuat model machine learning agar dapat mengklasifikasikan kualitas dari _wine_ ?

## Tujuan

Adapun tujuan dari proyek ini yaitu :
* Melakukan _pra-pemrosesan_ data agar bisa digunakan pada model machine learning
* Membuat model macbine learning untuk mengklasifikasi kualitas _wine_

## Solusi

Adapun solusi untuk mencapai tujuan diatas yaitu :

* _Pra-pemrosesan_ dapat dilakukan dengan beberapa teknik, yaitu
  * Melakukan _Categorical Encoding_ sebagai proses untuk mengubah data numerik menjadi data kategori menggunakan One-Hot Encoding
  * Melakukan _Split Data_ dengan membagi 2 dataset sebagai data latih (train data) dan data test (test data) dengan perbandingan rasio 80% : 20%.
  * Melakukan standardisasi data pada fitur numerik dengan _StandarScaler_.

* Untuk pembuatan model proyek ini menggunakan algoritma *Support Vector Machine* (SVM) sebagai model baseline. Konsep SVM dapat dijelaskan secara sederhana sebagai usaha mencari hyperplane terbaik yang berfungsi sebagai pemisah dua buah kelas pada input space. Pattern merupakan anggota dari dua buah kelas: +1 dan -1 dan berbagi alternatif garis pemisah (discrimination boundaries). Margin adalah jarak antara hyperplane tersebut dengan pattern terdekat dari masing-masing kelas. Pattern yang paling dekat ini disebut sebagai support vector. Usaha untuk mencari lokasi hyperplane ini merupakan inti dari proses pembelajaran pada SVM [2].
 <img width="596" alt="image" src="https://user-images.githubusercontent.com/96508690/196625493-76f16037-f2e3-468d-a12c-f98c97e8d11e.png"> 
 <sub>Gambar 1. Batas keputusan yang mungkin untuk set datapada SVM</sub>
 
 
 
  Dalam proyek ini menggunakan SVM Klasifikasi Non-Linier. Adapun cara kerjanya yaitu : 
  * Data dimuat
  * Mentransformasikan data menjadi ruang baru
  * Memisahkan data dengan mengimplementasikan beberapa fungsi kernel, antara lain yaitu:
    1. Polynomial
    
          $$K (xi,x) = (γ.xi^T,x+r)^p$$

       
    2. Gaussian 
    
       $$K(xi.xj)=exp(−{‖xi−xj‖^2 \over 2a^2})$$

       
    3. Sigmoid 
    
          $$K(xi,x)= tanh (γxi^T+r)$$

   
   Adapun kelebihan dan kekurangan dari SVM, antara lain :
   * Kelebihan :
     * Pengklasifikasi SVM menawarkan akurasi yang tinggi dan dapat bekerja dengan baik dengan ruang dimensi tinggi. SVM melakukan klasifikasi pada dasarnya menggunakan subset dari poin pelatihan sehingga hasilnya menggunakan memori yang sangat sedikit.
     * Landasan teori Sebagai metode yang berbasis statistik, SVM memiliki landasan teori yang dapat dianalisa dengan jelas, dan tidak bersifat black box.
     * Feasibility SVM bisa diimplementasikan dengan relatif mudah, karena proses penentuan support vector dapat dirumuskan dalam QP problem.
   * Untuk keekurangan-nya sendiri yaitu :
     * Sulit dipakai dalam problem berskala besar. Yang mana skala besar dalam hal ini dimaksudkan dengan jumlah sample yang diolah.
     * SVM secara teoritik dikembangkan untuk problem klasifikasi dengan dua class
     * Memiliki waktu pelatihan yang tinggi sehingga pada saat praktiknya tidak cocok untuk kumpulan data yang besar

## Data Understanding

Dataset yang digunakan pada proyek ini adalah dataset untuk memprediksi kulalitas _Red Wine_ yang diunduh melalui kaggle : [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/)
Pada dataset yang diunduh terdapat 1599 baris dan memiliki 12 kolom. Berdasarkan informasi dari dataset, variabel yang ada didalam dataset _Red Wine Quality_ sebagai berikut :
 
 (berdasarkan _physicochemical tests_):
  1. fixed acidity
  2. volatile acidity
  3. citric acid
  4. residual sugar
  5. chlorides
  6. free sulfur dioxide
  7. total sulfur dioxide
  8. densty
  9. pH
  10. sulphates
  11. alcohol

 (berdasarkan _sensory data_) :
  
  12. quality
```
report.wine()
```

   |    | Column               | d_type  | unique_sample                      | n_unique_sample |
|----|----------------------|---------|------------------------------------|-----------------|
| 0  | fixed acidity        | float64 | [7.4,7.8,11.2,7.9,7.3]             | 96              |
| 1  | volatile acidity     | float64 | [0.7,0.88,0.76,0.28,0.66]          | 143             |
| 2  | citric acid          | float64 | [0.0,0.04,0.56,0.06,0.02]          | 80              |
| 3  | residual sugar       | float64 | [1.9,2.6,2.3,1.8,1.6]              | 91              |
| 4  | chlorides            | float64 | [0.076,0.098,0.092,0.075,0.069]    | 153             |
| 5  | free sulfur dioxide  | float64 | [11.0,25.0,15.0,17.0,13.0]         | 60              |
| 6  | total sulfur dioxide | float64 | [34.0,67.0,54.0,60.0,40.0]         | 144             |
| 7  | densty               | float64 | [0.9978,0.9968,0.997,0.998,0.9964] | 436             |
| 8  | pH                   | float64 | [3.51,3.2,3.26,3.16,3.3]           | 89              |
| 9  | sulphates            | float64 | 0.56,0.68,0.65,0.58,0.46]          | 96              |
| 10 | alcohol              | float64 | [9.4,9.8,10.0,9.5,10.5]            | 65              |
| 11 | quality              | object  | [medium,high,easy,very high]       | 4               |


Pada table yang tertera diatas dijelaskan bahwa pada data hanya memiliki 1 data kategori bertipe object dan data lainnya merupakan data numerik bertipe float64.
Berikut Visualisasi data kategori, yaitu:

![download](https://user-images.githubusercontent.com/96508690/196654702-3edbfdd2-5d6a-4860-bd89-30f25e81d12d.png)
<sub>Gambar 2. Hasil analisa dari _categorical features_</sub>


Selanjutnya untuk visualisasi numeriknya dapat dilihat sebagai berikut :

![download](https://user-images.githubusercontent.com/96508690/196655619-833be0fe-aebc-4ce8-9474-aeba27a7d890.png)

<sub>Gambar 3. Hasil analisa visualisasi _numerical features_</sub>


Lalu terdapat visualisasi distribusi data pada kolom dengan numerik features dan antar numeric features, yang dapat dilihat sebagai berikut :

![image](https://user-images.githubusercontent.com/96508690/196657155-5e3ab751-57fd-4298-90b9-3d4b51586997.png)
<sub>Gambar 4. Visualisasi hasil distribusi _Numeric_ dan _Categorical features_</sub>


Dan berikut untuk visualisasi heatmap atau kolerasi numeric features :

![image](https://user-images.githubusercontent.com/96508690/196657513-97406276-27ee-4e8f-93ea-655ea7e05892.png)

<sub>Gambar 5. Visualisasi hasil dari _Colleration_</sub>



 - Jika heatmap mendekati 1 maka semakin tinggi pula kolerasi antar fitur numerik
 - Jika heatmap mendekati -1 maka kolerasi antar fitur numerik semakin rendah
 - Jika heatmap mendekati 0 maka kolerasi antar fitur numerik mendekati netral
 
 
 ## Data Preparation
 
 Seperti yang sudah diketahui sebelumnya pada bagian Solution statements ada beberapa tahap-tahap dalam melakukan pra-pemrosesan, yaitu sebagai berikut :
  1. Melakukan _Categorical Encoding_ yang digunakan sebagai proses untuk mengubah data numerik ke data kategori. Untuk teknik Encoding fitur kategori menggunakan One-Hot Encoding. One-Hot Encoding berfungsi untuk data nominal. yang mana data nominal diklasifikasikan tanpa urutan atau peringkat.
  2. _Split Data_ yang merupakan pembagian dataset menjadi 2, yaitu data latih (_train data_) dan data tes (_test data_). Data latih berguna untuk pelatihan model dan data tes untuk menguji model.
  3. Standarisasi data pada _numeric features_ yang memiliki tujuan yaitu agar membuat data numerik pada variabel independen memiliki rentang nilai yang sama.
  
 
 ## Modeling
 
 Setelah melakukan pra-pemrosesan data yang baik, pada tahap modeling akan melakukan 2 hal yaitu tahap pembuatan model baseline dan tahap pembuatan model yang dikembangkan.
 * Model baseline pada tahap ini akan membuat model dasar dengan menggunakan modul dari scikit-learn yaitu SVC dengan parameter default lalu selanjutnya akan melakukan prediksi pada data tes.
 * Model yang dikembangkan akan dilakukan setelah melihat kinerja dari model _baseline_, agar model dapat bekerja lebih optimal maka membutuhkan _Hyper Parameter Tuning_.
 _Hyper Parameter Tuning_ digunakan untuk mencari parameter terbaik yang nanti akan diterpakan pada model _baseline_. Pada analis proyek kali ini akan menggunakan _Grid Search Cross Validation_ dan _Grid Search Cross Validation_ yang mana merupakan metode pemilihan kombinasi model dan hyperparameter dengan cara menguji 1/1 kombinasi dan melakukan validasi untuk setiap kombinasi, tujuannya agar dapat digunakan untuk jadi model saat prediksi.
 
 
 ## Evaluasi
 
 Pada proses evaluasi proyek ini menggunakan _confussion matriks_ .
  * _Confussion matriks_ yaitu pengukur performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih.

Berikut perbandingan dari _confussion matriks_ pada analisa kedua model dengan hyperparameter yang akan di tuning adalah 
```
parameters = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'] }
```
setelah menetapkan StratifiedKFold, melakukan pembuatan model untuk GridSearchCV, dan melakukan training maka hasilnya :
``` print("Best parameter: ", grid.best_estimator_)
print("Score: ", grid.best_score_)
```
```
Best parameter:  SVC(C=100, gamma=0.01)
Score:  0.7102540970262643
```
dengan hasil visualisasi gambar berikut :
 * Model _baseline_


    ![image](https://user-images.githubusercontent.com/96508690/196663921-405ca076-8920-415b-b029-3ede1bef1f5b.png)   
    <sub>Gambar 6. Visualisasi Confussion matriks pada model baseline</sub>


* Model yang dikembangkan


    ![image](https://user-images.githubusercontent.com/96508690/196664084-e1b5b04c-2759-4868-9cea-21d84420e8ed.png)
     
     <sub>Gambar 7. Visualisasi Confussion matriks untuk best parameters</sub>
     
   Dari 2 gambar diatas bisa dilihat bahwa nilai _False Positif_ dan _False Negatif_ yang terlihat di model _baseline_ lebih besar daripada model yang dikembangkan.

* _Classificastion Report_
 
Berikut tabel untuk bagian pembuatan model baseline yaitu :
 
 
|           | easy | medium     | high       | very high | accuracy | macro avg  | weighed avg |
|-----------|------|------------|------------|-----------|----------|------------|-------------|
| precision | 0.0  | 0.624000   | 0.714286   | 0.0       | 0.672794 | 0.334571   | 0.634097    |
| recall    | 0.0  | 0.678261   | 0.744681   | 0.0       | 0.672794 | 0.355735   | 0.672794    |
| f1-score  | 0.0  | 0.650000   | 0.729167   | 0.0       | 0.672794 | 0.344792   | 0.652803    |
| support   | 13.0 | 115.000000 | 141.000000 | 3.0       | 0.672794 | 272.000000 | 272.000000  |

yang mana yang ditampilkan adalah nilai dari akurasi,precision,recall,dan f1-score untuk model.

Selanjutnya pada bagian evaluasi model yang dikembangkan terdapat tabel parameter terbaik sebagai berikut :
|           | easy      | medium     | high       | very high | accuracy | macro avg  | weighed avg |
|-----------|-----------|------------|------------|-----------|----------|------------|-------------|
| precision | 0.500000  | 0.636364   | 0.702703   | 0.0       | 0.669118 | 0.459767   | 0.657217    |
| recall    | 0.076923  | 0.669565   | 0.737589   | 0.0       | 0.669118 | 0.371019   | 0.669118    |
| f1-score  | 0.133333  | 0.652542   | 0.719723   | 0.0       | 0.669118 | 0.376400   | 0.655355    |
| support   | 13.000000 | 115.000000 | 141.000000 | 3.0       | 0.669118 | 272.000000 | 272.000000  |

 Sehingga secara keseluruhan dapat disimpulkan bahwa:
 - Pada model baseline mendapatkan nilai accuracy yaitu 67.27% begitupun dengan nilai precision, recall dan f1-score. sedangkan pada model parameter terbaik nilai accuracy yang di dapat lebih rendah yaitu 66,91% begitupun dengan nilai pada precision, recall dan f1-score.
 - Dari _confusion matriks_ dapat dilihat bahwa model baseline menggunakan Hyper Parameter Tuning memiliki nilai yang lebih baik. 

Sehingga model yang dipilih adalah model baseline menggunakan _Hyperparameter Tuning_.
   
   - _Accuracy_ merupakan gambaran seberapa akurat model dalam mengklasifikasi.
   - _Precision_ merupakan gambaran _accuracy_ antara dua data yang diminta dengan hasil prediksi yang diberikan oleh model.
   -  _Recall_ merupakan gambaran keberhasilan dari model dalam menemukan kembali suatu informasi.
   -  F1-Score merupakan gambaran perbandingan rata-rata _precision_ dan _recall_ yang dibobotkan. dengan _accuracy_ yang tepat dapat digunakan sebagai acuan performansi algoritma jika dataset memiliki jumlah data _False negatif_ dan _False positif_ yang sangat mendekati, namun jika jumlah tidak mendekati maka gunakan f1-score sebagai acuan [3].

## References
[1]Andono, P. N., & Rachmawanto, E. H. (2020). Evaluasi Ekstraksi Fitur GLCM dan LBO Menggunakan Multikernel SVM untuk Klasifikasi Batik. JURNAL RESTI, 1-9

[2]Supryadi, R., Gata, W., Maulidah, N., & Fauzi, A. (2020). Penerapan Algoritma Random Forest Untuk Menentukan Kualitas Anggur Merah. JURNAL ILMIAH EKONOMI DAN BISNIS, 1-9.

[3] https://socs.binus.ac.id/2020/11/01/confusion-matrix/
