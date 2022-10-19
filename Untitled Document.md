# Laporan Proyek 1 Machine Learning Terapan - Fauziah Umri
## Domain Proyek
Anggur merupakan salah satu buah yang banyak di konsumsi dan cukup populer di berbagai wilayah. Buah anggur biasanya di konsumsi secara langsung dan juga diolah menjadi suatu produk seperti makanan dan minuman yang difermentasi yang akan menjadi minuman beralkohol seperti _wine_. Biasanya jangka waktu yang dibutuhkan untuk fermentasi anggur untuk menjadi _wine_ bervariasi, ada yang membutuhkan waktu singkat dan adapula yang membutuhkan waktu yang lama.
Beberapa jenis-jenis _wine_ yaitu _Rose Wine, Sweet Wine, Sparkling Wine, Red Wine, White Wine,_ dan _Fortified Wine_. 

Anggur memiliki berbagai karakteristik seperti kepadatan, nilai pH, alkohol dan asam lainnya. Dalam perkembanganya _wine_ semakin bermacam varianya. Hal itu pula yang membuat _wine_ di bagi berdasarkan kualitasnya untuk memnetukan harga jual di pasaran. Kulaitas pada _wine_ dipengaruhi oleh bebrapa faktor, contohnya komposisi yang terdapat di dalamnya. Untuk menentukan kualitas _wine_ tentu harus ada ahli yang bertugas untuk mencicipi sampel dari minuman anggur tersebut.

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
  * Melakukan _Categorilcal Encoding_ sebagai proses untuk mengubah data numerik menjadi data kategori menggunakan One-Hot Encoding
  * Melakukan _Split Data_ dengan membagi 2 dataset sebagai data latih (train data) dan data test (test data) dengan perbandingan rasio 80% : 20%.
  * Melakukan standardisasi data pada fitur numerik dengan _StandarScaler_.

* Untuk pembuatan model proyek ini menggunakan algoritma *Support Vector Machine* (SVM) sebagai model baseline. Konsep SVM dapat dijelaskan secara sederhana sebagai usaha mencari hyperplane terbaik yang berfungsi sebagai pemisah dua buah kelas pada input space. Pattern merupakan anggota dari dua buah kelas: +1 dan -1 dan berbagi alternatif garis pemisah (discrimination boundaries). Margin adalah jarak antara hyperplane tersebut dengan pattern terdekat dari masing-masing kelas. Pattern yang paling dekat ini disebut sebagai support vector. Usaha untuk mencari lokasi hyperplane ini merupakan inti dari proses pembelajaran pada SVM 
 <img width="596" alt="image" src="https://user-images.githubusercontent.com/96508690/196625493-76f16037-f2e3-468d-a12c-f98c97e8d11e.png">


  Dalam proyek ini menggunakan SVM Klasifikasi Non-Linier. Adapun cara kerjanya yaitu : 
  * Data dimuat
  * Mentransformasikan data menjadi ruang baru
  * Memisahkan data dengan mengimplementasikan beberapa fungsi kernel, antara lain yaitu:
    1. Polynomial
    
          <img width="217" alt="image" src="https://user-images.githubusercontent.com/96508690/196625157-e29ddd80-2c3d-411d-9817-9aa43851e204.png">

       
    2. Gaussian 
    
          <img width="233" alt="image" src="https://user-images.githubusercontent.com/96508690/196625250-40c350e4-9d3c-4135-b3af-01c1fb048207.png">

       
    3. Sigmoid 
    
          <img width="250" alt="image" src="https://user-images.githubusercontent.com/96508690/196625352-8f5a94ca-dc51-4526-8c9c-9f3ce883001f.png">

   
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

Dataset yang digunakan pada proyek ini adalah dataset untuk memprediksi kulalitas _Red Wine_ yang diunduh melalui kaggle : https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
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

   <img width="417" alt="image" src="https://user-images.githubusercontent.com/96508690/196632795-a8de7606-67ab-468e-a1ac-ffa63dcd734e.png">


Pada gambar yang tertera diatas dijelaskan bahwa pada data hanya memiliki 1 data kategori bertipe object dan data lainnya merupakan data numerik bertipe float64.
Berikut Visualisasi data kategori, yaitu:
  
   <img width="257" alt="image" src="https://user-images.githubusercontent.com/96508690/196633785-c4740c32-a8d9-477e-9291-b4939751c154.png">

