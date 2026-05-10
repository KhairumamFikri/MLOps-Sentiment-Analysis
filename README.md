# MLOps - Sentiment Analysis

## Deskripsi Proyek

Project ini bertujuan untuk membangun fondasi sistem MLOps untuk tugas analisis sentimen menggunakan pendekatan reproducible development environment melalui GitHub Codespaces dan penerapan GitHub Flow sebagai strategi pengembangan kode.

Repository ini dirancang agar proses pengembangan, eksperimen, dan kolaborasi dapat dilakukan secara konsisten pada lingkungan yang sama tanpa konfigurasi manual yang kompleks.

---

## Tujuan Proyek

Tujuan utama dari project ini adalah:

* Membangun struktur proyek machine learning yang terstandarisasi
* Mengimplementasikan workflow pengembangan menggunakan GitHub Flow
* Menyediakan environment pengembangan reproducible menggunakan GitHub Codespaces
* Menyiapkan fondasi pipeline MLOps untuk eksperimen machine learning
* Mendukung kolaborasi pengembangan model secara terstruktur

---

## Teknologi yang Digunakan

* Python 3.10
* GitHub Codespaces
* GitHub Flow
* Jupyter Notebook
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## Struktur Direktori

```text
MLOps-Sentiment-Analysis/
│
├── .devcontainer/         -> Konfigurasi GitHub Codespaces
├── config/                -> File konfigurasi project
├── data/
│   ├── raw/               -> Dataset mentah
│   └── processed/         -> Dataset hasil preprocessing
│
├── models/                -> Penyimpanan model machine learning
├── notebooks/             -> Notebook eksperimen dan EDA
│
├── src/
│   ├── data/              -> Script pengolahan data
│   ├── features/          -> Feature engineering
│   ├── models/            -> Training dan evaluasi model
│   └── visualization/     -> Visualisasi data
│
├── requirements.txt       -> Dependency project
├── setup.py               -> Setup package project
├── README.md              -> Dokumentasi project
└── LICENSE                -> Lisensi project
```

---

## Konfigurasi GitHub Codespaces

Project ini menggunakan GitHub Codespaces untuk menyediakan lingkungan pengembangan yang konsisten.

Konfigurasi Codespaces mencakup:

* Python 3.10 environment
* Instalasi dependency otomatis
* Extension VS Code untuk Python dan Jupyter
* Dukungan pengembangan machine learning

File konfigurasi berada pada:

```text
.devcontainer/devcontainer.json
```

---

## Cara Menjalankan Project Menggunakan Codespaces

### 1. Membuka Repository

Buka repository GitHub project ini.

### 2. Membuat Codespace

* Klik tombol `Code`
* Pilih tab `Codespaces`
* Klik `Create codespace on main`

### 3. Instalasi Dependency

Setelah Codespaces aktif, jalankan:

```bash
pip install -r requirements.txt
```

### 4. Menjalankan Jupyter Notebook

```bash
jupyter notebook
```

---

## Branching Strategy

Project ini menggunakan GitHub Flow sebagai strategi pengembangan.

Branch utama:

```text
main
```

Branch eksperimen awal:

```text
feat/initial-eda
```

Perubahan hanya akan di-merge ke branch `main` setelah proses validasi selesai dilakukan melalui Pull Request.

---

## Workflow Pengembangan

1. Membuat branch fitur atau eksperimen baru
2. Melakukan pengembangan dan commit perubahan
3. Push branch ke GitHub
4. Membuat Pull Request
5. Review dan validasi perubahan
6. Merge ke branch `main`

---

## Lisensi

Project ini menggunakan lisensi MIT License.

---

