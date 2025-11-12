# 🚗 Sistem Analisis Lalu Lintas (YOLOv8, Flask & MySQL)

Sistem analisis lalu lintas berbasis web yang menyediakan deteksi dan penghitungan kendaraan secara *real-time* dari *stream* YouTube, serta pemrosesan video *on-demand* dengan model AI yang berbeda.

<p align="center">
  <img src="https://imgur.com/a/NCLeYoS" alt="Demo Aplikasi" width="800"/>
</p>
<p align="center">
  <img src="https://imgur.com/a/ACXSSwo" alt="snapshot" width="800"/>
</p>

---

## Abstract

Proyek ini adalah aplikasi web Flask lengkap yang dirancang untuk pemantauan lalu lintas cerdas. Aplikasi ini memiliki dua fungsionalitas inti: (1) **Dasbor Pemantauan Live**, yang melacak kendaraan dari *stream* YouTube menggunakan model YOLOv8 kustom dan mencatat total hitungan ke database. (2) **Modul Pemrosesan Latar Belakang**, yang memungkinkan pengguna mengunggah file video atau link YouTube untuk dianalisis oleh model YOLOv8 standar (`yolov8n.pt`).

Untuk mengatasi masalah kompatibilitas video, hasil pemrosesan disimpan sebagai kumpulan gambar (JPG) dari setiap *event* deteksi, yang kemudian dikompresi ke dalam satu file `.zip`. Seluruh sistem terhubung ke database MySQL yang mencatat total hitungan dan log *timestamp* terperinci untuk setiap kendaraan yang terdeteksi.

---

## ✨ Fitur Utama

-   **🖥️ Dasbor Live Dinamis:**
    -   Streaming video langsung dari YouTube ke antarmuka web.
    -   Formulir UI untuk **mengganti link YouTube kapan saja**.
    -   **Reset Otomatis:** Mengganti link stream akan secara otomatis **mereset** total hitungan di database ke 0, memulai sesi pemantauan baru.
    -   Statistik *real-time* (FPS, Status DB, Total Kendaraan) diperbarui setiap 2 detik.

-   **🚀 Arsitektur Dua Model:**
    1.  **Live Stream:** Menggunakan model **YOLOv8 kustom (`best.pt`)** yang dilatih untuk 4 kelas (Mobil, Motor, Bus, Truk).
    2.  **Proses Upload:** Menggunakan model **YOLOv8n standar (`yolov8n.pt`)** untuk fleksibilitas, memetakan kelas COCO ke database.

-   **📤 Pemrosesan Latar Belakang & Output ZIP:**
    -   Halaman `/upload` terpisah untuk memproses video atau link YouTube di latar belakang (`threading`).
    -   **Anti-Video Rusak:** Prosesor tidak membuat file MP4. Sebaliknya, ia **menyimpan satu gambar JPG** berkualitas tinggi setiap kali kendaraan **dihitung** (melintasi garis).
    -   Secara otomatis mengompres semua gambar hasil deteksi ke dalam satu **file `.zip`** yang siap diunduh.
    -   Halaman status tugas (berwarna merah) yang me-refresh otomatis hingga file ZIP siap.

-   **📊 Manajemen Database Ganda (MySQL):**
    1.  **Tabel `traffic_stats`:** Menyimpan 1 baris **total akumulasi** (untuk tampilan dasbor).
    2.  **Tabel `detection_log`:** Menyimpan **log *timestamp*** terperinci untuk *setiap* kendaraan yang dihitung dari *semua sumber* (Live Stream & Upload).

-   **📈 Download Laporan CSV:**
    -   Fitur di Navbar untuk mengunduh **rekap penuh** dari tabel `detection_log` sebagai file `.csv` untuk analisis data historis.

-   **⚙️ Manajemen Server (Siap Hosting):**
    -   Menggunakan `apscheduler` untuk **menghapus file secara otomatis** dari folder `uploads/` 5-15 menit setelah diunggah/diunduh, mencegah server penuh.

---

## 🛠️ Tumpukan Teknologi (Technology Stack)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=yellow)
![Flask](https://img.shields.io/badge/Flask-Web_Server-black?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object-Detection-blueviolet)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_process-orange?logo=opencv)
![MySQL](https://img.shields.io/badge/MySQL-Database-blue?logo=mysql)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-red?logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-Styling-blue?logo=css3)
![JavaScript](https://img.shields.io/badge/JavaScript-Dynamic-yellow?logo=javascript)

---

## 🏛️ Arsitektur Sistem

Sistem ini berjalan dengan dua alur kerja utama yang berinteraksi dengan database yang sama.

---

## 🚀 Instalasi & Konfigurasi

Ikuti langkah-langkah ini untuk menjalankan aplikasi secara lokal.

### 1. Prasyarat
* Python 3.11+
* Server MySQL (Contoh: **XAMPP** dengan **phpMyAdmin**)

### 2. Kloning Repositori
```bash
git clone https://github.com/Pashinoh/car-detection-recognition.git
cd car-detection-recognition