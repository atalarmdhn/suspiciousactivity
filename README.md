# Deteksi Suspicious Activity dengan 3D Convolutional Autoencoder (PyTorch)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Proyek ini adalah implementasi **PyTorch** dari paper "Learning Temporal Regularity in Video Sequences". Tujuannya adalah untuk melatih sebuah model **3D Convolutional Autoencoder** untuk mempelajari pola gerakan "normal" dalam video. Model yang sudah terlatih kemudian dapat digunakan untuk mendeteksi anomali atau kejadian tidak wajar dengan mengukur seberapa besar kesalahan rekonstruksi pada klip video baru.



---

## 📌 Fitur Utama

-   **Model 3D ConvAE**: Menggunakan arsitektur autoencoder dengan konvolusi 3D untuk menangkap fitur spasial (ruang) dan temporal (waktu) dari video.
-   **Dua Mode Operasi**: Mendukung mode `train` untuk melatih model dari awal dan mode `eval` untuk mengevaluasi video menggunakan model yang sudah ada.
-   **Input Fleksibel**: Dapat memproses data baik dari file video langsung (`.mp4`, `.avi`) maupun dari folder yang sudah berisi frame-frame gambar.
-   **Antarmuka Command-Line**: Semua parameter seperti *learning rate*, *batch size*, dan path direktori dapat diatur dengan mudah melalui argumen command-line.
-   **Checkpointing**: Secara otomatis menyimpan model terbaik (`best.pt`) dan model terakhir (`last.pt`) selama proses pelatihan.
-   **Output Jelas**: Menghasilkan file `.csv` yang berisi *regularity score* untuk setiap klip video pada mode evaluasi, memudahkan analisis lebih lanjut.

---

## ⚙️ Cara Kerja

Konsepnya sederhana:
1.  **Pelatihan**: Sebuah *Autoencoder* dilatih hanya dengan video-video berisi aktivitas **normal**. Model belajar untuk memadatkan (encode) dan merekonstruksi (decode) klip-klip normal ini dengan kesalahan (MSE) sekecil mungkin.
2.  **Evaluasi**: Ketika model yang sudah terlatih diberi klip video baru:
    -   Jika klip tersebut **normal**, model akan mampu merekonstruksinya dengan baik (kesalahan rendah).
    -   Jika klip tersebut **anomali**, model akan kesulitan merekonstruksinya (kesalahan tinggi).
3.  **Skor Anomali**: Kesalahan rekonstruksi yang tinggi diubah menjadi *Regularity Score* yang rendah, yang menandakan adanya anomali.

---

## ▶️ Cara Penggunaan

Program ini dijalankan melalui terminal dengan menentukan mode (`train` atau `eval`) dan argumen lainnya.

### 1. Mode Pelatihan (`train`)

Gunakan mode ini untuk melatih model dari awal. Siapkan sebuah folder berisi video-video yang hanya menampilkan aktivitas normal.

**Contoh Perintah:**
```bash
python unsuplearn.py --mode train --data_root /path/ke/folder_video --clip_len 16 --height 128 --width 192 --epochs 10 --batch_size 4 --lr 1e-4 --work_dir ./runs/exp1
```

### 2. Mode Evaluasi (`eval`)

Gunakan mode ini untuk mengevaluasi video yang ingin ditest.

**Contoh Perintah:**
```bash
python unsuplearn.py --mode eval --data_root "C:\VIXMO AI\0-Code\loitering\Avenue Dataset\testing_videos" --clip_len 16 --height 128 --width 192 --checkpoint ./runs/exp1/checkpoints/best.pt --out_csv ./runs/exp1/test_scores.csv
```
## 😵‍💫 Hasil
Pada proses uji coba ini video yang telah di evaluasi nantinya akan menjadi .csv, lalu untuk mempermudah analisa maka data ditampilkan dalam bentuk grafik dan untuk melihat frame dari video yang telah dievaluasi maka video tersebut dipisah per framenya dan diletakkan dalam 1 folder, berikut beberapa dokumentasi dari hasil percobaan:

<img width="1171" height="1027" alt="Screenshot 2025-08-14 154808" src="https://github.com/user-attachments/assets/d68543b4-6bcc-4e01-b317-13a152bc5d5f" />
Frame ke-59

<img width="1186" height="1046" alt="Screenshot 2025-08-14 154840" src="https://github.com/user-attachments/assets/a49a5be9-9ba4-4897-aa5a-526716767a0b" />
Frame Ke-257

<img width="1181" height="1079" alt="Screenshot 2025-08-14 154858" src="https://github.com/user-attachments/assets/07fe2fd0-62f2-4ea0-b83c-447aa6fdc24f" />
Frame Ke-197

<img width="1192" height="1079" alt="Screenshot 2025-08-14 154934" src="https://github.com/user-attachments/assets/aa921587-8b1f-4ad4-9ac9-1c5ebf353650" />
Frame Ke-576



