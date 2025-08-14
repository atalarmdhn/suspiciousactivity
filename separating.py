import cv2
import os

# --- PENGATURAN ---
# Ganti dengan path video Anda
nama_video = "C:/VIXMO AI/0-Code/loitering/lab/lab.mp4"
# Nama folder untuk menyimpan frame
nama_folder_output = "lab_frames"
# --------------------

# Membuat folder output jika belum ada
if not os.path.exists(nama_folder_output):
    os.makedirs(nama_folder_output)
    print(f"Folder '{nama_folder_output}' berhasil dibuat.")

# Membuka file video
vidcap = cv2.VideoCapture(nama_video)
if not vidcap.isOpened():
    print(f"Error: Tidak bisa membuka video '{nama_video}'")
    exit()

# Variabel untuk menghitung frame
jumlah_frame = 0

print(f"Memulai proses ekstraksi frame dari '{nama_video}'...")

while True:
    # Membaca satu frame dari video
    # 'success' adalah boolean (True/False)
    # 'image' adalah data frame itu sendiri
    success, image = vidcap.read()

    # Jika 'success' bernilai False, berarti video sudah habis
    if not success:
        break

    # Membuat nama file untuk setiap frame (misal: frame_0.jpg, frame_1.jpg, dst.)
    nama_file_frame = os.path.join(nama_folder_output, f"frame_{jumlah_frame}.jpg")

    # Menyimpan frame sebagai file gambar
    cv2.imwrite(nama_file_frame, image)
    
    # Menambah hitungan frame
    jumlah_frame += 1

# Melepaskan video capture setelah selesai
vidcap.release()

print(f"Proses selesai. Total {jumlah_frame} frame berhasil disimpan di folder '{nama_folder_output}'.")