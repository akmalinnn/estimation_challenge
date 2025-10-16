
# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Akmal Nafis  
**NRP**: 5025211216  
**Judul TA**: Estimasi Kecepatan Relatif dengan Penghapusan Objek Di Optical Flow  

---


# DashFlow
Proyek ini menyediakan pipeline untuk mengestimasi kecepatan kendaraan menggunakan optical flow pada video dengan model RAFT untuk optical flow dan Sernet-former untuk segmentasi. Pipeline ini mengekstrak frame dari video, menghitung optical flow, menerapkan segmentation mask, serta menghasilkan perbedaan antara flow field dan mengestimasi kecepatan menggunakan model deep learning.


## Persyaratan

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- ffmpeg 
- Pandas
- Matplotlib
- tqdm
- Dependensi lain yang tercantum di `Sernet_former/requirements.txt`
- Dependensi lain yang tercantum di `Raft/requirements.txt`

- model pretrained pada Sernet Former dan Raft menggunakan KITTI dan cityscapes atau sejenisnya, agar sesuai dengan video dashcam. 

Masuk ke folder terkait, lalu instal dependensi Python:
```
pip install -r Sernet_former/requirements.txt
pip install -r Raft/requirements.txt
```

# Flow Difference
Digunakan untuk menghasilkan file flow_diff.npy

## Penggunaan

1. **Siapkan Video Input**

   Tempatkan file `.mp4` Anda di direktori `input_videos/`.

2. **Jalankan Pipeline**

   ```
   python main.py
   ```

   Ini akan:
   - Mengekstrak frame dari setiap video.
   - Menghitung optical flow antar frame menggunakan RAFT.
   - Menerapkan segmentation mask menggunakan Sernet-former.
   - Menghitung dan menyimpan perbedaan antar flow field, beserta visualisasinya.

3. **Hasil**

   - Frame yang diekstrak: `extracted_frames/<nama_video>/`
   - Optical flow: `flow/<nama_video>/`
   - Perbedaan flow dan visualisasi: `flow_diff/diff_<nama_video>/`

## Catatan

- Pastikan Anda memiliki bobot model (model weights) untuk RAFT dan Sernet-former di direktori yang sesuai.
- Script akan otomatis membuat folder output yang diperlukan jika belum ada.
- Pipeline akan memproses semua file `.mp4` di folder `input_videos/` (batch video).

## Struktur Proyek

```
main.py
Raft/
Sernet_former/
input_videos/
extracted_frames/
flow/
flow_diff/
```

- **main.py**: Script utama untuk menjalankan pipeline.
- **Raft/**: Berisi model RAFT dan utilitas optical flow.
- **Raft/rdemo.py** : file tuning dari demo.py yang disesuaikan untuk project ini
- **Sernet_former/**: Berisi model segmentasi dan script terkait.
- **input_videos/**: Tempatkan video `.mp4` Anda di sini.
- **extracted_frames/**: Frame hasil ekstraksi dari video.
- **flow/**: Hasil optical flow.
- **flow_diff/**: Perbedaan antar flow field.


---
# Training Model Klasifikasi & Regresi

Repositori ini berisi kode dan data untuk tugas klasifikasi dan regresi berbasis deep learning menggunakan PyTorch.

## Cara Menjalankan

1. Siapkan data Anda di direktori `data/`.
2. Sesuaikan data loader dan definisi model jika diperlukan.
3. Jalankan script training:
   ```
   python training_clasification/main.py
   ```
   ```
   python training_regresion/main.py
   ```
4. Hasil, plot, dan checkpoint model akan disimpan di direktori `result/`.
---

## Struktur Direktori

- **data/**  
  Berisi semua data dan script untuk preprocessing.
  - `data_filtered.json`: Dataset utama dalam format JSON, speed pada tiap frame video awal. 
  - `flow_diff.json`: Hasil perbedaan kecepatan antar frame.
  - `diff_speed.py`: Script untuk menghasilkan flow_diff.json dan membagi kelas training.
  - `sub.py`: Script untuk memindahkan file .npy dari subfolder. Contohnya`flow_diff/vid0100/flow_diff_vid0100_frame1.jpg.npy` menjadi `flow_diff/flow_diff_vid0100_frame1.jpg.npy`. Sangat berguna jika terdapat batch folder.
  - `flow_diff/`: Berisi file `.npy` yang mewakili perbedaan optical flow untuk setiap frame/urutan.
- **result/**  
  Menyimpan hasil training model, termasuk plot dan checkpoint model.
  - `training_clf/`, `training_reg/`: Hasil untuk klasifikasi dan regresi.

- **testing/**  
  Script dan modul untuk pengujian model.
  - `data_loader.py`, `test.py`: Script untuk loading data dan pengujian.
  - `cnn/`, `model/`: Definisi model cnn dan model hasil training.

- **training_clasification/**  
  Kode untuk training model klasifikasi.
  - `train.py`: Training loop utama untuk klasifikasi.
  - `data_loader.py`, `main.py`: Script loading data, dan main program
  - `cnn/`: Definisi model CNN.

- **training_regresion/**  
  Kode untuk training model regresi.
  - `train_reg.py`: Training loop utama untuk regresi.
  - `data_loader.py`, `main.py`: Script loading data dan entry point.
  - `cnn/`: Definisi model CNN.

---

## Data dan File JSON

### `data_filtered.json`
- **Tujuan:**  
    Data raw mentah setiap frame dari proses video. untuk membuat `diff_speed.py` sebagai label pada setiap flow_diff.npy

### `flow_diff.json`
- **Tujuan:**  
  Menyimpan informasi tentang perbedaan optical flow antar frame. Digunakan untuk training.

### `flow_diff/`
- **Tujuan:**  
  Berisi file `.npy` yang mewakili perbedaan optical flow untuk setiap frame/urutan. File tersebut diasosiasikan dengan flow_diff.json



# Inference DashFlow

## Penggunaan

1. **Tambahkan file video** ke folder `input_videos/` (misal: `test1.mp4`).

2. **Jalankan pipeline utama:**
   ```
   python main.py --initial_speed 0 --folder test1 --gt_json json/test1.json
   ```

   - `--initial_speed`: Nilai kecepatan awal (misal: 65.0)
   - `--folder`: Nama subfolder di dalam `flow_diff` (misal: test1)
   - `--gt_json`: (Opsional) Path ke file JSON ground truth

3. **Hasil** akan disimpan di folder `result/`.

## Fitur

- **Ekstraksi Frame:** Mengekstrak frame dari video input menggunakan FFmpeg.
- **Optical Flow:** Menghitung optical flow antar frame menggunakan RAFT.
- **Segmentasi:** Melakukan segmentasi pada frame menggunakan Sernet_former.
- **Flow Difference:** Menghitung perbedaan optical flow antar frame, dengan masking berbasis segmentasi.
- **Estimasi Kecepatan:** Mengestimasi kecepatan menggunakan model deep learning.

## Struktur Proyek

```
main.py
input_videos/
extracted_frames/
flow/
flow_diff/
json/
Raft/
Sernet_former/
Speed_estimation/
result/
temp/ (view.py untuk visualisasi panah vektor)
```

## Penggunaan

1. **Tambahkan file video** ke folder `input_videos/` (misal: `test1.mp4`).

2. **Jalankan pipeline utama:**

menggunakan Ground truth

   ```
   python main.py --initial_speed 0 --folder test1 --gt_json json/test1.json
   ```
tidak menggunakan Ground truth    
  ```
   python main.py --initial_speed 0 --folder test1 
   ```

   - `--initial_speed`: Nilai kecepatan awal (misal: 65.0)
   - `--folder`: Nama subfolder di dalam `flow_diff` (misal: test1)
   - `--gt_json`: Path ke file JSON ground truth (opsional)

3. **Hasil** akan disimpan di folder `result/`.

## Referensi

- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT)
- https://github.com/serdarch/SERNet-Former

## Catatan

- Pipeline akan memproses semua file `.mp4` di folder `input videos`
- Fps pada video sangat berpengaruh, hasil dari internet tidak konsisten. Meskipun saya menggunakan video 24fps/ hasil konversi dari 30fps ke 24fps (untuk menghindari frame duplaicate). Masih mengakibatkan adanya frame noise, pergerakannya berbeda sendiri. Sehingga pada hasil optical flow difference setiap beberapa frame akan muncul frame noise yang sangat berbeda
-  Dibeberapa frame perpindahan pikselnya tidak konsisten, biasanya tiap 6 frame pergerakannya berbeda dari frame sebelumnya. Hal tersebut mengakibatkannya noise (optical flownya berbeda sendiri). Frame tersebut seharusnya tidak dimasukkan dalam proses training.


![GIF OF](https://github.com/user-attachments/assets/0ebc7803-8d47-4069-aa85-8dd9ac9cf22c)
<img width="1668" height="496" alt="image" src="https://github.com/user-attachments/assets/4ef64a23-bcb2-474a-8ded-a2379b3c3936" />
<img width="543" height="217" alt="image" src="https://github.com/user-attachments/assets/bed18efe-0aef-4be1-b905-e0f233601256" />


