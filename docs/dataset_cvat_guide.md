# Dataset Guide (CVAT -> PRTReid)

Panduan ini fokus ke kebutuhan **multitask**: `pid + team + role`.

## 1) Apakah CVAT bisa untuk Re-ID?

Bisa. Untuk Re-ID, yang paling penting adalah **ID orang konsisten antar frame**.  
Di CVAT itu artinya kamu harus pakai mode **Track**, bukan gambar per gambar manual.

## 2) Track-by-track atau frame-by-frame?

Rekomendasi: **track-by-track**.

- `Track-by-track`:
  - pro: ID konsisten otomatis, cepat untuk video panjang
  - kontra: tetap perlu koreksi saat occlusion/keluar-masuk frame
- `Frame-by-frame`:
  - pro: detail tinggi per frame
  - kontra: sangat lambat, rawan ID tidak konsisten

Untuk pemula, workflow efektif:
1. Buat track otomatis/semi-otomatis di CVAT.
2. Koreksi drift track.
3. Pastikan setiap orang punya track ID stabil.

## 3) Format export CVAT yang dipilih

Untuk script converter yang sudah disiapkan di project ini, pilih:

- **CVAT for video 1.1 (XML)**

Kenapa:
- Menyimpan `track id` + `box per frame`.
- Mudah diparse jadi crop ReID.

## 4) Setup label di CVAT (disarankan)

Label utama:
- `player`
- `goalkeeper`
- `referee`
- `ball` (opsional untuk pipeline lain, tidak wajib untuk ReID person)

Attribute yang disarankan pada track/box:
- `team`: `left` / `right` / `other`
- `role`: `player` / `goalkeeper` / `referee` / `ball` / `other`

Catatan:
- Jika `role` tidak diisi, converter fallback ke nama label track.
- Jika `team` tidak diisi, fallback ke `other`.

## 5) Konversi hasil CVAT ke dataset crop ReID

Prasyarat:
- sudah export XML dari CVAT (`annotations.xml`)
- frame video sudah diextract jadi image (`000000.jpg`, `000001.jpg`, dst)

Contoh command:

```powershell
cd "C:\Programming\PRTReid Training"
python .\scripts\cvat_video_xml_to_reid.py `
  --cvat_xml .\data\cvat\match1\annotations.xml `
  --frames_dir .\data\cvat\match1\frames `
  --output_images_dir .\data\reid\images `
  --pid_labels_csv .\data\processed\reid\pid_labels_from_cvat.csv `
  --video_id 21 `
  --frame_pattern "{frame:06d}.jpg" `
  --frame_index_offset 0 `
  --train_pid_ratio 0.8 `
  --min_track_boxes 5 `
  --min_box_area 100 `
  --crop_padding 0.05
```

Output:
- `data/reid/images/train/<video_id>/...jpg`
- `data/reid/images/val/<video_id>/...jpg`
- `data/processed/reid/pid_labels_from_cvat.csv`

## 6) Build manifest CSV yang dipakai PRTReid

Setelah crop selesai:

```powershell
python .\scripts\build_manifests_from_reid.py `
  --source_dir .\data\reid\images `
  --output_root .\data\processed `
  --dataset_name reid `
  --pid_labels_csv .\data\processed\reid\pid_labels_from_cvat.csv `
  --require_multitask_labels
```

Ini akan membuat:
- `data/processed/reid/splits/train.csv`
- `data/processed/reid/splits/query.csv`
- `data/processed/reid/splits/gallery.csv`

## 7) Apakah ada tools selain CVAT?

Ada. Pilihan praktis:

1. Label Studio
- pro: UI simpel, cepat setup
- kontra: tracking video untuk ID stabil biasanya kurang nyaman dibanding CVAT

2. Supervisely
- pro: tooling video/tracking kuat
- kontra: setup/dependency dan lisensi bisa jadi pertimbangan

3. Roboflow Annotate
- pro: UX ramah pemula
- kontra: untuk use case ID-consistency tracking panjang, CVAT biasanya lebih fleksibel

Kesimpulan praktis:
- Untuk ReID berbasis video + konsistensi ID, **CVAT tetap opsi paling aman**.

## 8) Quality checklist sebelum training

Sebelum train, cek cepat:
- setiap PID punya minimal beberapa frame (`>=5`)  
- tidak ada merge orang berbeda dalam satu track PID  
- label `team` dan `role` konsisten untuk PID yang sama  
- split train/val punya cukup PID berbeda

