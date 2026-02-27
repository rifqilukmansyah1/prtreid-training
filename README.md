# PRTReid Training (Standalone)

Tutorial ini dibuat dalam format command-step seperti contoh kamu.
Semua runtime tetap standalone: tidak import `sn-gamestate`/`tracklab` saat train/eval.

## 0) Project Folder

```powershell
cd C:\Programming
mkdir "PRTReid Training"
cd "PRTReid Training"
```

## 1) Clone Repos

### Wajib (engine PRTReid)

```powershell
cd "C:\Programming\PRTReid Training"
mkdir third_party
cd third_party
git clone https://github.com/VlSomers/prtreid.git
cd prtreid
git checkout 30617a7
cd "C:\Programming\PRTReid Training"
```

### Opsional (hanya referensi, tidak dipakai runtime)

```powershell
cd C:\Programming
mkdir _refs
cd _refs
git clone https://github.com/SoccerNet/sn-gamestate.git
git clone https://github.com/TrackingLaboratory/tracklab.git
```

## 2) Create Conda Environment

```powershell
conda create -n prtreid_train python=3.9 pip -y
conda activate prtreid_train
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

## 3) Install Dependencies

Pastikan python/pip mengarah ke env yang benar:

```powershell
where python
where pip
```

Install dependency project + PRTReid:

```powershell
cd "C:\Programming\PRTReid Training"
pip install --upgrade pip
pip install -r .\requirements\requirements.txt
pip install -r .\third_party\prtreid\requirements.txt
pip install scikit-learn
```

Coba editable install (opsional tapi bagus kalau berhasil):

```powershell
pip install -e .\third_party\prtreid
```

Kalau editable install gagal karena compiler C++ (MSVC), tidak masalah.
Project ini sudah fallback import dari `third_party/prtreid`.

## 4) (Opsional) Auto Setup Script

Kalau mau 1 command otomatis:

```powershell
cd "C:\Programming\PRTReid Training"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -EnvName prtreid_train
conda activate prtreid_train
```

## 5) Verify Install

```powershell
cd "C:\Programming\PRTReid Training"
python .\scripts\check_install.py
```

## 6) Siapkan Dataset Multitask

Mode `multitask` wajib ada kolom `team` dan `role`.

### 6.1 Build manifest awal dari crop image

```powershell
python .\scripts\build_manifests_from_reid.py `
  --source_dir .\data\reid\images `
  --output_root .\data\processed `
  --dataset_name reid `
  --build_smoke `
  --smoke_dataset_name reid_smoke
```

### 6.2 Generate template PID label

```powershell
python .\scripts\make_pid_label_template.py `
  --train_csv .\data\processed\reid\splits\train.csv `
  --output_csv .\data\processed\reid\pid_labels_template.csv
```

Isi file `pid_labels_template.csv`:
- `team`: `left` / `right` / `other`
- `role`: `player` / `goalkeeper` / `referee` / `ball` / `other`

### 6.3 Rebuild manifest dengan label multitask

```powershell
python .\scripts\build_manifests_from_reid.py `
  --source_dir .\data\reid\images `
  --output_root .\data\processed `
  --dataset_name reid `
  --build_smoke `
  --smoke_dataset_name reid_smoke `
  --pid_labels_csv .\data\processed\reid\pid_labels_template.csv `
  --require_multitask_labels
```

## 7) Train Multitask

### Smoke run (1 epoch)

```powershell
python .\main_train.py `
  --mode multitask `
  --profile_config .\configs\prtreid\profiles\multitask_soccernet_like.yaml `
  --dataset_name reid_smoke `
  --dataset_nickname rsmk `
  --data_root .\data\processed `
  --output_dir .\outputs\smoke_multitask `
  --max_epoch 1 `
  --train_batch_size 16 `
  --test_batch_size 32 `
  --workers 2
```

### Full run

```powershell
python .\main_train.py `
  --mode multitask `
  --profile_config .\configs\prtreid\profiles\multitask_soccernet_like.yaml `
  --dataset_name reid `
  --dataset_nickname rid `
  --data_root .\data\processed `
  --output_dir .\outputs\runs `
  --max_epoch 60 `
  --train_batch_size 32 `
  --test_batch_size 64 `
  --workers 4
```

## 8) Eval Multitask

```powershell
python .\main_eval.py `
  --mode multitask `
  --profile_config .\configs\prtreid\profiles\multitask_soccernet_like.yaml `
  --dataset_name reid `
  --dataset_nickname rid `
  --data_root .\data\processed `
  --output_dir .\outputs\eval `
  --weights .\outputs\runs\<job_folder>\model\job-<id>_<epoch>_model.pth.tar
```

## 9) DLL / OMP / CUDA Troubleshooting (Windows)

### Error: `Unrecognized CachingAllocator option: expandable_segments`

```powershell
# sementara hanya di session terminal aktif
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
```

Catatan: `main_train.py` dan `main_eval.py` sudah otomatis handle ini untuk Torch 1.x.

### Error: `OMP: Error #15 ... libiomp5md.dll already initialized`

Urutan fix yang aman:

```powershell
conda activate prtreid_train
conda install intel-openmp mkl mkl-service -y
```

Jika masih muncul, workaround sementara (tidak direkomendasikan jangka panjang):

```powershell
setx KMP_DUPLICATE_LIB_OK TRUE
```

Buka terminal baru setelah `setx`.

### Error saat `pip install -e third_party/prtreid` butuh MSVC

Install:
- Visual Studio Build Tools 2019/2022
- workload: `Desktop development with C++`

Lalu ulang:

```powershell
pip install -e .\third_party\prtreid
```

Kalau tidak mau install MSVC, tetap bisa jalan karena fallback source sudah aktif.

### Cek GPU terbaca

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count())"
```

## 10) Config yang Sering Diubah

- `configs/prtreid/base.yaml`
- `configs/prtreid/profiles/multitask_soccernet_like.yaml`

Parameter utama:
- `train.max_epoch`
- `train.batch_size`
- `test.batch_size`
- `sampler.num_instances`
- `model.load_weights`
- `model.bpbreid.test_embeddings`

## 11) Default Model Paths

- HRNet backbone weights:
  - `C:\Programming\PRTReid Training\model\hrnetv2_w32_imagenet_pretrained.pth`
- PRTReid pretrained Soccernet:
  - `C:\Programming\PRTReid Training\model\prtreid-soccernet-baseline.pth.tar`

## 12) Dataset Labeling Guide (CVAT)

Panduan detail CVAT + converter script:

- `docs/dataset_cvat_guide.md`
