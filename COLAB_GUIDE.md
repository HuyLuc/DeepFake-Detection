# 🚀 Hướng dẫn Training trên Google Colab với Google Drive

Hướng dẫn chi tiết để training model trên Google Colab và tự động lưu checkpoint, log, model vào Google Drive.

---

## ⚡ BẮT ĐẦU NHANH (Nếu bạn đã upload project lên Drive)

**Nếu bạn đã:**
- ✅ Upload toàn bộ project lên Google Drive
- ✅ Mount Google Drive trong Colab
- ✅ Bật GPU trong Colab

**→ Bỏ qua phần "Chuẩn bị" và bắt đầu từ [Cấu hình Google Drive](#⚙️-cấu-hình-google-drive)**

**Các bước tiếp theo:**
1. [Cell 2: Copy project từ Drive](#bước-1-copy-project-từ-drive-vào-colab)
2. [Cell 3: Cài đặt dependencies](#bước-2-cài-đặt-dependencies)
3. [Cell 4: Cấu hình cho Colab](#bước-3-cấu-hình-cho-colab)
4. [Cell 5: Tải checkpoint (nếu có)](#bước-4-tải-checkpoint-từ-drive-nếu-muốn-tiếp-tục-training)
5. [Cell 6: Kiểm tra cấu hình](#bước-5-kiểm-tra-cấu-hình-và-dữ-liệu)
6. [Cell 7: Bắt đầu Training](#bước-6-bắt-đầu-training)

---

## 📋 Mục lục

1. [Chuẩn bị](#chuẩn-bị)
2. [Setup Google Colab](#setup-google-colab)
3. [Cấu hình Google Drive](#cấu-hình-google-drive)
4. [Chạy Training](#chạy-training)
5. [Quản lý Checkpoint và Logs](#quản-lý-checkpoint-và-logs)
6. [Lưu ý quan trọng](#lưu-ý-quan-trọng)
7. [Troubleshooting](#troubleshooting)
8. [Đề xuất tối ưu](#đề-xuất-tối-ưu)

---

## 🎯 Chuẩn bị

### Bước 1: Chuẩn bị dữ liệu trên máy local

Bạn cần có sẵn:
- ✅ **Code project** (toàn bộ thư mục `DeepFake-Detection`)
- ✅ **Dataset đã processed** (`processed_data/` folder) - **QUAN TRỌNG!**
- ✅ **Checkpoint hiện tại** (nếu muốn tiếp tục training)

### Bước 2: Upload lên Google Drive

1. **Nén project thành file zip:**
   ```bash
   # Trên Windows (PowerShell)
   Compress-Archive -Path .\* -DestinationPath DeepFake-Detection.zip
   
   # Hoặc dùng WinRAR/7-Zip để nén thủ công
   ```

2. **Upload lên Google Drive:**
   - Truy cập [Google Drive](https://drive.google.com/)
   - Tạo thư mục mới: `DeepFake-Detection`
   - Upload file `DeepFake-Detection.zip` vào thư mục này
   - **LƯU Ý:** Nếu dataset `processed_data/` quá lớn (>15GB), có thể upload riêng hoặc dùng Google Drive API

3. **Cấu trúc trên Drive:**
   ```
   MyDrive/
   └── DeepFake-Detection/
       ├── DeepFake-Detection.zip  (code project)
       ├── saved_models/           (sẽ tự động tạo)
       └── evaluation_results/    (sẽ tự động tạo)
   ```

---

## 🖥️ Setup Google Colab

### Bước 1: Tạo Notebook mới

1. Truy cập [Google Colab](https://colab.research.google.com/)
2. Tạo notebook mới: `File` → `New notebook`
3. Đổi tên: `DeepFake Training`

### Bước 2: Bật GPU

1. `Runtime` → `Change runtime type`
2. Chọn:
   - **Hardware accelerator:** `GPU`
   - **GPU type:** `T4` (free) hoặc `V100` (Colab Pro)
3. Click `Save`

### Bước 3: Mount Google Drive

Chạy cell đầu tiên:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Kiểm tra
!ls /content/drive/MyDrive/
```

**LƯU Ý:** 
- Lần đầu sẽ yêu cầu xác thực Google
- Click link → Chọn tài khoản → Copy mã → Paste vào ô
- Mount sẽ tự động reconnect khi runtime restart

---

## ⚙️ Cấu hình Google Drive

### Bước 1: Copy project từ Drive vào Colab

**Nếu bạn đã upload dạng ZIP:**

```python
# Cell 2: Giải nén project từ Drive
import zipfile
import os

drive_project_path = '/content/drive/MyDrive/DeepFake-Detection'
colab_project_path = '/content/DeepFake-Detection'

# Giải nén project
zip_path = f'{drive_project_path}/DeepFake-Detection.zip'
if os.path.exists(zip_path):
    print("📦 Đang giải nén project...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("✅ Đã giải nén project")
else:
    print("⚠️ Không tìm thấy DeepFake-Detection.zip trên Drive")
    print("   Vui lòng upload file zip lên Drive trước!")
```

**Nếu bạn đã upload trực tiếp folder (không zip):**

```python
# Cell 2: Copy project từ Drive
import shutil
import os

drive_project_path = '/content/drive/MyDrive/DeepFake-Detection'
colab_project_path = '/content/DeepFake-Detection'

# Copy toàn bộ project từ Drive
if os.path.exists(drive_project_path):
    if os.path.exists(colab_project_path):
        print("⚠️ Project đã tồn tại, đang xóa...")
        shutil.rmtree(colab_project_path)
    
    print("📦 Đang copy project từ Drive...")
    shutil.copytree(drive_project_path, colab_project_path)
    print("✅ Đã copy project")
else:
    print("⚠️ Không tìm thấy project trên Drive")
    print(f"   Kiểm tra đường dẫn: {drive_project_path}")
```

### Bước 2: Cài đặt dependencies

```python
# Cell 3: Cài đặt packages
!pip install -q torch torchvision timm opencv-python mediapipe Flask tqdm scikit-learn numpy seaborn matplotlib psutil pandas

# Kiểm tra cài đặt
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### Bước 3: Cấu hình cho Colab

```python
# Cell 4: Cấu hình cho Colab
import shutil
import os

project_path = '/content/DeepFake-Detection'

# Backup config cũ (nếu có)
if os.path.exists(f'{project_path}/configs/config.py'):
    if not os.path.exists(f'{project_path}/configs/config_local_backup.py'):
        shutil.copy(f'{project_path}/configs/config.py',
                    f'{project_path}/configs/config_local_backup.py')
        print("✅ Đã backup config local")

# Copy config_colab.py thành config.py
if os.path.exists(f'{project_path}/configs/config_colab.py'):
    shutil.copy(f'{project_path}/configs/config_colab.py',
                f'{project_path}/configs/config.py')
    print("✅ Đã cấu hình cho Colab")
else:
    print("❌ Không tìm thấy config_colab.py")
```

### Bước 4: Tải checkpoint từ Drive (nếu muốn tiếp tục training)

**Nếu bạn đã train được 2 epochs và muốn tiếp tục:**

```python
# Cell 5: Tải checkpoint từ Drive
import sys
import os
import shutil

sys.path.insert(0, '/content/DeepFake-Detection')

# Đường dẫn checkpoint trên Drive
drive_checkpoint_dir = '/content/drive/MyDrive/DeepFake-Detection/saved_models'
local_checkpoint_dir = '/content/DeepFake-Detection/saved_models'

# Tạo thư mục local nếu chưa có
os.makedirs(local_checkpoint_dir, exist_ok=True)

# Tải checkpoint từ Drive
if os.path.exists(drive_checkpoint_dir):
    checkpoint_files = ['checkpoint.pth.tar', 'model_best.pth.tar']
    found = False
    
    for file in checkpoint_files:
        drive_file = os.path.join(drive_checkpoint_dir, file)
        local_file = os.path.join(local_checkpoint_dir, file)
        
        if os.path.exists(drive_file):
            shutil.copy2(drive_file, local_file)
            size_mb = os.path.getsize(local_file) / (1024**2)
            print(f"✅ Đã tải {file} từ Drive ({size_mb:.2f} MB)")
            found = True
    
    if found:
        print("\n✅ Đã tải checkpoint. Training sẽ tiếp tục từ epoch đã lưu!")
    else:
        print("\n⚠️ Không tìm thấy checkpoint trên Drive. Training sẽ bắt đầu từ đầu.")
else:
    print("⚠️ Thư mục checkpoint trên Drive chưa tồn tại. Training sẽ bắt đầu từ đầu.")
```

**Nếu bạn muốn training từ đầu (bỏ qua bước này):**

```python
# Cell 5: Bỏ qua - Training từ đầu
print("Training sẽ bắt đầu từ epoch 0")
```

### Bước 5: Kiểm tra cấu hình và dữ liệu

```python
# Cell 6: Kiểm tra cấu hình và dữ liệu
import sys
import os
import glob

sys.path.insert(0, '/content/DeepFake-Detection')

from configs import config

print("=" * 60)
print("📋 KIỂM TRA CẤU HÌNH")
print("=" * 60)
print(f"Base directory: {config.BASE_DIR}")
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Model: {config.MODEL_NAME}")
print(f"Epochs: {config.NUM_EPOCHS}")

print(f"\n💾 Google Drive:")
print(f"   Use Drive for checkpoints: {getattr(config, 'USE_DRIVE_FOR_CHECKPOINTS', False)}")
print(f"   Use Drive for logs: {getattr(config, 'USE_DRIVE_FOR_LOGS', False)}")
print(f"   Auto sync every epoch: {getattr(config, 'AUTO_SYNC_EVERY_EPOCH', False)}")

# Kiểm tra Drive đã mount chưa
if os.path.exists('/content/drive'):
    print("\n✅ Google Drive đã được mount")
else:
    print("\n❌ Google Drive chưa được mount! Vui lòng mount lại.")

# Kiểm tra processed_data
print("\n📁 KIỂM TRA DỮ LIỆU:")
processed_data_path = config.PROCESSED_DATA_DIR
train_path = os.path.join(processed_data_path, 'train')
val_path = os.path.join(processed_data_path, 'val')

if os.path.exists(train_path):
    train_fake = len(glob.glob(os.path.join(train_path, 'FAKE', '**', '*.png'), recursive=True))
    train_real = len(glob.glob(os.path.join(train_path, 'REAL', '**', '*.png'), recursive=True))
    print(f"✅ Train set: {train_fake} FAKE, {train_real} REAL")
else:
    print(f"❌ Không tìm thấy train set tại: {train_path}")

if os.path.exists(val_path):
    val_fake = len(glob.glob(os.path.join(val_path, 'FAKE', '**', '*.png'), recursive=True))
    val_real = len(glob.glob(os.path.join(val_path, 'REAL', '**', '*.png'), recursive=True))
    print(f"✅ Val set: {val_fake} FAKE, {val_real} REAL")
else:
    print(f"❌ Không tìm thấy val set tại: {val_path}")

# Kiểm tra checkpoint
checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 'checkpoint.pth.tar')
if os.path.exists(checkpoint_path):
    print(f"\n✅ Tìm thấy checkpoint: {checkpoint_path}")
    print("   Training sẽ tiếp tục từ checkpoint này")
else:
    print(f"\n⚠️ Không tìm thấy checkpoint")
    print("   Training sẽ bắt đầu từ đầu")

print("\n" + "=" * 60)
print("✅ Kiểm tra hoàn tất!")
print("=" * 60)
```

---

## 🏃 Chạy Training

### Bước 6: Bắt đầu Training

**Sau khi đã hoàn tất các bước trên, chạy cell này để bắt đầu training:**

```python
# Cell 7: BẮT ĐẦU TRAINING
import sys
import os

# Thêm project vào Python path
sys.path.insert(0, '/content/DeepFake-Detection')

# Chuyển vào thư mục project
os.chdir('/content/DeepFake-Detection')

# Chạy training
print("=" * 60)
print("🚀 BẮT ĐẦU TRAINING")
print("=" * 60)
print("Training sẽ tự động:")
print("  ✅ Lưu checkpoint sau mỗi epoch vào Drive")
print("  ✅ Lưu best model vào Drive khi có improvement")
print("  ✅ Sync logs vào Drive sau mỗi epoch")
print("=" * 60)
print()

# Chạy training
!python main.py train
```

**HOẶC chạy trực tiếp từ module:**

```python
# Cell 7: BẮT ĐẦU TRAINING (Alternative)
import sys
import os

sys.path.insert(0, '/content/DeepFake-Detection')
os.chdir('/content/DeepFake-Detection')

from src.training.train import run_training

print("=" * 60)
print("🚀 BẮT ĐẦU TRAINING")
print("=" * 60)
print("Training sẽ tự động:")
print("  ✅ Lưu checkpoint sau mỗi epoch vào Drive")
print("  ✅ Lưu best model vào Drive khi có improvement")
print("  ✅ Sync logs vào Drive sau mỗi epoch")
print("=" * 60)
print()

# Chạy training
run_training()
```

### Theo dõi Training

**Trong khi training, bạn có thể:**

1. **Xem progress trong Colab:**
   - Progress bar hiển thị loss và accuracy real-time
   - Thông tin chi tiết cho mỗi epoch

2. **Kiểm tra log files trên Drive:**
   - `MyDrive/DeepFake-Detection/evaluation_results/training.log` - Log chi tiết
   - `MyDrive/DeepFake-Detection/evaluation_results/training_log.csv` - Metrics CSV

3. **Kiểm tra checkpoint trên Drive:**
   - `MyDrive/DeepFake-Detection/saved_models/checkpoint.pth.tar` - Checkpoint mới nhất
   - `MyDrive/DeepFake-Detection/saved_models/model_best.pth.tar` - Model tốt nhất

**Lưu ý:** Tất cả files sẽ tự động sync vào Drive sau mỗi epoch, không cần làm gì thêm!

---

## 💾 Quản lý Checkpoint và Logs

### Tự động sync (Đã cấu hình sẵn)

Với cấu hình hiện tại, hệ thống sẽ **TỰ ĐỘNG**:
- ✅ Lưu checkpoint sau mỗi epoch vào Drive
- ✅ Lưu best model vào Drive ngay khi có improvement
- ✅ Sync log files (`training.log`, `training_log.csv`) sau mỗi epoch

### Kiểm tra files trên Drive

```python
# Cell: Kiểm tra files đã sync
import os

drive_checkpoint = '/content/drive/MyDrive/DeepFake-Detection/saved_models'
drive_logs = '/content/drive/MyDrive/DeepFake-Detection/evaluation_results'

print("📁 Checkpoints trên Drive:")
if os.path.exists(drive_checkpoint):
    for file in os.listdir(drive_checkpoint):
        size = os.path.getsize(os.path.join(drive_checkpoint, file)) / (1024**2)
        print(f"   - {file} ({size:.2f} MB)")
else:
    print("   ⚠️ Chưa có checkpoint")

print("\n📁 Logs trên Drive:")
if os.path.exists(drive_logs):
    for file in os.listdir(drive_logs):
        if file.endswith(('.log', '.csv')):
            size = os.path.getsize(os.path.join(drive_logs, file)) / (1024**2)
            print(f"   - {file} ({size:.2f} MB)")
else:
    print("   ⚠️ Chưa có log")
```

### Sync thủ công (nếu cần)

```python
# Cell: Sync thủ công
import sys
sys.path.insert(0, '/content/DeepFake-Detection')
from colab_helper import sync_checkpoint_to_drive

sync_checkpoint_to_drive()
```

### Tải checkpoint từ Drive (khi restart runtime)

**Khi runtime bị disconnect và bạn muốn tiếp tục training:**

```python
# Cell: Tải checkpoint từ Drive (khi restart runtime)
import sys
import os
import shutil

sys.path.insert(0, '/content/DeepFake-Detection')

# Đường dẫn
drive_checkpoint_dir = '/content/drive/MyDrive/DeepFake-Detection/saved_models'
local_checkpoint_dir = '/content/DeepFake-Detection/saved_models'

os.makedirs(local_checkpoint_dir, exist_ok=True)

# Tải checkpoint từ Drive
if os.path.exists(drive_checkpoint_dir):
    checkpoint_files = ['checkpoint.pth.tar', 'model_best.pth.tar']
    found = False
    
    for file in checkpoint_files:
        drive_file = os.path.join(drive_checkpoint_dir, file)
        local_file = os.path.join(local_checkpoint_dir, file)
        
        if os.path.exists(drive_file):
            shutil.copy2(drive_file, local_file)
            size_mb = os.path.getsize(local_file) / (1024**2)
            print(f"✅ Đã tải {file} từ Drive ({size_mb:.2f} MB)")
            found = True
    
    if found:
        print("\n✅ Đã tải checkpoint. Có thể tiếp tục training!")
    else:
        print("\n⚠️ Không tìm thấy checkpoint trên Drive.")
else:
    print("⚠️ Thư mục checkpoint trên Drive chưa tồn tại.")
```

---

## ⚠️ Lưu ý quan trọng

### 1. Runtime Disconnect

**Vấn đề:** Colab free có thể disconnect sau 12 giờ không hoạt động.

**Giải pháp:**
- ✅ **Luôn lưu checkpoint vào Drive** (đã tự động)
- ✅ Bật "Keep runtime alive" trong Colab Pro
- ✅ Hoặc dùng extension để tự động click (không khuyến nghị)

### 2. Dung lượng Google Drive

**Vấn đề:** Checkpoint và logs có thể tốn nhiều dung lượng.

**Giải pháp:**
- ✅ Xóa checkpoint cũ nếu không cần
- ✅ Chỉ giữ `model_best.pth.tar` và `checkpoint.pth.tar` mới nhất
- ✅ Nén logs cũ nếu cần

### 3. Upload dataset lớn

**Vấn đề:** Dataset `processed_data/` có thể rất lớn (>10GB).

**Giải pháp:**
- ✅ Upload trực tiếp lên Drive (mất thời gian nhưng ổn định)
- ✅ Hoặc dùng Google Drive API để upload song song
- ✅ Hoặc chia nhỏ dataset nếu có thể

### 4. Memory và VRAM

**Vấn đề:** Colab T4 có 16GB VRAM, nhưng có thể hết nếu batch size quá lớn.

**Giải pháp:**
- ✅ Giảm `BATCH_SIZE` trong `config_colab.py` nếu gặp OOM
- ✅ Bật `MIXED_PRECISION = True` (đã bật sẵn)
- ✅ Giảm `NUM_WORKERS` nếu RAM hết

### 5. Tốc độ training

**Colab free:**
- GPU: T4 (16GB VRAM)
- Runtime: 12 giờ max
- Tốc độ: ~15-30 phút/epoch (tùy dataset)

**Colab Pro:**
- GPU: V100 hoặc A100 (mạnh hơn)
- Runtime: 24 giờ max
- Tốc độ: ~10-20 phút/epoch

---

## 🔧 Troubleshooting

### Lỗi: "CUDA out of memory"

**Nguyên nhân:** Batch size quá lớn hoặc model quá nặng.

**Giải pháp:**
```python
# Giảm batch size trong config_colab.py
BATCH_SIZE = 16  # Giảm từ 32 xuống 16
```

### Lỗi: "Runtime disconnected"

**Nguyên nhân:** Runtime bị disconnect do không hoạt động.

**Giải pháp:**
1. Checkpoint đã được lưu vào Drive (tự động)
2. Restart runtime
3. Tải checkpoint từ Drive (dùng `sync_checkpoint_from_drive()`)
4. Tiếp tục training

### Lỗi: "Module not found"

**Nguyên nhân:** Chưa cài đặt package hoặc path sai.

**Giải pháp:**
```python
# Cài đặt lại packages
!pip install --upgrade torch torchvision timm opencv-python mediapipe

# Kiểm tra path
import sys
print(sys.path)
sys.path.insert(0, '/content/DeepFake-Detection')
```

### Lỗi: "Drive not mounted"

**Nguyên nhân:** Google Drive chưa được mount.

**Giải pháp:**
```python
# Mount lại Drive
from google.colab import drive
drive.mount('/content/drive')
```

### Lỗi: "Checkpoint not found"

**Nguyên nhân:** Checkpoint chưa được sync hoặc bị mất.

**Giải pháp:**
1. Kiểm tra trên Drive: `MyDrive/DeepFake-Detection/saved_models/`
2. Nếu có trên Drive, tải về:
   ```python
   from colab_helper import sync_checkpoint_from_drive
   sync_checkpoint_from_drive()
   ```
3. Nếu không có, bắt đầu training từ đầu

---

## 💡 Đề xuất tối ưu

### 1. Tối ưu tốc độ training

```python
# Trong config_colab.py
BATCH_SIZE = 32  # Tăng nếu VRAM đủ
NUM_WORKERS = 4  # Tối ưu cho Colab
MIXED_PRECISION = True  # Bật để tăng tốc ~2x
PREFETCH_FACTOR = 2  # Prefetch data
```

### 2. Tiết kiệm dung lượng Drive

```python
# Chỉ lưu checkpoint tốt nhất
# (Đã tự động: chỉ lưu model_best.pth.tar và checkpoint.pth.tar)
```

### 3. Monitoring training

```python
# Xem log real-time trên Drive
# File: MyDrive/DeepFake-Detection/evaluation_results/training.log
```

### 4. Backup định kỳ

```python
# Tạo script backup định kỳ (chạy sau mỗi N epochs)
# Hoặc dùng Google Drive API để tự động backup
```

### 5. Sử dụng Colab Pro (nếu có)

**Lợi ích:**
- GPU mạnh hơn (V100/A100)
- Runtime lâu hơn (24h)
- Priority cao hơn
- Tốc độ training nhanh hơn ~2-3x

---

## ✅ Checklist trước khi training

- [ ] Đã upload project lên Google Drive
- [ ] Đã mount Google Drive trong Colab
- [ ] Đã bật GPU trong Colab settings
- [ ] Đã cài đặt dependencies
- [ ] Đã copy `config_colab.py` → `config.py`
- [ ] Đã kiểm tra GPU available
- [ ] Đã có `processed_data/` folder (hoặc đã upload)
- [ ] Đã backup checkpoint cũ (nếu có)
- [ ] Đã kiểm tra cấu hình Drive sync

---

## 📊 So sánh Local vs Colab

| Tính năng | Local (CPU) | Colab (GPU Free) | Colab Pro |
|-----------|-------------|------------------|-----------|
| **Tốc độ** | ~5 giờ/epoch | ~15-30 phút/epoch | ~10-20 phút/epoch |
| **Batch size** | 2 | 32 | 32-64 |
| **GPU** | ❌ | T4 (16GB) | V100/A100 |
| **Runtime** | Không giới hạn | 12 giờ | 24 giờ |
| **Chi phí** | Miễn phí | Miễn phí | ~$10/tháng |
| **Auto-save** | ❌ | ✅ (Drive) | ✅ (Drive) |

---

## 🎉 Kết luận

Với cấu hình này, bạn có thể:
- ✅ Training nhanh hơn ~10-20 lần so với CPU local
- ✅ Tự động lưu checkpoint, log, model vào Drive
- ✅ Không lo mất dữ liệu khi runtime disconnect
- ✅ Dễ dàng tiếp tục training sau khi restart

**Chúc bạn training thành công! 🚀**

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra log file trên Drive
2. Xem phần Troubleshooting
3. Kiểm tra cấu hình trong `config_colab.py`


