# 📓 Google Colab Notebook - Code sẵn sàng

Copy các cell code sau vào Google Colab notebook:

---

## Cell 1: Mount Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Kiểm tra
!ls /content/drive/MyDrive/
```

---

## Cell 2: Giải nén project từ Drive

```python
# Copy project từ Drive
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
    
    # Kiểm tra
    if os.path.exists(colab_project_path):
        print(f"✅ Project đã sẵn sàng tại: {colab_project_path}")
    else:
        print("⚠️ Không tìm thấy project sau khi giải nén")
else:
    print("⚠️ Không tìm thấy DeepFake-Detection.zip trên Drive")
    print("   Vui lòng upload file zip lên Drive trước!")
```

---

## Cell 3: Cài đặt dependencies

```python
# Cài đặt packages
!pip install -q torch torchvision timm opencv-python mediapipe Flask tqdm scikit-learn numpy seaborn matplotlib psutil pandas

# Kiểm tra cài đặt
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ Không có GPU. Vui lòng bật GPU trong Runtime settings!")
```

---

## Cell 4: Cấu hình cho Colab

```python
# Cấu hình cho Colab
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

---

## Cell 5: Kiểm tra cấu hình

```python
# Kiểm tra cấu hình
import sys
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
if hasattr(config, 'DRIVE_CHECKPOINT_DIR'):
    print(f"   Drive checkpoint dir: {config.DRIVE_CHECKPOINT_DIR}")
if hasattr(config, 'DRIVE_LOG_DIR'):
    print(f"   Drive log dir: {config.DRIVE_LOG_DIR}")

# Kiểm tra Drive đã mount chưa
if os.path.exists('/content/drive'):
    print("\n✅ Google Drive đã được mount")
else:
    print("\n⚠️ Google Drive chưa được mount!")

# Kiểm tra processed_data
processed_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'train')
if os.path.exists(processed_data_path):
    import glob
    train_files = len(glob.glob(os.path.join(processed_data_path, '**', '*.png'), recursive=True))
    print(f"\n✅ Processed data: {train_files} images trong train set")
else:
    print(f"\n⚠️ Chưa có processed_data. Cần chạy preprocessing trước!")
```

---

## Cell 6: Tải checkpoint từ Drive (nếu có)

```python
# Tải checkpoint từ Drive (nếu muốn tiếp tục training)
import sys
sys.path.insert(0, '/content/DeepFake-Detection')
from colab_helper import sync_checkpoint_from_drive

sync_checkpoint_from_drive()
```

---

## Cell 7: Chạy Training

```python
# Chạy training
import sys
import os

sys.path.insert(0, '/content/DeepFake-Detection')
os.chdir('/content/DeepFake-Detection')

# Chạy training
!python main.py train
```

---

## Cell 8: Kiểm tra kết quả trên Drive

```python
# Kiểm tra files đã sync vào Drive
import os

drive_checkpoint = '/content/drive/MyDrive/DeepFake-Detection/saved_models'
drive_logs = '/content/drive/MyDrive/DeepFake-Detection/evaluation_results'

print("=" * 60)
print("📁 CHECKPOINTS TRÊN DRIVE")
print("=" * 60)
if os.path.exists(drive_checkpoint):
    files = os.listdir(drive_checkpoint)
    if files:
        for file in files:
            file_path = os.path.join(drive_checkpoint, file)
            size = os.path.getsize(file_path) / (1024**2)
            print(f"   ✅ {file} ({size:.2f} MB)")
    else:
        print("   ⚠️ Chưa có checkpoint")
else:
    print("   ⚠️ Thư mục chưa tồn tại")

print("\n" + "=" * 60)
print("📁 LOGS TRÊN DRIVE")
print("=" * 60)
if os.path.exists(drive_logs):
    files = [f for f in os.listdir(drive_logs) if f.endswith(('.log', '.csv'))]
    if files:
        for file in files:
            file_path = os.path.join(drive_logs, file)
            size = os.path.getsize(file_path) / (1024**2)
            print(f"   ✅ {file} ({size:.2f} MB)")
    else:
        print("   ⚠️ Chưa có log files")
else:
    print("   ⚠️ Thư mục chưa tồn tại")
```

---

## Cell 9: Sync thủ công (nếu cần)

```python
# Sync thủ công tất cả vào Drive
import sys
sys.path.insert(0, '/content/DeepFake-Detection')
from colab_helper import sync_all_to_drive

sync_all_to_drive()
```

---

## 📝 Lưu ý khi sử dụng

1. **Chạy các cell theo thứ tự** từ 1 đến 7
2. **Cell 6** chỉ cần chạy nếu muốn tiếp tục training từ checkpoint cũ
3. **Cell 8 và 9** dùng để kiểm tra và sync thủ công
4. **Training sẽ tự động sync** vào Drive sau mỗi epoch (nếu đã cấu hình)

---

## 🔄 Khi runtime disconnect

Nếu runtime bị disconnect:

1. **Restart runtime** và mount lại Drive (Cell 1)
2. **Chạy lại Cell 2-5** để setup lại
3. **Chạy Cell 6** để tải checkpoint từ Drive
4. **Chạy Cell 7** để tiếp tục training

---

**Chúc bạn training thành công! 🚀**


