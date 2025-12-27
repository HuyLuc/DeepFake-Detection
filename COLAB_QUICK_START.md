# ⚡ Quick Start - Copy & Paste vào Colab

**Nếu bạn đã upload project lên Drive và mount Drive rồi, copy các cell sau vào Colab:**

---

## Cell 1: Mount Google Drive (Đã làm rồi - Bỏ qua)

```python
# Đã mount rồi, bỏ qua
```

---

## Cell 2: Copy project từ Drive

**Nếu upload dạng ZIP:**
```python
import zipfile
import os

drive_project_path = '/content/drive/MyDrive/DeepFake-Detection'
zip_path = f'{drive_project_path}/DeepFake-Detection.zip'

if os.path.exists(zip_path):
    print("📦 Đang giải nén project...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("✅ Đã giải nén project")
else:
    print("⚠️ Không tìm thấy file zip")
```

**Nếu upload trực tiếp folder:**
```python
import shutil
import os

drive_project_path = '/content/drive/MyDrive/DeepFake-Detection'
colab_project_path = '/content/DeepFake-Detection'

if os.path.exists(drive_project_path):
    if os.path.exists(colab_project_path):
        shutil.rmtree(colab_project_path)
    print("📦 Đang copy project...")
    shutil.copytree(drive_project_path, colab_project_path)
    print("✅ Đã copy project")
else:
    print("⚠️ Không tìm thấy project trên Drive")
```

---

## Cell 3: Cài đặt packages

```python
!pip install -q torch torchvision timm opencv-python mediapipe Flask tqdm scikit-learn numpy seaborn matplotlib psutil pandas

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 4: Cấu hình cho Colab

```python
import shutil
import os

project_path = '/content/DeepFake-Detection'

# Copy config_colab.py thành config.py
if os.path.exists(f'{project_path}/configs/config_colab.py'):
    shutil.copy(f'{project_path}/configs/config_colab.py',
                f'{project_path}/configs/config.py')
    print("✅ Đã cấu hình cho Colab")
else:
    print("❌ Không tìm thấy config_colab.py")
```

---

## Cell 5: Tải checkpoint (nếu muốn tiếp tục từ epoch 2)

```python
import sys
import os
import shutil

sys.path.insert(0, '/content/DeepFake-Detection')

drive_checkpoint_dir = '/content/drive/MyDrive/DeepFake-Detection/saved_models'
local_checkpoint_dir = '/content/DeepFake-Detection/saved_models'

os.makedirs(local_checkpoint_dir, exist_ok=True)

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
        print("\n⚠️ Không tìm thấy checkpoint. Training sẽ bắt đầu từ đầu.")
else:
    print("⚠️ Thư mục checkpoint chưa tồn tại. Training sẽ bắt đầu từ đầu.")
```

---

## Cell 6: Kiểm tra cấu hình

```python
import sys
import os
import glob

sys.path.insert(0, '/content/DeepFake-Detection')

from configs import config

print("=" * 60)
print("📋 KIỂM TRA CẤU HÌNH")
print("=" * 60)
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Model: {config.MODEL_NAME}")
print(f"Epochs: {config.NUM_EPOCHS}")

# Kiểm tra dữ liệu
train_path = os.path.join(config.PROCESSED_DATA_DIR, 'train')
if os.path.exists(train_path):
    train_fake = len(glob.glob(os.path.join(train_path, 'FAKE', '**', '*.png'), recursive=True))
    train_real = len(glob.glob(os.path.join(train_path, 'REAL', '**', '*.png'), recursive=True))
    print(f"\n✅ Train set: {train_fake} FAKE, {train_real} REAL")
else:
    print(f"\n❌ Không tìm thấy train set!")

# Kiểm tra checkpoint
checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 'checkpoint.pth.tar')
if os.path.exists(checkpoint_path):
    print(f"\n✅ Tìm thấy checkpoint - Training sẽ tiếp tục")
else:
    print(f"\n⚠️ Không tìm thấy checkpoint - Training sẽ bắt đầu từ đầu")

print("=" * 60)
```

---

## Cell 7: BẮT ĐẦU TRAINING 🚀

```python
import sys
import os

sys.path.insert(0, '/content/DeepFake-Detection')
os.chdir('/content/DeepFake-Detection')

print("=" * 60)
print("🚀 BẮT ĐẦU TRAINING")
print("=" * 60)
print("Training sẽ tự động:")
print("  ✅ Lưu checkpoint sau mỗi epoch vào Drive")
print("  ✅ Lưu best model vào Drive khi có improvement")
print("  ✅ Sync logs vào Drive sau mỗi epoch")
print("=" * 60)
print()

!python main.py train
```

---

## ✅ Sau khi training

**Tất cả đã tự động lưu vào Drive:**
- Checkpoint: `MyDrive/DeepFake-Detection/saved_models/`
- Logs: `MyDrive/DeepFake-Detection/evaluation_results/`

**Không cần làm gì thêm!** 🎉

---

## 🔄 Nếu runtime disconnect

1. Restart runtime
2. Mount lại Drive (Cell 1)
3. Chạy lại Cell 2-5
4. Chạy Cell 7 để tiếp tục training (sẽ tự động load checkpoint từ Drive)

---

**Chúc bạn training thành công! 🚀**

