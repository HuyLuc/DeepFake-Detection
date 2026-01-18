# 📊 Nâng cấp Xử lý Dữ liệu Đầu vào cho DeepFake Detection

## 🎯 Tổng quan

Tài liệu này mô tả các nâng cấp về xử lý dữ liệu đầu vào để cải thiện hiệu suất mô hình DeepFake Detection. Các thay đổi tập trung vào 3 khía cạnh chính:

1. **Tăng độ phân giải đầu vào** từ 224x224 lên 380x380
2. **Data Augmentation chuyên biệt** cho bài toán Deepfake
3. **Oversampling** để cân bằng dữ liệu

---

## 🔍 Chi tiết các thay đổi

### 1. Tăng độ phân giải đầu vào (224x224 → 380x380)

#### ✅ Lý do
- **EfficientNet-B4** được thiết kế tối ưu cho kích thước **380x380**
- Độ phân giải cao hơn giúp model nhìn rõ các chi tiết nhỏ (artifacts) tại:
  - Mép da (skin boundary)
  - Răng (teeth)
  - Mắt (eyes)
  - Các vùng thường để lại dấu vết Deepfake

#### ⚙️ Thay đổi trong code

**File: `configs/config.py`**
```python
# Trước
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # GPU có thể xử lý được
NUM_EPOCHS = 7

# Sau
IMAGE_SIZE = (380, 380)  # Tối ưu cho EfficientNet-B4
BATCH_SIZE = 8           # Giảm để tránh OOM với resolution cao
NUM_EPOCHS = 10          # Tăng vì cần nhiều thời gian hơn
```

#### ⚠️ Lưu ý
- **VRAM tăng**: Resolution cao hơn → cần nhiều VRAM hơn
- **Giải pháp**: Giảm `BATCH_SIZE` từ 16 → 8
- **Mixed Precision**: BẮT BUỘC bật để tiết kiệm VRAM

---

### 2. Data Augmentation chuyên biệt cho Deepfake

#### ✅ Lý do
Deepfake có các đặc điểm riêng biệt cần augmentation chuyên biệt:

1. **JPEG Compression Artifacts**
   - Deepfake thường bị mờ và xuất hiện artifacts khi nén
   - Mô phỏng: Giảm chất lượng JPEG (quality 30-95)

2. **Gaussian Noise**
   - Mô phỏng camera chất lượng thấp
   - Mô phỏng điều kiện ánh sáng xấu

3. **Adaptive Gaussian Blur**
   - Deepfake thường bị mất nét (blurry)
   - Mô phỏng video chất lượng thấp

4. **Face Cutout**
   - Xóa ngẫu nhiên một phần nhỏ trên khuôn mặt
   - Giúp model học từ nhiều features khác nhau
   - Tránh overfitting vào một vùng cụ thể (chỉ nhìn mắt hoặc miệng)

#### ⚙️ Thay đổi trong code

**File mới: `src/data_processing/deepfake_augmentation.py`**

Module này cung cấp:

- `JPEGCompression`: Compression artifacts (p=0.5, quality 30-95)
- `AdaptiveGaussianNoise`: Gaussian noise (p=0.3, std 0.01-0.05)
- `AdaptiveGaussianBlur`: Gaussian blur (p=0.2, sigma 0.1-1.5)
- `FaceCutout`: Random cutout (p=0.3, max 15% kích thước ảnh)
- `MixedDeepfakeAugmentation`: Kết hợp tất cả các phép trên
- `get_deepfake_train_transforms()`: Transform pipeline cho training
- `get_deepfake_val_transforms()`: Transform pipeline cho validation

**File: `configs/config.py`**
```python
# Cấu hình Deepfake Augmentation
USE_DEEPFAKE_AUGMENTATION = True  # Bật/tắt
ENABLE_COMPRESSION_AUG = True     # JPEG compression
ENABLE_NOISE_AUG = True           # Gaussian noise
ENABLE_BLUR_AUG = True            # Blur
ENABLE_CUTOUT_AUG = True          # Face cutout
```

**File: `src/training/train.py`**
```python
# Trước: Augmentation cơ bản
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(...),
        ...
    ])
}

# Sau: Deepfake-specific Augmentation
from src.data_processing.deepfake_augmentation import (
    get_deepfake_train_transforms,
    get_deepfake_val_transforms
)

data_transforms = {
    'train': get_deepfake_train_transforms(
        image_size=config.IMAGE_SIZE,
        use_deepfake_aug=True
    ),
    'val': get_deepfake_val_transforms(image_size=config.IMAGE_SIZE)
}
```

#### 📊 Minh họa Augmentation Pipeline

```
Input Image (380x380)
    ↓
Resize & Geometric Transforms
    ↓
[50%] JPEG Compression (quality 30-95)
    ↓
[30%] Gaussian Noise (std 0.01-0.05)
    ↓
[20%] Gaussian Blur (sigma 0.1-1.5)
    ↓
[30%] Face Cutout (max 15% size)
    ↓
ColorJitter & Standard Augmentation
    ↓
ToTensor & Normalize
    ↓
[5%] Random Erasing
    ↓
Output Tensor
```

---

### 3. Oversampling để cân bằng dữ liệu

#### ✅ Lý do
- Dataset thường bị **lệch** (imbalanced): nhiều FAKE hơn REAL
- **Class weights** chỉ điều chỉnh loss → chưa đủ
- **Oversampling** cho model nhìn thấy lớp REAL nhiều hơn → giảm **False Positive**

#### ⚙️ Thay đổi trong code

**File mới: `src/training/balanced_dataset.py`**

Module này cung cấp:

- `OversampledDataset`: Wrapper dataset với oversampling
- `create_weighted_sampler()`: Tạo WeightedRandomSampler
- `get_balanced_dataloader()`: Tạo balanced DataLoader

**Hai phương pháp cân bằng dữ liệu:**

1. **Oversampling** (Dataset wrapper)
   - Tạo dataset mới với các mẫu được lặp lại
   - Lớp thiểu số (REAL) được oversample theo `OVERSAMPLE_RATIO`

2. **WeightedRandomSampler** (Sampler)
   - Không tạo dataset mới
   - Dùng sampling weights để tăng xác suất chọn lớp thiểu số

**File: `configs/config.py`**
```python
# Cấu hình Oversampling
USE_OVERSAMPLING = True              # Bật/tắt
OVERSAMPLING_METHOD = 'oversampling' # 'oversampling' hoặc 'weighted_sampler'
OVERSAMPLE_RATIO = 1.3               # Tỷ lệ oversample
                                     # 1.0 = cân bằng hoàn toàn
                                     # 1.3 = REAL có 1.3x số mẫu của FAKE
```

**File: `src/training/train.py`**
```python
# Trước: DataLoader thông thường
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Sau: Balanced DataLoader
from src.training.balanced_dataset import get_balanced_dataloader

train_loader = get_balanced_dataloader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=actual_workers,
    pin_memory=pin_memory_setting,
    method='oversampling',      # hoặc 'weighted_sampler'
    oversample_ratio=1.3
)
```

#### 📊 Ví dụ Oversampling

**Dataset gốc:**
- FAKE: 10,000 mẫu
- REAL: 6,000 mẫu
- Tỷ lệ: 1.67:1 (mất cân bằng)

**Sau oversampling với ratio=1.3:**
- FAKE: 10,000 mẫu (giữ nguyên)
- REAL: 13,000 mẫu (oversample: 6000 × 2.17 ≈ 13000)
- Tỷ lệ: 0.77:1 (REAL nhiều hơn FAKE một chút)
- **Lợi ích**: Model nhìn REAL nhiều hơn → giảm False Positive (nhầm REAL thành FAKE)

---

## 🚀 Cách sử dụng

### Bật/tắt các tính năng

Chỉnh sửa `configs/config.py`:

```python
# 1. Độ phân giải (luôn bật, không thể tắt)
IMAGE_SIZE = (380, 380)  # hoặc (224, 224) nếu muốn về cũ

# 2. Deepfake Augmentation
USE_DEEPFAKE_AUGMENTATION = True   # True để bật, False để tắt

# 3. Oversampling
USE_OVERSAMPLING = True            # True để bật, False để dùng class weights
OVERSAMPLE_RATIO = 1.3             # Điều chỉnh tỷ lệ (khuyến nghị: 1.2-1.5)
```

### Chạy training

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Chạy training
python main.py train
```

### Xem kết quả

Training logs sẽ in ra:

```
--- 🎨 Đang thiết lập Data Augmentation ---
✅ Sử dụng Deepfake-specific Augmentation:
   - JPEG Compression (mô phỏng compression artifacts)
   - Gaussian Noise (mô phỏng camera chất lượng thấp)
   - Adaptive Blur (mô phỏng video mất nét)
   - Face Cutout (khuyến khích model học nhiều features)

--- ⚖️ Đang thiết lập Data Balancing ---
✅ Sử dụng Data Balancing:
   - Method: oversampling
   - Oversample ratio: 1.3

📊 Phân bố dữ liệu gốc:
   FAKE: 10000 mẫu
   REAL: 6000 mẫu

📊 Phân bố dữ liệu sau oversampling (ratio=1.3):
   FAKE: 10000 mẫu
   REAL: 13000 mẫu

📊 Kích thước dataset:
   Tập huấn luyện: 16000 mẫu (gốc)
   Tập huấn luyện: 23000 mẫu (sau oversampling)
   Tập kiểm định: 4000 mẫu
```

---

## 📈 Kỳ vọng cải thiện

### 1. Độ chính xác tổng thể (Overall Accuracy)
- **Trước**: ~85-90%
- **Sau**: ~90-95% (kỳ vọng +3-5%)

### 2. False Positive Rate (FPR)
- **Trước**: ~15-20% (nhầm REAL thành FAKE)
- **Sau**: ~5-10% (giảm nhờ oversampling REAL)

### 3. True Positive Rate (TPR - Recall)
- **Trước**: ~90-95%
- **Sau**: ~93-97% (tăng nhờ augmentation tốt hơn)

### 4. Robustness (Khả năng tổng quát)
- Model sẽ robust hơn với:
  - Video bị nén (compression)
  - Video chất lượng thấp (noise, blur)
  - Nhiều loại Deepfake khác nhau

---

## ⚠️ Lưu ý quan trọng

### 1. VRAM & Training Time

**Với resolution 380x380:**
- VRAM cần: ~2-3GB (với BATCH_SIZE=8, MIXED_PRECISION=True)
- Training time: Tăng ~30-50% so với 224x224
- Mỗi epoch: ~15-20 phút (thay vì ~10 phút)

**Khuyến nghị:**
- GPU >= 2GB VRAM
- Nếu GPU < 2GB: Giảm `BATCH_SIZE` xuống 4 hoặc quay về IMAGE_SIZE=(224, 224)

### 2. Checkpoint cũ

**⚠️ CẢNH BÁO**: Checkpoint huấn luyện với 224x224 **KHÔNG tương thích** với 380x380!

**Giải pháp:**
1. **Huấn luyện từ đầu** (khuyến nghị)
2. Hoặc **Fine-tune** từ pretrained weights (chậm hơn nhưng có thể giữ một phần kiến thức)

### 3. Preprocessing

**KHÔNG cần chạy lại preprocessing!**
- Preprocessing chỉ crop khuôn mặt, không resize
- Resize được thực hiện trong transforms khi training
- Tiết kiệm thời gian và dung lượng đĩa

---

## 🔧 Troubleshooting

### Lỗi: Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm `BATCH_SIZE` trong `configs/config.py`:
   ```python
   BATCH_SIZE = 4  # Giảm từ 8 xuống 4
   ```

2. Hoặc giảm `IMAGE_SIZE`:
   ```python
   IMAGE_SIZE = (224, 224)  # Quay về cũ
   ```

### Lỗi: Import Module

```
ModuleNotFoundError: No module named 'src.data_processing.deepfake_augmentation'
```

**Giải pháp:**
```bash
# Đảm bảo đang ở thư mục gốc dự án
cd f:\DoAn\DeepFake-Detection

# Kiểm tra file tồn tại
dir src\data_processing\deepfake_augmentation.py
```

### Training chậm

**Nguyên nhân:**
- Resolution cao + Augmentation phức tạp = Chậm hơn

**Giải pháp:**
1. Giảm `NUM_WORKERS` nếu CPU yếu:
   ```python
   NUM_WORKERS = 2  # Giảm từ 4 xuống 2
   ```

2. Tắt một số augmentation:
   ```python
   ENABLE_COMPRESSION_AUG = False  # Tắt compression (tốn thời gian nhất)
   ```

---

## 📚 Tài liệu tham khảo

1. **EfficientNet Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
2. **Data Augmentation for Deepfake**: [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
3. **Imbalanced Learning**: [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381)

---

## 📝 Changelog

**Phiên bản 2.0.0** (2026-01-18)
- ✅ Tăng IMAGE_SIZE từ 224x224 lên 380x380
- ✅ Thêm Deepfake-specific Augmentation (compression, noise, blur, cutout)
- ✅ Implement Oversampling cho lớp REAL
- ✅ Tối ưu BATCH_SIZE và NUM_EPOCHS cho resolution mới
- ✅ Cập nhật documentation

**Phiên bản 1.0.0** (Trước đó)
- Sử dụng IMAGE_SIZE = 224x224
- Augmentation cơ bản (flip, rotation, color jitter)
- Class weights để xử lý imbalance

---

## 🤝 Contributing

Nếu bạn muốn đóng góp thêm augmentation hoặc cải thiện:

1. Thêm augmentation mới vào `src/data_processing/deepfake_augmentation.py`
2. Cập nhật config trong `configs/config.py`
3. Test kỹ để đảm bảo không làm chậm training quá nhiều
4. Cập nhật documentation này

---

**Tác giả**: HuyLuc  
**Ngày cập nhật**: 2026-01-18
