# ✅ Tóm tắt Nâng cấp - Xử lý Dữ liệu Đầu vào

## 🎯 Mục tiêu
Nâng cấp việc xử lý dữ liệu đầu vào để cải thiện hiệu suất mô hình DeepFake Detection.

## 📝 Các thay đổi chính

### 1. ⬆️ Tăng độ phân giải: 224x224 → 380x380
- ✅ File: `configs/config.py`
- ✅ Lý do: EfficientNet-B4 tối ưu cho 380x380
- ✅ Lợi ích: Model nhìn rõ hơn artifacts nhỏ (mép da, răng, mắt)
- ⚠️ Lưu ý: BATCH_SIZE giảm từ 16 → 8 để tránh OOM

### 2. 🎨 Data Augmentation chuyên biệt cho Deepfake
- ✅ File mới: `src/data_processing/deepfake_augmentation.py`
- ✅ Bao gồm:
  - **JPEG Compression** (p=0.5): Mô phỏng compression artifacts
  - **Gaussian Noise** (p=0.3): Mô phỏng camera chất lượng thấp
  - **Adaptive Blur** (p=0.2): Mô phỏng video mất nét
  - **Face Cutout** (p=0.3): Khuyến khích model học nhiều features

### 3. ⚖️ Oversampling để cân bằng dữ liệu
- ✅ File mới: `src/training/balanced_dataset.py`
- ✅ Oversample lớp REAL với ratio=1.3
- ✅ Lợi ích: Giảm False Positive (nhầm REAL thành FAKE)

### 4. 🔧 Cập nhật Training Pipeline
- ✅ File: `src/training/train.py`
- ✅ Sử dụng Deepfake-specific transforms
- ✅ Sử dụng balanced DataLoader
- ✅ NUM_EPOCHS tăng từ 7 → 10

### 5. 📄 Config mới
- ✅ File: `configs/config.py`
```python
# Độ phân giải
IMAGE_SIZE = (380, 380)

# Augmentation
USE_DEEPFAKE_AUGMENTATION = True
ENABLE_COMPRESSION_AUG = True
ENABLE_NOISE_AUG = True
ENABLE_BLUR_AUG = True
ENABLE_CUTOUT_AUG = True

# Oversampling
USE_OVERSAMPLING = True
OVERSAMPLING_METHOD = 'oversampling'
OVERSAMPLE_RATIO = 1.3
```

## 🚀 Cách sử dụng

### Chạy training với cấu hình mới
```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Chạy training
python main.py train
```

### Tắt một tính năng (nếu cần)
Sửa `configs/config.py`:
```python
USE_DEEPFAKE_AUGMENTATION = False  # Tắt Deepfake augmentation
# hoặc
USE_OVERSAMPLING = False           # Tắt oversampling
```

## 📊 Kỳ vọng cải thiện

| Metric | Trước | Sau (kỳ vọng) |
|--------|-------|---------------|
| Overall Accuracy | ~85-90% | ~90-95% (+3-5%) |
| False Positive Rate | ~15-20% | ~5-10% (giảm nhờ oversampling) |
| True Positive Rate | ~90-95% | ~93-97% (tăng nhờ augmentation) |

## ⚠️ Lưu ý quan trọng

### VRAM & Training Time
- **VRAM cần**: ~2-3GB (với BATCH_SIZE=8, MIXED_PRECISION=True)
- **Training time**: Tăng ~30-50% so với trước
- **Mỗi epoch**: ~15-20 phút (thay vì ~10 phút)

### Checkpoint cũ
⚠️ **CẢNH BÁO**: Checkpoint huấn luyện với 224x224 **KHÔNG tương thích** với 380x380!

**Giải pháp**: Phải huấn luyện lại từ đầu (pretrained ImageNet weights).

### Preprocessing
✅ **KHÔNG CẦN chạy lại preprocessing!**
- Preprocessing chỉ crop khuôn mặt, không resize
- Resize được thực hiện trong transforms khi training

## 🔧 Troubleshooting

### Out of Memory (OOM)
```python
# Giảm BATCH_SIZE trong configs/config.py
BATCH_SIZE = 4  # Giảm từ 8 xuống 4
```

### Training quá chậm
```python
# Tắt compression augmentation (tốn thời gian nhất)
ENABLE_COMPRESSION_AUG = False
```

## 📚 Files được thêm/sửa

### Files mới
1. `src/data_processing/deepfake_augmentation.py` - Augmentation chuyên biệt
2. `src/training/balanced_dataset.py` - Oversampling dataset
3. `NANG_CAP_DU_LIEU_DAU_VAO.md` - Tài liệu chi tiết
4. `TOM_TAT_NANG_CAP.md` - Tài liệu tóm tắt (file này)

### Files đã sửa
1. `configs/config.py` - Thêm cấu hình mới
2. `src/training/train.py` - Sử dụng augmentation và oversampling mới

### Files không đổi
- `src/data_processing/preprocess.py` - Không cần thay đổi
- `src/app/main_app.py` - Tự động dùng IMAGE_SIZE mới từ config
- `src/training/dataset.py` - Giữ nguyên
- `src/training/evaluate.py` - Giữ nguyên

## 📖 Tài liệu chi tiết
Xem file `NANG_CAP_DU_LIEU_DAU_VAO.md` để biết thêm chi tiết về:
- Lý do từng thay đổi
- Cách hoạt động của augmentation
- Công thức oversampling
- Ví dụ minh họa
- Troubleshooting chi tiết

---

**Tác giả**: HuyLuc  
**Ngày cập nhật**: 2026-01-18  
**Phiên bản**: 2.0.0
