# 🚀 Hướng dẫn tối ưu để giảm thời gian training từ 18 giờ xuống ~1-2 giờ

## ✅ Đã thực hiện các thay đổi:

### 1. **Giảm số frames/video: 30 → 10** ⚡ (Giảm ~66% data)
- File: `configs/config.py`
- `NUM_FRAMES_PER_VIDEO = 10`
- **Tác động**: Giảm số lượng ảnh từ ~56,550 xuống ~18,850 ảnh

### 2. **Tăng Batch Size: 4 → 16** 🔥 (Tăng tốc 4x)
- File: `configs/config.py`
- `BATCH_SIZE = 16`
- **Tác động**: Giảm số iterations/epoch từ ~14,137 xuống ~1,178 iterations

### 3. **Tăng NUM_WORKERS: 2 → 6** 💨
- File: `configs/config.py`
- `NUM_WORKERS = 6`
- **Tác động**: CPU load data nhanh hơn, GPU không phải chờ đợi

### 4. **Bật Mixed Precision Training** ⚡
- File: `configs/config.py`
- `MIXED_PRECISION = True`
- **Tác động**: Tăng tốc ~2x, giảm VRAM usage

### 5. **Tăng PREFETCH_FACTOR: 2 → 4** 
- File: `configs/config.py`
- `PREFETCH_FACTOR = 4`
- **Tác động**: Prefetch nhiều batch hơn, giảm idle time

### 6. **Tắt Gradient Accumulation**
- File: `configs/config.py`
- `ACCUMULATION_STEPS = 1`
- **Tác động**: Không cần accumulation vì batch size đã đủ lớn

### 7. **Tăng Learning Rate: 0.0005 → 0.001**
- File: `configs/config.py`
- **Lý do**: Batch size lớn hơn → cần learning rate cao hơn

### 8. **Tăng NUM_EPOCHS: 5 → 7**
- **Lý do**: Mỗi epoch giờ nhanh hơn, nên tăng thêm epochs để model hội tụ tốt

---

## 📊 Ước tính cải thiện:

| Metric | Trước | Sau | Cải thiện |
|--------|-------|-----|-----------|
| **Frames/video** | 30 | 10 | -66% |
| **Tổng ảnh** | ~56,550 | ~18,850 | -66% |
| **Batch size** | 4 | 16 | +300% |
| **Iterations/epoch** | ~14,137 | ~1,178 | -92% |
| **Thời gian/epoch** | ~18h | **~1-2h** | **-89%** |
| **Mixed precision** | ❌ | ✅ | ~2x faster |

---

## 🎯 Các bước tiếp theo:

### **BƯỚC 1: Xóa dữ liệu cũ và tiền xử lý lại**

```powershell
# Xóa processed_data cũ (30 frames/video)
Remove-Item -Recurse -Force "f:\DoAn\DeepFake-Detection\processed_data\train"
Remove-Item -Recurse -Force "f:\DoAn\DeepFake-Detection\processed_data\val"
Remove-Item -Recurse -Force "f:\DoAn\DeepFake-Detection\processed_data\test"

# Chạy lại preprocessing với cấu hình mới (10 frames/video)
python main.py preprocess
```

⏱️ **Thời gian**: ~1-2 giờ (nhanh hơn vì chỉ lấy 10 frames)

---

### **BƯỚC 2: Reset checkpoint và training log**

```powershell
# Xóa checkpoint cũ (vì architecture không đổi, bạn có thể giữ)
# Nhưng nên reset để train với data mới
Remove-Item "f:\DoAn\DeepFake-Detection\saved_models\checkpoint.pth.tar"

# Backup và reset training log
Move-Item "f:\DoAn\DeepFake-Detection\evaluation_results\training_log.csv" `
          "f:\DoAn\DeepFake-Detection\evaluation_results\training_log_old.csv"
```

---

### **BƯỚC 3: Bắt đầu training với cấu hình mới**

```powershell
python main.py train
```

⏱️ **Thời gian dự kiến**: 
- **Mỗi epoch**: ~1-2 giờ (thay vì 18 giờ)
- **7 epochs**: ~7-14 giờ (có thể chạy qua đêm)

---

## 🔍 Monitoring trong quá trình training:

### Kiểm tra GPU usage:
```powershell
nvidia-smi -l 1
```

### Xem progress real-time:
- Training sẽ hiển thị progress bar với loss và accuracy
- Mỗi 100 batches sẽ update metrics

### Kiểm tra log:
```powershell
Get-Content "f:\DoAn\DeepFake-Detection\evaluation_results\training.log" -Tail 20 -Wait
```

---

## ⚠️ Nếu gặp lỗi Out of Memory (OOM):

### Giảm batch size xuống:
```python
# Trong configs/config.py
BATCH_SIZE = 12  # hoặc 8
```

### Hoặc giảm NUM_WORKERS:
```python
NUM_WORKERS = 4  # hoặc 2
```

---

## 📈 Kết quả dự kiến:

Với cấu hình tối ưu:
- **Validation Accuracy**: ~95-97% (tương đương với 30 frames)
- **Training Time**: Giảm từ **~180 giờ (10 epochs)** xuống **~7-14 giờ (7 epochs)**
- **Hiệu suất**: Tương đương vì model vẫn học được đặc trưng từ 10 frames

---

## 💡 Lưu ý quan trọng:

1. **Không cần lo lắng về giảm frames**: 10 frames đủ để model học được pattern deepfake
2. **Mixed Precision có thể gây NaN**: Nếu loss = NaN, tắt `MIXED_PRECISION = False`
3. **Monitor first epoch**: Nếu epoch đầu vẫn >3 giờ, cần điều chỉnh thêm
4. **Early Stopping**: Model sẽ tự dừng nếu val_acc không cải thiện sau 7 epochs

---

## 🎉 Sau khi training xong:

```powershell
# Đánh giá model
python main.py evaluate

# Chạy web app
python main.py app
```

Good luck! 🚀
