# ✅ Checklist Trước Khi Chạy Training

## 📋 Kiểm tra trước khi training

### 1. ✅ Cấu hình Hardware

- [ ] **GPU có đủ VRAM?**
  - Tối thiểu: 2GB VRAM cho BATCH_SIZE=8
  - Khuyến nghị: 4GB+ VRAM
  - Kiểm tra: `nvidia-smi` (Windows/Linux)
  
- [ ] **RAM đủ?**
  - Tối thiểu: 8GB RAM
  - Khuyến nghị: 16GB+ RAM
  
### 2. ✅ Dữ liệu

- [ ] **Đã chạy preprocessing chưa?**
  ```bash
  # Kiểm tra xem thư mục processed_data có dữ liệu chưa
  dir processed_data\train\FAKE
  dir processed_data\train\REAL
  dir processed_data\val\FAKE
  dir processed_data\val\REAL
  ```
  
- [ ] **Đủ dữ liệu để train?**
  - Tối thiểu: 1000+ ảnh cho mỗi lớp (FAKE, REAL)
  - Khuyến nghị: 5000+ ảnh cho mỗi lớp

### 3. ✅ Cấu hình (configs/config.py)

- [ ] **IMAGE_SIZE đã đúng?**
  ```python
  IMAGE_SIZE = (380, 380)  # ✅ Đúng cho nâng cấp mới
  ```

- [ ] **BATCH_SIZE phù hợp với VRAM?**
  ```python
  # GPU 2GB: BATCH_SIZE = 4-8
  # GPU 4GB: BATCH_SIZE = 8-16
  # GPU 6GB+: BATCH_SIZE = 16-32
  ```

- [ ] **Deepfake Augmentation đã bật?**
  ```python
  USE_DEEPFAKE_AUGMENTATION = True  # ✅ Bật
  ```

- [ ] **Oversampling đã bật?**
  ```python
  USE_OVERSAMPLING = True  # ✅ Bật
  OVERSAMPLE_RATIO = 1.3   # ✅ Tỷ lệ hợp lý (1.2-1.5)
  ```

- [ ] **Mixed Precision đã bật? (quan trọng!)**
  ```python
  MIXED_PRECISION = True  # ✅ Bắt buộc với resolution 380x380
  ```

### 4. ✅ Dependencies

- [ ] **Đã cài đặt tất cả packages?**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Kiểm tra torch version**
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
  ```

### 5. ✅ Môi trường

- [ ] **Đã kích hoạt virtual environment?**
  ```bash
  .venv\Scripts\activate  # Windows
  # hoặc
  source .venv/bin/activate  # Linux/Mac
  ```

- [ ] **Đang ở thư mục dự án?**
  ```bash
  pwd  # Phải là f:\DoAn\DeepFake-Detection
  ```

## 🚀 Chạy Training

### Cách 1: Sử dụng main.py (Khuyến nghị)
```bash
python main.py train
```

### Cách 2: Chạy trực tiếp
```bash
python -c "from src.training.train import run_training; run_training()"
```

## 📊 Theo dõi Training

### 1. Xem logs real-time
- Logs được in ra console
- File log: `evaluation_results/training.log`

### 2. Các metrics cần chú ý

**Epoch đầu tiên:**
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
   FAKE: XXXX mẫu
   REAL: XXXX mẫu

📊 Phân bố dữ liệu sau oversampling (ratio=1.3):
   FAKE: XXXX mẫu
   REAL: XXXX mẫu (tăng lên)
```

**Mỗi epoch:**
```
Epoch X/10
--------------------
Train Loss: 0.XXXX Acc: 0.XXXX
Validation Loss: 0.XXXX Acc: 0.XXXX
🎉 New best validation accuracy: 0.XXXX  # Khi có cải thiện
```

### 3. Dấu hiệu tốt

✅ **Train Loss giảm dần**
✅ **Val Loss giảm dần (không tăng)**
✅ **Val Accuracy tăng dần**
✅ **Gap giữa Train Acc và Val Acc < 5%** (không overfitting)

### 4. Dấu hiệu xấu

❌ **Val Loss tăng trong khi Train Loss giảm** → Overfitting
❌ **Train Acc >> Val Acc (chênh lệch >10%)** → Overfitting nghiêm trọng
❌ **Out of Memory (OOM)** → Giảm BATCH_SIZE

## 🛑 Dừng Training Khẩn cấp

```
Ctrl+C  # Dừng training
```

Checkpoint sẽ được lưu tại:
- `saved_models/checkpoint.pth.tar` (checkpoint cuối cùng)
- `saved_models/model_best.pth.tar` (model tốt nhất)

## 📈 Sau khi Training xong

### 1. Kiểm tra kết quả
```bash
# Xem training log
cat evaluation_results/training_log.csv

# Hoặc dùng analyze_training.py (nếu có)
python analyze_training.py
```

### 2. Test model
```bash
# Chạy evaluation trên test set
python main.py evaluate

# Hoặc test bằng web app
python main.py app
```

### 3. So sánh với baseline

| Metric | Baseline (224x224) | New (380x380) |
|--------|-------------------|---------------|
| Best Val Acc | ? | ? |
| Train Time/Epoch | ~10 phút | ~15-20 phút |
| Final Test Acc | ? | ? |

## 🔧 Troubleshooting

### Lỗi: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Giải pháp:**
1. Giảm BATCH_SIZE trong `configs/config.py`
2. Hoặc giảm IMAGE_SIZE về (224, 224)

### Lỗi: Import Error
```
ModuleNotFoundError: No module named 'src.data_processing.deepfake_augmentation'
```
**Giải pháp:**
```bash
# Kiểm tra file có tồn tại không
dir src\data_processing\deepfake_augmentation.py

# Nếu không có, file bị mất. Cần tạo lại từ code.
```

### Training quá chậm
**Giải pháp:**
1. Giảm NUM_WORKERS nếu CPU yếu
2. Tắt JPEG Compression (tốn thời gian nhất):
   ```python
   ENABLE_COMPRESSION_AUG = False
   ```

### Val Accuracy không tăng
**Nguyên nhân:** Model có thể cần nhiều epochs hơn hoặc learning rate chưa phù hợp

**Giải pháp:**
1. Tăng NUM_EPOCHS lên 15-20
2. Hoặc điều chỉnh LEARNING_RATE

## 📞 Liên hệ hỗ trợ

Nếu gặp vấn đề không giải quyết được:
1. Kiểm tra file logs: `evaluation_results/training.log`
2. Xem lại tài liệu: `NANG_CAP_DU_LIEU_DAU_VAO.md`
3. Đọc phần Troubleshooting trong tài liệu

---

**Chúc bạn training thành công! 🚀**
