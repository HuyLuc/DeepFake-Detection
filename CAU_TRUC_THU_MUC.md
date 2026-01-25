# 📂 Cấu trúc Thư mục Mới - DeepFake Detection

## Tổng quan

Dự án đã được tổ chức lại thành **2 kiến trúc riêng biệt**:

```
src/
├── architectures/                    # 🏗️ CÁC KIẾN TRÚC MODEL
│   │
│   ├── 🔵 standard/                  # KIẾN TRÚC 1: EfficientNet đơn giản
│   │   ├── __init__.py
│   │   ├── model.py                  # Model factory
│   │   ├── dataset.py                # Dataset (frame độc lập)
│   │   └── train.py                  # Training script
│   │
│   └── 🟢 advanced/                  # KIẾN TRÚC 2: Temporal + Ensemble
│       ├── __init__.py
│       ├── temporal_model.py         # EfficientNet + LSTM
│       ├── ensemble_model.py         # EfficientNet + Swin
│       ├── temporal_dataset.py       # Dataset (sequences)
│       └── train.py                  # Training script
│
├── data_processing/                  # 📊 TIỀN XỬ LÝ (dùng chung)
│   ├── preprocess.py                 # Trích xuất khuôn mặt
│   └── deepfake_augmentation.py      # Data augmentation
│
├── app/                              # 🌐 WEB APPLICATION
│   └── main_app.py                   # Flask app
│
└── utils/                            # 🔧 TIỆN ÍCH
    ├── utils.py                      # Helper functions
    └── balanced_dataset.py           # Oversampling
```

---

## 🔵 Kiến trúc 1: Standard

**Đặc điểm:**
- EfficientNet-B4 đơn giản
- Phân loại từng frame độc lập
- Nhanh, nhẹ, phù hợp GPU yếu (2GB VRAM)

**Sử dụng:**
```bash
python main.py train
```

**Files:**
- `src/architectures/standard/model.py` - Tạo model
- `src/architectures/standard/dataset.py` - Load dataset
- `src/architectures/standard/train.py` - Training loop

---

## 🟢 Kiến trúc 2: Advanced

**Đặc điểm:**
- Kết hợp EfficientNet + LSTM + Swin Transformer
- Học temporal patterns (flickering)
- Accuracy cao (~95%), cần GPU mạnh (8GB+ VRAM)

**Sử dụng:**
```bash
# Temporal only (EfficientNet + LSTM)
python main.py train_advanced --model temporal

# Ensemble only (EfficientNet + Swin)
python main.py train_advanced --model ensemble

# Full power (cả hai)
python main.py train_advanced --model temporal_ensemble
```

**Files:**
- `src/architectures/advanced/temporal_model.py` - LSTM model
- `src/architectures/advanced/ensemble_model.py` - Ensemble model
- `src/architectures/advanced/temporal_dataset.py` - Sequence dataset
- `src/architectures/advanced/train.py` - Training loop

---

## 📊 So sánh

| | 🔵 Standard | 🟢 Advanced |
|---|---|---|
| **Lệnh** | `train` | `train_advanced` |
| **Backbone** | EfficientNet-B4 | EfficientNet + Swin |
| **Temporal** | ❌ | ✅ LSTM |
| **VRAM tối thiểu** | 2GB | 8GB |
| **Accuracy** | ~90% | ~95% |
| **Training time** | ~10 phút/epoch | ~25 phút/epoch |

---

## 🚀 Flow sử dụng

```
1. Preprocessing (DÙNG CHUNG)
   python main.py preprocess
   
2. Chọn Training:
   
   Option A: Standard (nhanh)
   python main.py train
   
   Option B: Advanced (chính xác)
   python main.py train_advanced --model temporal_ensemble

3. Evaluate
   python main.py evaluate

4. Web App
   python main.py app
```

---

## 📁 Files cũ (có thể xóa)

Các files cũ trong `src/training/` và `src/models/` vẫn được giữ lại để backward compatibility:

```
src/training/                         # CŨ - có thể xóa sau
├── dataset.py                        # → src/architectures/standard/dataset.py
├── train.py                          # → src/architectures/standard/train.py
├── train_temporal.py                 # → src/architectures/advanced/train.py
└── temporal_dataset.py               # → src/architectures/advanced/temporal_dataset.py

src/models/                           # CŨ - có thể xóa sau
├── temporal_model.py                 # → src/architectures/advanced/temporal_model.py
└── ensemble_model.py                 # → src/architectures/advanced/ensemble_model.py
```

**Lưu ý:** Không xóa ngay! Giữ lại để test cấu trúc mới trước.
