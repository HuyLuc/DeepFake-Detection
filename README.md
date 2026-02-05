# 🛡️ DeepFake Detection System V2.0
> **Bảo vệ sự thật trong kỷ nguyên số với công nghệ AI tiên tiến.**

**DeepFake Detection** là hệ thống phân tích video thông minh, sử dụng **Deep Learning** để phát hiện nội dung giả mạo với độ chính xác cao. Dự án kết hợp sức mạnh của **EfficientNet** và **Swin Transformer** để phân tích cả không gian và thời gian, giúp nhận diện những dấu vết chỉnh sửa tinh vi nhất.

Điểm đột phá của hệ thống là khả năng **Explainable AI (XAI)** - không chỉ đưa ra kết quả mà còn giải thích lý do:
*   🔍 **X-Ray Lens (Heatmap)**: Soi chiếu vùng khuôn mặt bị can thiệp.
*   📉 **Smart Video Timeline**: Phân tích diễn biến độ giả mạo theo từng giây.
*   🏆 **Verification Badge**: Cấp chứng chỉ "Verified Real" cho nội dung sạch.

Phát triển với 2 kiến trúc linh hoạt:
- 🔵 **Standard Mode**: Tốc độ cao, nhẹ (EfficientNet-B4).
- 🟢 **Advanced Mode**: Độ chính xác tối đa (Ensemble Spatial-Temporal).

---

## 📋 Mục lục

1. [Tính năng](#-tính-năng)
2. [2 Kiến trúc Model](#-2-kiến-trúc-model)
3. [Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
5. [Cài đặt](#-cài-đặt)
6. [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
7. [Cấu hình chi tiết](#-cấu-hình-chi-tiết)
8. [Troubleshooting](#-troubleshooting)

---

## ✨ Tính năng

### Core Features
- ✅ Phát hiện deepfake với độ chính xác cao (~95%)
- ✅ **2 kiến trúc model** (Standard & Advanced)
- ✅ **X-Ray Lens**: Soi vùng giả mạo (Heatmap) thời gian thực
- ✅ **Smart Video Timeline**: Biểu đồ phân tích video theo từng giây
- ✅ **DeepFake Prevention Badge**: Cấp chứng nhận "Verified Real" cho ảnh/video thật
- ✅ **Batch Processing**: Xử lý hàng loạt file cùng lúc
- ✅ **API Documentation**: Tích hợp Swagger UI
- ✅ **Export Reports**: Xuất kết quả ra PDF/JSON

### Technical Features
- ✅ **Độ phân giải 380×380** - Tối ưu cho EfficientNet-B4
- ✅ **Uniform Sampling**: 20 frames/video cho video ngắn, adaptive cho video dài
- ✅ **Deepfake-Specific Augmentation** (JPEG, Noise, Blur, Cutout)
- ✅ **Oversampling** để cân bằng dữ liệu
- ✅ **LSTM/GRU** để học temporal patterns (flickering)
- ✅ **Ensemble** EfficientNet + Swin Transformer
- ✅ Mixed precision training & Early stopping
- ✅ Client-side optimizations (Canvas Badge, RequestAnimationFrame)

---

## 🏗️ 2 Kiến trúc Model

### 🔵 Kiến trúc 1: Standard

| Đặc điểm | Giá trị |
|----------|---------|
| **Model** | EfficientNet-B4 |
| **Input** | Từng frame độc lập |
| **VRAM tối thiểu** | 2GB |
| **Accuracy** | ~90% |
| **Thời gian/epoch** | ~10 phút |
| **Phù hợp** | GPU yếu, training nhanh |

**Cách dùng:**
```bash
python main.py train
```

### 🟢 Kiến trúc 2: Advanced

| Đặc điểm | Giá trị |
|----------|---------|
| **Model** | EfficientNet + LSTM + Swin Transformer |
| **Input** | Sequence 10 frames |
| **VRAM tối thiểu** | 8GB (khuyến nghị 16GB) |
| **Accuracy** | ~95% |
| **Thời gian/epoch** | ~25 phút |
| **Phù hợp** | GPU mạnh (T4, V100), cần accuracy cao |

**Cách dùng:**
```bash
# Temporal only (EfficientNet + LSTM)
python main.py train_advanced --model temporal

# Ensemble only (EfficientNet + Swin)
python main.py train_advanced --model ensemble

# Full power - KHUYẾN NGHỊ
python main.py train_advanced --model temporal_ensemble --epochs 15
```

### So sánh 2 kiến trúc

| | 🔵 Standard | 🟢 Advanced |
|---|---|---|
| Backbone | EfficientNet-B4 | EfficientNet + Swin |
| Temporal Learning | ❌ | ✅ LSTM |
| Phát hiện Flickering | ❌ | ✅ |
| Global Structure | ❌ | ✅ Transformer |
| Accuracy | ~90% | ~95% |
| VRAM | 2GB+ | 8GB+ |

---

## 📁 Cấu trúc dự án

```
DeepFake-Detection/
│
├── main.py                               # 🚀 Entry point chính
├── requirements.txt                       # Dependencies
├── README.md                              # Tài liệu này
│
├── configs/
│   └── config.py                          # Cấu hình chung
│
├── src/
│   ├── architectures/                     # 🏗️ CÁC KIẾN TRÚC MODEL
│   │   │
│   │   ├── 🔵 standard/                   # Kiến trúc 1: EfficientNet
│   │   │   ├── model.py                   # Model factory
│   │   │   ├── dataset.py                 # Dataset (frame độc lập)
│   │   │   └── train.py                   # Training script
│   │   │
│   │   ├── 🟢 advanced/                   # Kiến trúc 2: Temporal + Ensemble
│   │   │   ├── temporal_model.py          # EfficientNet + LSTM
│   │   │   ├── ensemble_model.py          # EfficientNet + Swin
│   │   │   ├── temporal_dataset.py        # Dataset (sequences)
│   │   │   └── train.py                   # Training script
│   │   │
│   │   └── evaluate.py                    # Script đánh giá (chung)
│   │
│   ├── data_processing/                   # 📊 TIỀN XỬ LÝ
│   │   ├── preprocess.py                  # Trích xuất khuôn mặt
│   │   └── deepfake_augmentation.py       # Data augmentation
│   │
│   ├── app/                               # 🌐 WEB APP
│   │   ├── main_app.py                    # Flask application
│   │   └── templates/index.html
│   │
│   └── utils/                             # 🔧 TIỆN ÍCH
│       ├── utils.py                       # Helper functions
│       └── balanced_dataset.py            # Oversampling
│
├── data/all/                              # Dữ liệu gốc (không commit)
├── processed_data/                        # Dữ liệu đã xử lý
├── saved_models/                          # Models đã train
└── evaluation_results/                    # Kết quả đánh giá
```

---

## 💻 Yêu cầu hệ thống

### Cho kiến trúc Standard (🔵)
- **GPU**: Tối thiểu 2GB VRAM (hoặc CPU)
- **RAM**: 8GB+
- **Python**: 3.8+

### Cho kiến trúc Advanced (🟢)
- **GPU**: Tối thiểu 8GB VRAM (khuyến nghị 16GB)
- **RAM**: 16GB+
- **Python**: 3.8+
- **Khuyến nghị**: Google Colab với GPU T4/V100

---

## 🔧 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/HuyLuc/DeepFake-Detection.git
cd DeepFake-Detection
```

### 2. Tạo môi trường ảo

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu

Đặt video vào thư mục theo cấu trúc:
```
data/all/
├── original_sequences/
│   └── youtube/c23/videos/     # Video thật
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/   # Video fake
    ├── Face2Face/c23/videos/
    ├── FaceSwap/c23/videos/
    └── NeuralTextures/c23/videos/
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Tiền xử lý dữ liệu

```bash
python main.py preprocess
```

**Quá trình này sẽ:**
- Đọc video từ `data/all/`
- Trích xuất 20 frames/video (uniform sampling)
- Phát hiện và crop khuôn mặt
- Lưu vào `processed_data/train/`, `val/`, `test/`

### Bước 2: Chọn kiến trúc và Training

#### Option A: 🔵 Standard (nhanh, GPU yếu)

```bash
python main.py train
```

#### Option B: 🟢 Advanced (chính xác, GPU mạnh)

```bash
# Khuyến nghị - Dùng trên Google Colab với GPU T4
python main.py train_advanced --model temporal_ensemble --epochs 15 --batch-size 8
```

**Các tham số cho `train_advanced`:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--model` | `temporal_ensemble` | Loại model: `temporal`, `ensemble`, `temporal_ensemble`, `lightweight` |
| `--seq-len` | `10` | Số frames mỗi sequence |
| `--epochs` | `10` | Số epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--resume` | `None` | Path checkpoint để resume |

### Bước 3: Đánh giá

```bash
python main.py evaluate
```

Kết quả lưu tại:
- `evaluation_results/classification_report.txt`
- `evaluation_results/confusion_matrix.png`

### Bước 4: Chạy Web App

```bash
python main.py app
```

Mở trình duyệt: `http://localhost:5000`

### Bước 5: Khám phá các tính năng nâng cao

1.  **X-Ray Lens**:
    - Upload ảnh/video -> Kết quả hiện ra.
    - Di chuột vào ảnh để bật chế độ "Kính lúp" soi Heatmap.

2.  **Smart Timeline (Video)**:
    - Upload video -> Xem biểu đồ biến thiên độ Fake.
    - Click vào điểm trên biểu đồ để tua video đến frame nghi ngờ.

3.  **Chứng nhận (Verified Badge)**:
    - Nếu kết quả là **REAL** (>90%).
    - Bấm nút **"Tải chứng nhận"** để lưu ảnh/video frame có đóng dấu bảo mật.

4.  **API Documentation**:
    - Truy cập `http://localhost:5000/apidocs/` để xem và test API.
    - Hỗ trợ `/predict`, `/history`, và `/export`.

---

## ⚙️ Cấu hình chi tiết

File: `configs/config.py`

### Cấu hình chung
```python
IMAGE_SIZE = (380, 380)              # Độ phân giải ảnh
NUM_FRAMES_PER_VIDEO = 20            # Frames lấy từ mỗi video
BATCH_SIZE = 8                       # Batch size
NUM_EPOCHS = 10                      # Số epochs
LEARNING_RATE = 0.0001               # Learning rate
```

### Data Augmentation
```python
USE_DEEPFAKE_AUGMENTATION = True     # Bật augmentation chuyên biệt
ENABLE_COMPRESSION_AUG = True        # JPEG compression
ENABLE_NOISE_AUG = True              # Gaussian noise
ENABLE_BLUR_AUG = True               # Adaptive blur
ENABLE_CUTOUT_AUG = True             # Face cutout
```

### Data Balancing
```python
USE_OVERSAMPLING = True              # Bật oversampling
OVERSAMPLE_RATIO = 1.3               # Tỷ lệ oversample lớp REAL
```

---

## 🔍 Troubleshooting

### Lỗi: CUDA out of memory

**Nguyên nhân:** GPU không đủ VRAM

**Giải pháp:**
```python
# Trong config.py, giảm BATCH_SIZE
BATCH_SIZE = 4  # hoặc 2

# Hoặc chuyển sang kiến trúc Standard
python main.py train  # thay vì train_advanced
```

### Lỗi: Module not found

**Giải pháp:**
```bash
# Đảm bảo chạy từ thư mục gốc
cd DeepFake-Detection
python main.py <task>
```

### Training quá chậm

**Giải pháp:**
```python
# Giảm số workers trong config.py
NUM_WORKERS = 0  # Dùng cho máy yếu

# Hoặc tắt một số augmentation
ENABLE_COMPRESSION_AUG = False
```

### Không tìm thấy GPU

```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, cài lại PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Kết quả mong đợi

| Kiến trúc | Train Acc | Val Acc | Test Acc |
|-----------|-----------|---------|----------|
| 🔵 Standard | ~95% | ~90% | ~88-90% |
| 🟢 Advanced | ~97% | ~95% | ~93-95% |

---

## 🛠️ Công nghệ sử dụng

- **PyTorch** - Deep learning framework
- **timm** - Pretrained models (EfficientNet, Swin)
- **MediaPipe** - Face detection
- **Flask** - Web framework
- **OpenCV** - Video processing
- **scikit-learn** - Metrics

---

## 👤 Tác giả

**HuyLuc**

---

## 📚 Tài liệu tham khảo

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 🤝 Đóng góp

1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

**Lưu ý:** Đây là dự án nghiên cứu và giáo dục. Kết quả có thể khác nhau tùy thuộc vào dataset và cấu hình phần cứng.
