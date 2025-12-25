# DeepFake Detection Project

Dự án phát hiện Deepfake sử dụng Deep Learning với mô hình EfficientNet-B4. Hệ thống có khả năng phân loại video thành hai lớp: **REAL** (thật) và **FAKE** (giả mạo) thông qua việc phân tích các frame khuôn mặt được trích xuất từ video

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Cấu trúc dự án](#cấu-trúc-dự-án)
4. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
5. [Cài đặt](#cài-đặt)
6. [Cấu hình](#cấu-hình)
7. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
8. [Mô tả các module](#mô-tả-các-module)
9. [Kết quả và đánh giá](#kết-quả-và-đánh-giá)
10. [Troubleshooting](#troubleshooting)
11. [Tác giả và License](#tác-giả-và-license)

---

## 🎯 Giới thiệu

Dự án này sử dụng mô hình **EfficientNet-B4** (từ thư viện `timm`) để phát hiện video deepfake. Quy trình bao gồm:

1. **Tiền xử lý**: Trích xuất khuôn mặt từ video sử dụng MediaPipe
2. **Huấn luyện**: Fine-tune mô hình EfficientNet-B4 trên dataset Deepfake
3. **Đánh giá**: Kiểm tra hiệu suất trên tập test
4. **Ứng dụng Web**: Giao diện Flask để người dùng upload và kiểm tra video

### Dataset

Dự án sử dụng dataset Deepfake Detection Challenge với các phương pháp giả mạo:
- **Deepfakes**
- **Face2Face**
- **FaceSwap**
- **NeuralTextures**
- **DeepFakeDetection**
- **FaceShifter**

---

## ✨ Tính năng

- ✅ Phát hiện deepfake với độ chính xác cao
- ✅ Hỗ trợ nhiều định dạng video (MP4, AVI, MOV, MKV, WebM)
- ✅ Giao diện web thân thiện với Flask
- ✅ Tự động phát hiện và trích xuất khuôn mặt từ video
- ✅ Hỗ trợ GPU và CPU
- ✅ Tối ưu hóa cho GPU nhỏ (2GB VRAM)
- ✅ Logging và visualization training history
- ✅ Early stopping và checkpoint management
- ✅ Gradient accumulation cho batch size lớn

---

## 📁 Cấu trúc dự án

```
DeepFake-Detection/
│
├── main.py                          # Entry point chính
├── requirements.txt                  # Danh sách dependencies
├── README.md                        # Tài liệu dự án
├── .gitignore                       # Git ignore rules
│
├── configs/                         # Cấu hình
│   └── config.py                    # File cấu hình chính
│
├── src/                             # Source code
│   ├── data_processing/            # Tiền xử lý dữ liệu
│   │   └── preprocess.py           # Trích xuất khuôn mặt từ video
│   │
│   ├── training/                   # Huấn luyện và đánh giá
│   │   ├── dataset.py              # Custom Dataset class
│   │   ├── train.py                # Script huấn luyện
│   │   └── evaluate.py             # Script đánh giá
│   │
│   ├── app/                        # Ứng dụng web
│   │   ├── main_app.py             # Flask application
│   │   └── templates/
│   │       └── index.html          # Giao diện web
│   │
│   └── utils/                      # Tiện ích
│       └── utils.py                # Helper functions
│
├── data/                            # Dữ liệu gốc (không commit)
│   └── all/
│       ├── original_sequences/     # Video thật
│       └── manipulated_sequences/  # Video giả mạo
│
├── processed_data/                  # Dữ liệu đã xử lý (không commit)
│   ├── train/
│   │   ├── REAL/                   # Frame khuôn mặt thật (train)
│   │   └── FAKE/                   # Frame khuôn mặt giả (train)
│   ├── val/                        # Validation set
│   └── test/                       # Test set
│
├── saved_models/                   # Model đã lưu (không commit)
│   ├── checkpoint.pth.tar         # Checkpoint hiện tại
│   └── model_best.pth.tar          # Model tốt nhất
│
├── evaluation_results/             # Kết quả đánh giá (không commit)
│   ├── training_log.csv           # Log training
│   ├── training_history.png        # Biểu đồ training
│   ├── confusion_matrix.png       # Ma trận nhầm lẫn
│   ├── classification_report.txt   # Báo cáo phân loại
│   ├── training.log                # Log file
│   └── evaluation.log              # Log evaluation
│
├── reset_checkpoint.py             # Script reset checkpoint
├── visualize_training.py           # Script visualization
└── .venv/                          # Virtual environment (không commit)
```

---

## 💻 Yêu cầu hệ thống

### Phần cứng tối thiểu
- **CPU**: Bất kỳ CPU hiện đại nào
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: Không bắt buộc, nhưng khuyến nghị:
  - NVIDIA GPU với CUDA support
  - Tối thiểu 2GB VRAM (đã tối ưu cho GPU nhỏ)
  - Khuyến nghị: 4GB+ VRAM cho hiệu suất tốt hơn

### Phần mềm
- **Python**: 3.7 trở lên
- **CUDA**: 11.0+ (nếu sử dụng GPU)
- **cuDNN**: 8.0+ (nếu sử dụng GPU)

---

## 🔧 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/HuyLuc/DeepFake-Detection.git
cd DeepFake-Detection
```

### 2. Kích hoạt virtual environment

**Nếu bạn đã có môi trường ảo `.venv`:**

**Windows PowerShell:**
```bash
.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```bash
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Nếu chưa có môi trường ảo, tạo mới:**

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Lưu ý**: Bạn chỉ cần **một** môi trường ảo cho dự án. Nếu đã có `.venv`, chỉ cần kích hoạt nó, không cần tạo mới. `.venv` sẽ được ẩn khỏi VS Code Source Control.

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình dữ liệu

Đảm bảo cấu trúc dữ liệu như sau:
```
data/
└── all/
    ├── original_sequences/
    │   ├── youtube/c23/videos/
    │   └── actors/c23/videos/
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/
        ├── Face2Face/c23/videos/
        ├── FaceSwap/c23/videos/
        ├── NeuralTextures/c23/videos/
        ├── DeepFakeDetection/c23/videos/
        └── FaceShifter/c23/videos/
```

---

## ⚙️ Cấu hình

Tất cả cấu hình được quản lý trong file `configs/config.py`. Các tham số quan trọng:

### Đường dẫn dữ liệu
```python
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'all')
```

### Cấu hình huấn luyện
```python
MODEL_NAME = 'efficientnet_b4'      # Mô hình sử dụng
IMAGE_SIZE = (224, 224)              # Kích thước ảnh đầu vào
NUM_EPOCHS = 5                       # Số epoch
BATCH_SIZE = 4                       # Batch size (GPU) / 2 (CPU)
LEARNING_RATE = 0.0005               # Learning rate
ACCUMULATION_STEPS = 4               # Gradient accumulation
```

### Cấu hình ứng dụng web
```python
EVIDENCE_THRESHOLD = 0.80            # Ngưỡng phát hiện fake
MAX_VIDEO_SIZE_MB = 500              # Kích thước video tối đa
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
```

**Lưu ý**: Đường dẫn sẽ tự động điều chỉnh theo vị trí dự án, không cần chỉnh sửa thủ công.

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Tiền xử lý dữ liệu

Trích xuất khuôn mặt từ video và chia thành train/val/test:

```bash
python main.py preprocess
```

**Quá trình này sẽ:**
- Đọc video từ `data/all/`
- Phát hiện và trích xuất khuôn mặt bằng MediaPipe
- Lưu 10 frame/ảnh khuôn mặt cho mỗi video
- Tự động chia train (80%), validation (10%), test (10%)
- Lưu vào `processed_data/`

**Thời gian**: Phụ thuộc vào số lượng video (có thể mất vài giờ)

### Bước 2: Huấn luyện mô hình

```bash
python main.py train
```

**Quá trình này sẽ:**
- Tải EfficientNet-B4 pretrained
- Fine-tune trên dataset đã xử lý
- Tự động lưu checkpoint và best model
- Ghi log vào `evaluation_results/training_log.csv`
- Hỗ trợ resume từ checkpoint nếu bị gián đoạn

**Tối ưu hóa:**
- Tự động phát hiện GPU/CPU
- Gradient accumulation cho batch size lớn
- Early stopping khi không cải thiện
- Mixed precision (có thể tắt trong config)

### Bước 3: Đánh giá mô hình

```bash
python main.py evaluate
```

**Kết quả sẽ được lưu tại:**
- `evaluation_results/classification_report.txt`: Báo cáo chi tiết
- `evaluation_results/confusion_matrix.png`: Ma trận nhầm lẫn
- `evaluation_results/evaluation.log`: Log file

### Bước 4: Chạy ứng dụng web

```bash
python main.py app
```

Sau đó mở trình duyệt và truy cập: `http://localhost:5000`

**Tính năng:**
- Upload video để kiểm tra
- Tự động phát hiện khuôn mặt
- Hiển thị kết quả với độ tin cậy
- Hỗ trợ nhiều định dạng video

### Các công cụ hỗ trợ

**Visualize training history:**
```bash
python visualize_training.py
```

**Reset checkpoint:**
```bash
python reset_checkpoint.py --epoch 1 --val-acc 0.96
# Hoặc reset model_best
python reset_checkpoint.py --best-model --epoch 1
```

---

## 📦 Mô tả các module

### 1. `src/data_processing/preprocess.py`

**Chức năng**: Trích xuất khuôn mặt từ video

**Quy trình:**
1. Đọc video từ thư mục gốc
2. Chọn ngẫu nhiên 30 frame
3. Phát hiện khuôn mặt bằng MediaPipe
4. Cắt và lưu khuôn mặt
5. Phân loại REAL/FAKE dựa trên thư mục nguồn

**Tối ưu:**
- Multiprocessing để xử lý song song
- Tự động skip video đã xử lý
- Memory-efficient cho máy yếu

### 2. `src/training/dataset.py`

**Chức năng**: Custom PyTorch Dataset

**Tính năng:**
- Tự động load ảnh từ thư mục
- Xử lý lỗi khi ảnh bị hỏng
- Đảm bảo label khớp với ảnh
- Hỗ trợ data augmentation

### 3. `src/training/train.py`

**Chức năng**: Huấn luyện mô hình

**Tính năng:**
- Tự động phát hiện và tối ưu hardware
- Class weights để xử lý imbalanced data
- Gradient clipping để ổn định training
- Early stopping
- Checkpoint management
- Logging chi tiết

### 4. `src/training/evaluate.py`

**Chức năng**: Đánh giá mô hình trên test set

**Output:**
- Accuracy
- Precision, Recall, F1-score
- Confusion matrix
- Classification report

### 5. `src/app/main_app.py`

**Chức năng**: Flask web application

**API Endpoints:**
- `GET /`: Trang chủ
- `POST /predict_video`: Upload và phân tích video

**Tính năng:**
- Validation file upload
- Tự động phát hiện khuôn mặt
- Xử lý video frame-by-frame
- Trả về kết quả với độ tin cậy

### 6. `src/utils/utils.py`

**Chức năng**: Helper functions

**Các hàm:**
- `save_checkpoint()`: Lưu checkpoint
- `load_checkpoint()`: Tải checkpoint với error handling
- `verify_data_structure()`: Kiểm tra cấu trúc dữ liệu

---

## 📊 Kết quả và đánh giá

### Metrics

Sau khi chạy evaluation, bạn sẽ nhận được:

- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Độ chính xác cho từng lớp
- **Recall**: Độ nhạy cho từng lớp
- **F1-score**: Harmonic mean của Precision và Recall

### Visualization

Chạy `visualize_training.py` để xem:
- Training/Validation loss curves
- Training/Validation accuracy curves
- Learning rate schedule
- Combined metrics

### Checkpoint Management

- `checkpoint.pth.tar`: Checkpoint hiện tại (để resume)
- `model_best.pth.tar`: Model tốt nhất (để inference)

---

## 🔍 Troubleshooting

### Lỗi: "Unable to write new index file"

**Nguyên nhân**: File Git index bị lock hoặc không có quyền

**Giải pháp**:
```powershell
# Windows (chạy PowerShell với quyền Admin)
icacls .git /grant Everyone:F /T
git add -A
```

### Lỗi: "CUDA out of memory"

**Nguyên nhân**: Batch size quá lớn cho GPU

**Giải pháp**: Giảm `BATCH_SIZE` trong `configs/config.py` hoặc tăng `ACCUMULATION_STEPS`

### Lỗi: "No module named 'configs'"

**Nguyên nhân**: Chưa cài đặt đúng hoặc chạy từ thư mục sai

**Giải pháp**: Đảm bảo chạy từ thư mục gốc của dự án:
```bash
cd DeepFake-Detection
python main.py <task>
```

### Lỗi: "Không tìm thấy video gốc"

**Nguyên nhân**: Cấu trúc dữ liệu không đúng

**Giải pháp**: Kiểm tra lại cấu trúc thư mục trong `data/all/` theo mô tả ở phần [Cài đặt](#cài-đặt)

### Video không phát hiện được khuôn mặt

**Nguyên nhân**: 
- Video không có khuôn mặt
- Chất lượng video quá thấp
- Khuôn mặt quá nhỏ hoặc bị che khuất

**Giải pháp**: Thử video khác với khuôn mặt rõ ràng hơn

---

## 📝 Các lệnh nhanh

```bash
# Tiền xử lý
python main.py preprocess

# Huấn luyện
python main.py train

# Đánh giá
python main.py evaluate

# Chạy web app
python main.py app

# Xem training history
python visualize_training.py

# Reset checkpoint về epoch 1
python reset_checkpoint.py --epoch 1
```

---

## 🛠️ Công nghệ sử dụng

- **PyTorch**: Deep learning framework
- **timm**: Pretrained models (EfficientNet-B4)
- **MediaPipe**: Face detection
- **Flask**: Web framework
- **OpenCV**: Video processing
- **scikit-learn**: Metrics và evaluation
- **Pandas, Matplotlib, Seaborn**: Data visualization

---

## 📈 Hiệu suất

### Tối ưu hóa cho GPU nhỏ (2GB VRAM)

- Batch size: 4
- Gradient accumulation: 4 (effective batch = 16)
- Mixed precision: Tắt (để tránh NaN)
- Gradient clipping: Bật
- Prefetch factor: 2

### Tự động điều chỉnh

Hệ thống tự động:
- Phát hiện GPU/CPU
- Điều chỉnh batch size theo VRAM
- Điều chỉnh số workers theo RAM
- Tối ưu data loading

---

## 👤 Tác giả và License

**Tác giả**: HuyLuc

**License**: Xem file LICENSE (nếu có)

---

## 📚 Tài liệu tham khảo

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng mở issue trên GitHub.

---

**Lưu ý**: Đây là dự án nghiên cứu và giáo dục. Kết quả có thể khác nhau tùy thuộc vào dataset và cấu hình.
