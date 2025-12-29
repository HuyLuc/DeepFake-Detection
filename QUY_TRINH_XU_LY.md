# QUY TRÌNH XỬ LÝ BÀI TOÁN PHÁT HIỆN DEEPFAKE

---

## 📋 MỤC LỤC

1. [Giải thích dễ hiểu về quy trình](#0-giải-thích-dễ-hiểu-về-quy-trình)
2. [Tổng quan quy trình](#1-tổng-quan-quy-trình)
3. [Giai đoạn 1: Tiền xử lý dữ liệu](#2-giai-đoạn-1-tiền-xử-lý-dữ-liệu)
4. [Giai đoạn 2: Huấn luyện mô hình](#3-giai-đoạn-2-huấn-luyện-mô-hình)
5. [Giai đoạn 3: Đánh giá mô hình](#4-giai-đoạn-3-đánh-giá-mô-hình)
6. [Giai đoạn 4: Dự đoán (Inference)](#5-giai-đoạn-4-dự-đoán-inference)
7. [Sơ đồ tổng hợp](#6-sơ-đồ-tổng-hợp)

---

## 0. GIẢI THÍCH DỄ HIỂU VỀ QUY TRÌNH

### 🎯 Bài toán là gì?

Bạn có một video và muốn biết video đó là **THẬT** hay **GIẢ MẠO (Deepfake)**. Hệ thống sẽ tự động phân tích và cho bạn biết kết quả.

### 🔄 Quy trình hoạt động như thế nào?

Hãy tưởng tượng bạn đang dạy một đứa trẻ nhận biết tiền giả:

#### **BƯỚC 1: Chuẩn bị tài liệu học tập (Tiền xử lý dữ liệu)**

**Giống như:** Bạn thu thập nhiều tờ tiền thật và tiền giả, rồi cắt ra phần quan trọng nhất (hình chân dung) để dễ so sánh.

**Trong hệ thống:**
- Có rất nhiều video (một số thật, một số giả)
- Từ mỗi video, hệ thống sẽ:
  1. Chọn ra 10 khung hình đại diện (giống như chọn 10 tờ tiền mẫu)
  2. Tìm khuôn mặt trong mỗi khung hình (giống như cắt phần chân dung)
  3. Lưu lại 10 ảnh khuôn mặt đó

**Kết quả:** Có hàng nghìn ảnh khuôn mặt được phân loại sẵn là "THẬT" hoặc "GIẢ"

---

#### **BƯỚC 2: Dạy máy tính học (Huấn luyện mô hình)**

**Giống như:** Bạn cho đứa trẻ xem hàng nghìn tờ tiền đã được đánh dấu "thật" hoặc "giả", và dạy nó nhận biết các đặc điểm khác biệt.

**Trong hệ thống:**
1. **Nạp dữ liệu:** Đưa tất cả ảnh khuôn mặt vào máy tính
2. **Xây dựng "bộ não":** Tạo một mô hình AI (giống như bộ não) có khả năng học
3. **Học từng bước:**
   - Xem một ảnh → Đoán là thật hay giả
   - So sánh với đáp án → Biết đúng hay sai
   - Điều chỉnh "bộ não" để đoán đúng hơn
   - Lặp lại hàng nghìn lần
4. **Kiểm tra:** Sau mỗi lần học, kiểm tra xem đoán đúng bao nhiêu phần trăm
5. **Lưu lại:** Khi "bộ não" đã học tốt, lưu lại để dùng sau

**Kết quả:** Một "bộ não" đã được huấn luyện, có thể nhận biết video thật/giả

---

#### **BƯỚC 3: Kiểm tra khả năng (Đánh giá mô hình)**

**Giống như:** Cho đứa trẻ xem những tờ tiền mới (chưa từng thấy) và xem nó đoán đúng bao nhiêu phần trăm.

**Trong hệ thống:**
- Dùng một bộ video mới (chưa từng dùng để học)
- Cho "bộ não" đoán từng video
- Tính toán: Đoán đúng bao nhiêu %, sai bao nhiêu %
- Tạo báo cáo chi tiết về khả năng của hệ thống

**Kết quả:** Biết được hệ thống chính xác đến mức nào

---

#### **BƯỚC 4: Sử dụng thực tế (Dự đoán)**

**Giống như:** Khi có một tờ tiền mới, bạn đưa cho đứa trẻ xem và nó sẽ nói "thật" hoặc "giả".

**Trong hệ thống:**
1. Người dùng upload một video mới
2. Hệ thống xử lý video:
   - Chọn một số khung hình từ video
   - Tìm khuôn mặt trong mỗi khung hình
   - Cắt khuôn mặt ra
3. Cho "bộ não" xem từng ảnh khuôn mặt:
   - "Bộ não" đoán: "Ảnh này 85% là giả"
   - "Ảnh này 92% là thật"
   - "Ảnh này 78% là giả"
4. Tổng hợp kết quả:
   - Nếu nhiều ảnh được đoán là "giả" → Video là **FAKE**
   - Nếu hầu hết là "thật" → Video là **REAL**
5. Trả về kết quả cho người dùng

**Kết quả:** Người dùng biết video của họ là thật hay giả

---

### 📊 Tóm tắt bằng ví dụ đơn giản

| Giai đoạn | Ví dụ đời thường | Trong hệ thống |
|-----------|------------------|----------------|
| **1. Tiền xử lý** | Cắt ảnh chân dung từ tiền | Cắt khuôn mặt từ video |
| **2. Huấn luyện** | Dạy trẻ nhận biết tiền giả | Dạy AI nhận biết Deepfake |
| **3. Đánh giá** | Kiểm tra khả năng của trẻ | Test độ chính xác của AI |
| **4. Dự đoán** | Xem tiền mới và đoán | Phân tích video mới |

---

### 🔍 Chi tiết từng bước (dễ hiểu hơn)

#### **Giai đoạn 1: Tại sao phải cắt khuôn mặt?**

- Video rất lớn và phức tạp
- Chỉ phần khuôn mặt mới có dấu hiệu Deepfake
- Giống như khi kiểm tra tiền, bạn chỉ cần xem phần chân dung, không cần xem toàn bộ tờ tiền

#### **Giai đoạn 2: "Bộ não" học như thế nào?**

- **Lần 1:** Xem ảnh → Đoán sai → Điều chỉnh
- **Lần 2:** Xem ảnh khác → Đoán đúng hơn một chút → Điều chỉnh tiếp
- **Lần 1000:** Đoán khá chính xác
- **Lần 10000:** Đoán rất chính xác

Quá trình này lặp lại hàng nghìn lần cho đến khi "bộ não" học được các đặc điểm để phân biệt.

#### **Giai đoạn 3: Tại sao phải kiểm tra?**

- Để biết "bộ não" có thực sự học được hay chỉ học thuộc lòng
- Giống như kiểm tra học sinh: không cho xem bài cũ, chỉ cho bài mới

#### **Giai đoạn 4: Tại sao phải xem nhiều khung hình?**

- Một khung hình có thể bị lỗi hoặc không rõ
- Xem nhiều khung hình → Nhiều bằng chứng → Kết quả chính xác hơn
- Giống như: Một người nói "tiền này giả" → Chưa chắc. Nhiều người nói → Chắc chắn hơn

---

### 🎓 Hiểu về các thuật ngữ

| Thuật ngữ | Giải thích đơn giản |
|-----------|---------------------|
| **Frame** | Một khung hình trong video (giống như một bức ảnh) |
| **Face Detection** | Tìm khuôn mặt trong ảnh (giống như tìm khuôn mặt trong đám đông) |
| **Model/Model** | "Bộ não" của máy tính, có khả năng học và nhận biết |
| **Training** | Quá trình dạy "bộ não" học |
| **Checkpoint** | Lưu lại trạng thái "bộ não" đã học (giống như lưu bài học) |
| **Inference** | Sử dụng "bộ não" đã học để đoán video mới |
| **Confidence** | Độ tin cậy (ví dụ: 85% chắc chắn là giả) |

---

### 💡 Ví dụ cụ thể

**Tình huống:** Bạn có một video và muốn kiểm tra xem có phải Deepfake không.

**Quy trình:**
1. Bạn upload video lên hệ thống
2. Hệ thống tự động:
   - Chọn 10 khung hình từ video (ví dụ: khung hình số 0, 100, 200, ...)
   - Tìm khuôn mặt trong mỗi khung hình
   - Cắt ra 10 ảnh khuôn mặt
3. Cho "bộ não" xem 10 ảnh:
   - Ảnh 1: 90% là FAKE
   - Ảnh 2: 85% là FAKE
   - Ảnh 3: 88% là FAKE
   - Ảnh 4: 92% là REAL
   - Ảnh 5: 87% là FAKE
   - ... (10 ảnh)
4. Tổng hợp: 7/10 ảnh là FAKE → Video là **FAKE**
5. Trả kết quả: "Video này có 70% khả năng là Deepfake"

---

### 🚀 Tại sao quy trình này hiệu quả?

1. **Tập trung vào phần quan trọng:** Chỉ xử lý khuôn mặt, không xử lý toàn bộ video
2. **Học từ nhiều ví dụ:** Xem hàng nghìn ảnh để học được đặc điểm
3. **Kiểm tra kỹ lưỡng:** Đảm bảo "bộ não" thực sự học được, không chỉ học thuộc
4. **Xem nhiều bằng chứng:** Phân tích nhiều khung hình để kết quả chính xác hơn

---

## 1. TỔNG QUAN QUY TRÌNH

| Bước | Giai đoạn | Mô tả | Đầu vào | Đầu ra |
|------|-----------|-------|---------|--------|
| **1** | Tiền xử lý | Xử lý video thô thành ảnh khuôn mặt | Video thô (MP4) | Ảnh khuôn mặt đã cắt (PNG) |
| **2** | Huấn luyện | Train mô hình phân loại | Ảnh đã xử lý | Mô hình đã train (checkpoint) |
| **3** | Đánh giá | Test mô hình trên tập test | Mô hình + Test set | Báo cáo đánh giá |
| **4** | Dự đoán | Phát hiện Deepfake trên video mới | Video mới | Kết quả FAKE/REAL |

---

## 2. GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU

### 2.1. Quy trình tổng thể

| Bước | Thao tác | Công cụ/Kỹ thuật | Đầu vào | Đầu ra |
|------|----------|------------------|---------|--------|
| **1.1** | Thu thập video | File system scan | `data/all/original_sequences/`<br>`data/all/manipulated_sequences/` | Danh sách video paths |
| **→** | **1.2** | Phân chia dữ liệu | Random shuffle (seed=42) | Danh sách video IDs | Train/Val/Test IDs<br>(80%/10%/10%) |
| **→** | **1.3** | Xử lý từng video | Multiprocessing | Video file (.mp4) | Thư mục chứa frames |

### 2.2. Chi tiết xử lý một video

| Bước | Thao tác | Công cụ | Tham số | Kết quả |
|------|----------|---------|---------|---------|
| **1.3.1** | Đọc video | `cv2.VideoCapture()` | Video path | Video object |
| **→** | **1.3.2** | Đếm tổng số frame | `CAP_PROP_FRAME_COUNT` | Video object | `total_frames` |
| **→** | **1.3.3** | Chọn frame cần xử lý | **Uniform Sampling**<br>(nếu video dài)<br>**Temporal Padding**<br>(nếu video ngắn) | `total_frames`<br>`NUM_FRAMES_PER_VIDEO=10` | Danh sách frame indices |
| **→** | **1.3.4** | Đọc frame từ video | `cap.read()` | Frame index | Frame image (BGR) |
| **→** | **1.3.5** | Chuyển đổi màu | `cv2.cvtColor(BGR→RGB)` | BGR frame | RGB frame |
| **→** | **1.3.6** | Phát hiện khuôn mặt | **MediaPipe Face Detection** | RGB frame<br>`model_selection=1`<br>`min_confidence=0.5` | Face bounding box |
| **→** | **1.3.7** | Cắt khuôn mặt | Array slicing với margin | Frame + Bounding box<br>`FACE_MARGIN=20px` | Ảnh khuôn mặt đã cắt |
| **→** | **1.3.8** | Lưu ảnh | `cv2.imwrite()` | Ảnh khuôn mặt | File PNG<br>`frame_{index}.png` |

### 2.3. Cấu trúc dữ liệu sau tiền xử lý

```
processed_data/
├── train/
│   ├── FAKE/
│   │   ├── video_id_001/
│   │   │   ├── frame_0.png
│   │   │   ├── frame_1.png
│   │   │   └── ... (10 frames)
│   │   └── video_id_002/
│   └── REAL/
│       ├── video_id_001/
│       └── video_id_002/
├── val/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

---

## 3. GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH

### 3.1. Quy trình tổng thể

| Bước | Thao tác | Mô tả | Đầu vào | Đầu ra |
|------|----------|-------|---------|--------|
| **2.1** | Chuẩn bị dữ liệu | Load và transform ảnh | `processed_data/train/`<br>`processed_data/val/` | DataLoader objects |
| **→** | **2.2** | Xây dựng mô hình | Tạo EfficientNet-B4 | Config | Model architecture |
| **→** | **2.3** | Thiết lập training | Optimizer, Loss, Scheduler | Model | Training components |
| **→** | **2.4** | Vòng lặp training | Train + Validate | Data + Model | Trained model |
| **→** | **2.5** | Lưu checkpoint | Save best model | Model state | Checkpoint files |

### 3.2. Chi tiết chuẩn bị dữ liệu

| Bước | Thao tác | Công cụ | Tham số | Kết quả |
|------|----------|---------|---------|---------|
| **2.1.1** | Load dataset | `DeepfakeDataset` | `data_dir` | Dataset object |
| **→** | **2.1.2** | Áp dụng transform | `transforms.Compose()` | Image | Transformed image |
| **→** | **2.1.3** | Data Augmentation | RandomHorizontalFlip<br>ColorJitter<br>RandomRotation | Image | Augmented image |
| **→** | **2.1.4** | Normalize | ImageNet stats | Image tensor | Normalized tensor |
| **→** | **2.1.5** | Tạo DataLoader | `DataLoader()` | Dataset | Batched data<br>`BATCH_SIZE=16`<br>`NUM_WORKERS=6` |

### 3.3. Chi tiết xây dựng mô hình

| Bước | Thao tác | Công cụ | Tham số | Kết quả |
|------|----------|---------|---------|---------|
| **2.2.1** | Tạo model | `timm.create_model()` | `efficientnet_b4`<br>`pretrained=True` | Model với ImageNet weights |
| **→** | **2.2.2** | Thay đổi classifier | Modify last layer | `num_classes=2` | Model cho binary classification |
| **→** | **2.2.3** | Chuyển model lên device | `.to(device)` | Model | Model trên GPU/CPU |

### 3.4. Chi tiết thiết lập training

| Bước | Thao tác | Công cụ | Tham số | Kết quả |
|------|----------|---------|---------|---------|
| **2.3.1** | Tính class weights | Count samples | Train dataset | `[weight_fake, weight_real]` |
| **→** | **2.3.2** | Tạo optimizer | `optim.Adam()` | Model parameters | Optimizer<br>`lr=0.0005`<br>`weight_decay=1e-5` |
| **→** | **2.3.3** | Tạo loss function | `nn.CrossEntropyLoss()` | Class weights | Loss function |
| **→** | **2.3.4** | Tạo scheduler | `ReduceLROnPlateau()` | Optimizer | LR Scheduler<br>`patience=3`<br>`factor=0.1` |
| **→** | **2.3.5** | Tạo GradScaler | `GradScaler('cuda')` | Mixed precision | Scaler (nếu dùng FP16) |

### 3.5. Chi tiết vòng lặp training (một epoch)

| Bước | Thao tác | Công cụ | Mô tả | Kết quả |
|------|----------|---------|-------|---------|
| **2.4.1** | Set model mode | `model.train()` | Training mode | Model sẵn sàng train |
| **→** | **2.4.2** | Loop qua batches | `for batch in train_loader` | Iterate data | Batch (images, labels) |
| **→** | **2.4.3** | Forward pass | `model(images)` | Inference | Predictions |
| **→** | **2.4.4** | Tính loss | `criterion(predictions, labels)` | Loss calculation | Loss value |
| **→** | **2.4.5** | Backward pass | `loss.backward()` | Gradient computation | Gradients |
| **→** | **2.4.6** | Gradient clipping | `torch.nn.utils.clip_grad_norm_()` | Clip gradients | Clipped gradients<br>`max_norm=1.0` |
| **→** | **2.4.7** | Update weights | `optimizer.step()` | Weight update | Updated model |
| **→** | **2.4.8** | Validation | `model.eval()` + Loop val data | Evaluate | Val accuracy, loss |
| **→** | **2.4.9** | Update scheduler | `scheduler.step(val_acc)` | Adjust LR | New learning rate |
| **→** | **2.4.10** | Early stopping check | Compare val acc | Check improvement | Continue/Stop |

### 3.6. Chi tiết lưu checkpoint

| Bước | Thao tác | Nội dung lưu | Điều kiện | File output |
|------|----------|-------------|-----------|-------------|
| **2.5.1** | Lưu checkpoint mỗi epoch | Model state<br>Optimizer state<br>Epoch number<br>Best val acc | Sau mỗi epoch | `checkpoint.pth.tar` |
| **→** | **2.5.2** | Lưu best model | Model state | Khi val acc cải thiện | `model_best.pth.tar` |
| **→** | **2.5.3** | Sync to Drive | Copy files | Nếu dùng Colab | Google Drive |

---

## 4. GIAI ĐOẠN 3: ĐÁNH GIÁ MÔ HÌNH

### 4.1. Quy trình đánh giá

| Bước | Thao tác | Công cụ | Đầu vào | Đầu ra |
|------|----------|---------|---------|--------|
| **3.1** | Load test dataset | `DeepfakeDataset` | `processed_data/test/` | Test dataset |
| **→** | **3.2** | Tải best model | `load_checkpoint()` | `model_best.pth.tar` | Trained model |
| **→** | **3.3** | Dự đoán trên test set | `model.eval()` + Loop | Test batches | Predictions |
| **→** | **3.4** | Tính metrics | `sklearn.metrics` | Predictions + Labels | Accuracy, Precision, Recall, F1 |
| **→** | **3.5** | Vẽ confusion matrix | `seaborn.heatmap()` | Predictions + Labels | Confusion matrix plot |
| **→** | **3.6** | Lưu kết quả | File I/O | Metrics + Plots | Report files |

### 4.2. Chi tiết tính toán metrics

| Metric | Công thức | Mô tả |
|--------|-----------|-------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Tỷ lệ dự đoán đúng |
| **Precision** | `TP / (TP + FP)` | Độ chính xác khi dự đoán FAKE |
| **Recall** | `TP / (TP + FN)` | Tỷ lệ phát hiện được FAKE |
| **F1-Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Trung bình điều hòa |

---

## 5. GIAI ĐOẠN 4: DỰ ĐOÁN (INFERENCE)

### 5.1. Quy trình dự đoán video

| Bước | Thao tác | Công cụ | Đầu vào | Đầu ra |
|------|----------|---------|---------|--------|
| **4.1** | User upload video | Flask `request.files` | Video file (.mp4) | File object |
| **→** | **4.2** | Validate file | Check extension, size | File object | Valid/Invalid |
| **→** | **4.3** | Lưu file tạm | `tempfile.NamedTemporaryFile()` | File object | Temp file path |
| **→** | **4.4** | Đọc video | `cv2.VideoCapture()` | Temp file path | Video object |
| **→** | **4.5** | Loop qua frames | `cap.read()` | Video object | Frames (mỗi N frame) |
| **→** | **4.6** | Phát hiện khuôn mặt | MediaPipe | Frame | Face bounding box |
| **→** | **4.7** | Cắt khuôn mặt | Array slicing | Frame + BBox | Face image |
| **→** | **4.8** | Transform ảnh | `inference_transform` | Face image | Tensor |
| **→** | **4.9** | Dự đoán | `model(image_tensor)` | Tensor | Prediction + Confidence |
| **→** | **4.10** | Tổng hợp kết quả | Logic aggregation | Tất cả predictions | Final verdict |
| **→** | **4.11** | Trả về kết quả | `jsonify()` | Verdict + Details | JSON response |

### 5.2. Chi tiết logic tổng hợp kết quả

| Bước | Điều kiện | Hành động | Kết quả |
|------|-----------|-----------|---------|
| **4.10.1** | Không phát hiện khuôn mặt | Return error | Error message |
| **→** | **4.10.2** | Đếm fake evidence | `fake_evidence_count` | Số frame có FAKE |
| **→** | **4.10.3** | Tính tỷ lệ fake | `fake_ratio = fake_count / total_faces` | Tỷ lệ % |
| **→** | **4.10.4** | Quyết định | `fake_ratio >= 0.3`<br>HOẶC<br>`fake_count >= 3 AND confidence >= 0.85` | **FAKE** |
| **→** | **4.10.5** | Ngược lại | Không thỏa điều kiện trên | **REAL** |

---

## 6. SƠ ĐỒ TỔNG HỢP

### 6.1. Luồng dữ liệu tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 1: TIỀN XỬ LÝ                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Raw Video Files (MP4)             │
        │   - Original sequences              │
        │   - Manipulated sequences           │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Video Processing                  │
        │   1. Read video                     │
        │   2. Uniform Sampling / Padding     │
        │   3. Face Detection (MediaPipe)     │
        │   4. Crop face                      │
        │   5. Save face images               │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Processed Images (PNG)            │
        │   processed_data/                   │
        │   ├── train/FAKE, REAL              │
        │   ├── val/FAKE, REAL                │
        │   └── test/FAKE, REAL               │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Load Dataset                      │
        │   - DataLoader                      │
        │   - Transforms + Augmentation       │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Build Model                        │
        │   - EfficientNet-B4 (pretrained)     │
        │   - Modify classifier (2 classes)     │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Training Loop                      │
        │   - Forward pass                     │
        │   - Loss calculation                 │
        │   - Backward pass                    │
        │   - Gradient clipping                │
        │   - Weight update                    │
        │   - Validation                       │
        │   - LR scheduling                    │
        │   - Early stopping                   │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Save Checkpoint                   │
        │   - checkpoint.pth.tar               │
        │   - model_best.pth.tar              │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GIAI ĐOẠN 3: ĐÁNH GIÁ MÔ HÌNH                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Load Best Model                   │
        │   model_best.pth.tar                │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Evaluate on Test Set              │
        │   - Predictions                     │
        │   - Calculate metrics               │
        │   - Confusion matrix                │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Evaluation Report                 │
        │   - Accuracy, Precision, Recall    │
        │   - F1-Score                       │
        │   - Confusion matrix plot           │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GIAI ĐOẠN 4: DỰ ĐOÁN (INFERENCE)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   User Upload Video                 │
        │   (via Web App)                     │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Process Video                     │
        │   1. Extract frames                 │
        │   2. Face detection                 │
        │   3. Crop faces                     │
        │   4. Predict each face              │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Aggregate Results                 │
        │   - Count fake evidence             │
        │   - Calculate ratio                 │
        │   - Make final decision             │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Return Result                     │
        │   - Verdict: FAKE/REAL              │
        │   - Confidence score                 │
        │   - Detailed statistics             │
        └─────────────────────────────────────┘
```

### 6.2. Bảng tóm tắt các thành phần chính

| Thành phần | Công cụ/Kỹ thuật | Vai trò |
|------------|------------------|---------|
| **Face Detection** | MediaPipe Face Detection | Phát hiện và định vị khuôn mặt trong frame |
| **Frame Sampling** | Uniform Sampling + Temporal Padding | Chọn frame đại diện từ video |
| **Model Architecture** | EfficientNet-B4 (timm) | Mô hình CNN phân loại ảnh |
| **Transfer Learning** | ImageNet pretrained weights | Tận dụng kiến thức từ ImageNet |
| **Data Augmentation** | RandomHorizontalFlip, ColorJitter, Rotation | Tăng tính đa dạng dữ liệu |
| **Loss Function** | CrossEntropyLoss với class weights | Xử lý mất cân bằng dữ liệu |
| **Optimizer** | Adam với weight decay | Tối ưu hóa tham số mô hình |
| **Learning Rate** | ReduceLROnPlateau | Điều chỉnh LR tự động |
| **Regularization** | Weight decay (L2) + Gradient clipping | Tránh overfitting |
| **Mixed Precision** | FP16 training | Tăng tốc độ và giảm bộ nhớ |
| **Early Stopping** | Patience-based | Dừng sớm khi không cải thiện |
| **Web Framework** | Flask | Ứng dụng web để inference |

---

## 7. CÁC THAM SỐ QUAN TRỌNG

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `NUM_FRAMES_PER_VIDEO` | 10 | Số frame trích xuất từ mỗi video |
| `IMAGE_SIZE` | (224, 224) | Kích thước ảnh đầu vào |
| `BATCH_SIZE` | 16 (GPU) / 2 (CPU) | Số mẫu mỗi batch |
| `NUM_EPOCHS` | 7 | Số epoch tối đa |
| `LEARNING_RATE` | 0.0005 | Tốc độ học ban đầu |
| `WEIGHT_DECAY` | 1e-5 | Hệ số L2 regularization |
| `EARLY_STOPPING_PATIENCE` | 4 | Số epoch chờ trước khi dừng |
| `SKIP_FRAMES` | 10 | Bỏ qua N-1 frame khi inference |
| `FACE_MARGIN` | 20 | Margin khi cắt khuôn mặt (pixels) |
| `EVIDENCE_THRESHOLD` | 0.80 | Ngưỡng confidence để coi là FAKE |
| `FACE_DETECTION_CONFIDENCE` | 0.5 | Ngưỡng confidence cho face detection |

---

## 8. CÁC FILE QUAN TRỌNG TRONG DỰ ÁN

| File/Thư mục | Vai trò | Giai đoạn |
|--------------|--------|-----------|
| `src/data_processing/preprocess.py` | Tiền xử lý video | Giai đoạn 1 |
| `src/training/dataset.py` | Dataset class | Giai đoạn 2 |
| `src/training/train.py` | Training script | Giai đoạn 2 |
| `src/training/evaluate.py` | Evaluation script | Giai đoạn 3 |
| `src/app/main_app.py` | Web application | Giai đoạn 4 |
| `configs/config.py` | Cấu hình | Tất cả |
| `main.py` | Entry point | Tất cả |
| `processed_data/` | Dữ liệu đã xử lý | Giai đoạn 2, 3 |
| `saved_models/` | Checkpoints | Giai đoạn 2, 3, 4 |
| `evaluation_results/` | Kết quả đánh giá | Giai đoạn 3 |

---

*Tài liệu này mô tả toàn bộ quy trình xử lý của hệ thống phát hiện Deepfake từ giai đoạn tiền xử lý đến inference.*

