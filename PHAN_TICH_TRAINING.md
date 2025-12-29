# PHÂN TÍCH QUÁ TRÌNH TRAINING

## 📊 DỮ LIỆU TRAINING

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Learning Rate |
|-------|------------|-----------|----------|---------|---------------|
| 0 | 0.6241 | 0.7207 | 0.5096 | 0.7362 | 0.0005 |
| 1 | 0.5629 | 0.7139 | **0.8575** ⚠️ | **0.5895** ⚠️ | 0.0005 |
| 2 | 0.5586 | 0.7284 | 0.4962 | **0.7718** ✅ | 0.0005 |
| 3 | 0.571 | 0.7208 | 0.595 | 0.7218 | 0.0005 |
| 4 | 0.5402 | 0.7394 | **0.4724** ✅ | 0.7655 | 0.0005 |
| 5 | 0.5238 | 0.7507 | 0.5768 | 0.6858 | 0.0005 |
| 6 | 0.5027 | 0.7674 | **0.9693** 🚨 | **0.6285** 🚨 | 0.0005 |

---

## 🚨 CÁC ĐIỂM BẤT THƯỜNG PHÁT HIỆN

### 1. **OVERFITTING NGHIÊM TRỌNG** (Vấn đề chính)

**Biểu hiện:**
- **Epoch 6:** Val loss tăng mạnh từ 0.5768 → **0.9693** (tăng 68%!)
- **Epoch 6:** Val accuracy giảm từ 0.6858 → **0.6285** (giảm 8.3%)
- Trong khi đó, Train loss tiếp tục giảm (0.5238 → 0.5027)
- Train accuracy tiếp tục tăng (0.7507 → 0.7674)

**Phân tích:**
```
Epoch 4 (Best):
- Train Loss: 0.5402, Train Acc: 0.7394
- Val Loss: 0.4724 (thấp nhất), Val Acc: 0.7655

Epoch 6 (Hiện tại):
- Train Loss: 0.5027 (giảm 7% so với epoch 4) ✅
- Train Acc: 0.7674 (tăng 3.8% so với epoch 4) ✅
- Val Loss: 0.9693 (tăng 105% so với epoch 4!) 🚨
- Val Acc: 0.6285 (giảm 17.9% so với epoch 4!) 🚨
```

**Kết luận:** Model đang học quá tốt trên tập train nhưng không tổng quát hóa được cho dữ liệu mới.

---

### 2. **EARLY STOPPING KHÔNG HOẠT ĐỘNG**

**Cấu hình hiện tại:**
- Early stopping patience: **4 epochs**
- Best val acc: **0.7718** (Epoch 2)
- Epoch 4: Val acc = 0.7655 (vẫn tốt, gần best)
- Epoch 5: Val acc = 0.6858 (giảm nhưng chưa trigger)
- Epoch 6: Val acc = 0.6285 (giảm mạnh)

**Vấn đề:**
- Early stopping đếm số epochs **không cải thiện** so với best
- Best val acc = 0.7718 (epoch 2)
- Epoch 3: 0.7218 (giảm) → counter = 1
- Epoch 4: 0.7655 (cải thiện) → counter reset về 0
- Epoch 5: 0.6858 (giảm) → counter = 1
- Epoch 6: 0.6285 (giảm) → counter = 2

**Kết luận:** Early stopping chưa trigger vì counter chỉ = 2, cần 4 epochs không cải thiện. Nhưng model đã overfitting nghiêm trọng!

---

### 3. **LEARNING RATE SCHEDULER KHÔNG HOẠT ĐỘNG**

**Cấu hình:**
- ReduceLROnPlateau với patience = 3
- Mode = 'max' (theo dõi val_acc)
- Factor = 0.1 (giảm xuống 10%)

**Vấn đề:**
- Learning rate vẫn là **0.0005** ở tất cả các epochs
- Scheduler chưa được trigger vì:
  - Epoch 2: Val acc = 0.7718 (best)
  - Epoch 3: 0.7218 (giảm) → patience counter = 1
  - Epoch 4: 0.7655 (cải thiện) → counter reset
  - Epoch 5: 0.6858 (giảm) → counter = 1
  - Epoch 6: 0.6285 (giảm) → counter = 2

**Kết luận:** Scheduler cần 3 epochs không cải thiện mới trigger, nhưng counter bị reset ở epoch 4.

---

### 4. **BIẾN ĐỘNG LỚN Ở EPOCH 1**

**Biểu hiện:**
- Epoch 0: Val loss = 0.5096, Val acc = 0.7362
- Epoch 1: Val loss = **0.8575** (tăng 68%!), Val acc = **0.5895** (giảm 20%!)

**Nguyên nhân có thể:**
- Learning rate quá cao ở đầu training
- Model chưa ổn định
- Batch size hoặc gradient accumulation có vấn đề

---

## 📈 BIỂU ĐỒ XU HƯỚNG

### Train vs Validation Loss:
```
Train Loss:  0.624 → 0.563 → 0.559 → 0.571 → 0.540 → 0.524 → 0.503 ✅ (giảm đều)
Val Loss:    0.510 → 0.858 ⚠️ → 0.496 → 0.595 → 0.472 ✅ → 0.577 → 0.969 🚨 (biến động lớn)
```

### Train vs Validation Accuracy:
```
Train Acc:   0.721 → 0.714 → 0.728 → 0.721 → 0.739 → 0.751 → 0.767 ✅ (tăng đều)
Val Acc:     0.736 → 0.590 ⚠️ → 0.772 ✅ → 0.722 → 0.766 → 0.686 → 0.629 🚨 (giảm mạnh)
```

**Khoảng cách giữa Train và Val:**
- Epoch 4: Train Acc (0.739) vs Val Acc (0.766) → Val tốt hơn! ✅
- Epoch 6: Train Acc (0.767) vs Val Acc (0.629) → Chênh lệch 13.8%! 🚨

---

## 🔍 NGUYÊN NHÂN CÓ THỂ

### 1. **Learning Rate Quá Cao**
- LR = 0.0005 có thể vẫn còn cao sau khi model đã học tốt
- Model "nhảy" quá xa và không tìm được điểm tối ưu tốt

### 2. **Weight Decay Chưa Đủ**
- Weight decay = 1e-5 có thể quá nhỏ
- Model học quá chi tiết các đặc điểm của tập train

### 3. **Data Augmentation Chưa Đủ Mạnh**
- Model có thể đã "nhớ" các đặc điểm cụ thể của tập train
- Cần augmentation mạnh hơn để tăng tính tổng quát

### 4. **Early Stopping Patience Quá Lớn**
- Patience = 4 có thể quá lớn
- Model đã overfitting từ epoch 5 nhưng vẫn tiếp tục train đến epoch 6

---

## ✅ KHUYẾN NGHỊ

### 1. **Dừng Training Ngay Lập Tức**
- Model đã đạt best performance ở **Epoch 2** (val_acc = 0.7718) hoặc **Epoch 4** (val_loss = 0.4724)
- Không nên tiếp tục train vì đã overfitting nghiêm trọng

### 2. **Sử Dụng Model Từ Epoch 2 hoặc Epoch 4**
- Load checkpoint từ epoch 2 (best val_acc) hoặc epoch 4 (best val_loss)
- Model này sẽ tổng quát hóa tốt hơn model ở epoch 6

### 3. **Điều Chỉnh Cấu Hình Cho Lần Train Tiếp Theo**

**a) Giảm Learning Rate:**
```python
LEARNING_RATE = 0.0001  # Giảm từ 0.0005 xuống 0.0001
```

**b) Tăng Weight Decay:**
```python
WEIGHT_DECAY = 1e-4  # Tăng từ 1e-5 lên 1e-4
```

**c) Giảm Early Stopping Patience:**
```python
EARLY_STOPPING_PATIENCE = 2  # Giảm từ 4 xuống 2
```

**d) Tăng Data Augmentation:**
- Thêm các augmentation mạnh hơn như:
  - RandomErasing
  - MixUp
  - CutMix

**e) Thêm Dropout (nếu chưa có):**
- Thêm dropout vào các layer cuối của model

### 4. **Theo Dõi Sát Hơn**
- Kiểm tra validation metrics sau mỗi epoch
- Nếu val loss tăng liên tục 2 epochs → dừng ngay

---

## 📊 KẾT LUẬN

**Tình trạng hiện tại:**
- ✅ Model đã học được (train acc tăng, train loss giảm)
- 🚨 Model đang overfitting nghiêm trọng (val metrics xấu đi rõ rệt)
- ⚠️ Early stopping và LR scheduler chưa hoạt động đúng lúc

**Hành động ngay:**
1. **Dừng training** (nếu đang chạy)
2. **Load model từ epoch 2 hoặc 4** (best performance)
3. **Điều chỉnh cấu hình** và train lại từ đầu

**Model tốt nhất nên sử dụng:**
- **Epoch 2:** Val Accuracy = 0.7718 (77.18%)
- **Epoch 4:** Val Loss = 0.4724 (thấp nhất)

---

*Phân tích dựa trên dữ liệu training từ epoch 0 đến epoch 6*


