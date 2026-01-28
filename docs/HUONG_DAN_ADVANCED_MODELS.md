# 🚀 Hướng dẫn Training Advanced Models

## Tổng quan

Dự án đã được nâng cấp với 3 kiến trúc model mới:

1. **Temporal Model**: EfficientNet-B4 + LSTM
2. **Ensemble Model**: EfficientNet-B4 + Swin Transformer  
3. **Temporal Ensemble**: Kết hợp cả hai (MẠNH NHẤT)

---

## 📁 Files đã tạo mới

```
src/
├── models/
│   ├── __init__.py
│   ├── temporal_model.py      # EfficientNet + LSTM
│   └── ensemble_model.py      # Ensemble + Temporal Ensemble
│
└── training/
    ├── temporal_dataset.py    # Dataset load sequences
    └── train_temporal.py      # Training script mới
```

---

## 🚀 Cách sử dụng

### 1. Train Temporal Model (EfficientNet + LSTM)

```bash
python main.py train_advanced --model temporal --epochs 10 --batch-size 8
```

### 2. Train Ensemble Model (EfficientNet + Swin)

```bash
python main.py train_advanced --model ensemble --epochs 10 --batch-size 4
```

### 3. Train Temporal Ensemble (FULL - MẠNH NHẤT)

```bash
python main.py train_advanced --model temporal_ensemble --epochs 10 --batch-size 4
```

### 4. Train Lightweight (cho máy yếu)

```bash
python main.py train_advanced --model lightweight --epochs 10 --batch-size 16
```

---

## ⚙️ Các tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--model` | `temporal_ensemble` | Loại model |
| `--seq-len` | `10` | Số frames mỗi sequence |
| `--epochs` | `10` | Số epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--resume` | `None` | Path checkpoint để resume |

---

## 💡 Khuyến nghị cho GPU T4 (16GB)

```bash
# Temporal Ensemble - Đầy đủ sức mạnh
python main.py train_advanced \
    --model temporal_ensemble \
    --seq-len 10 \
    --epochs 15 \
    --batch-size 8 \
    --lr 0.0001
```

---

## 📊 So sánh Models

| Model | Accuracy kỳ vọng | VRAM cần | Thời gian/epoch |
|-------|-----------------|----------|-----------------|
| Standard (train) | ~90% | 2GB | ~10 phút |
| Temporal | ~93% | 4GB | ~15 phút |
| Ensemble | ~92% | 6GB | ~20 phút |
| Temporal Ensemble | ~95% | 8GB | ~25 phút |

---

## 📂 Output

Models sẽ được lưu tại:
- `saved_models/{model_type}_checkpoint.pth.tar` - Checkpoint
- `saved_models/{model_type}_best.pth.tar` - Model tốt nhất

Logs tại:
- `evaluation_results/training_{model_type}_log.csv`
