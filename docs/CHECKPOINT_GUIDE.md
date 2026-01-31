# 💾 Checkpoint System - Không bị mất dữ liệu khi training!

**Mục đích**: Lưu training state mỗi epoch để có thể resume khi bị gián đoạn

---

## ✅ **ĐÃ IMPLEMENT CHECKPOINT SYSTEM**

### **Tính năng:**
- 💾 **Auto-save checkpoint mỗi epoch**
- 🔄 **Resume training từ checkpoint**
- 📦 **Lưu đầy đủ state**: model, optimizer, scheduler, scaler, best metrics
- 🎯 **Lưu 2 loại checkpoint**:
  - `best_model.pth` - Model tốt nhất (theo val_acc)
  - `latest_checkpoint.pth` - Checkpoint mới nhất (để resume)

---

## 📁 **CẤU TRÚC CHECKPOINT**

```
/kaggle/working/saved_models/
├── standard/
│   ├── best_model.pth              ← Model tốt nhất
│   └── checkpoints/
│       └── latest_checkpoint.pth   ← Full checkpoint để resume
└── advanced/
    ├── best_temporal_model.pth
    └── checkpoints/
        └── latest_checkpoint.pth
```

---

## 🔍 **CHECKPOINT CHỨA GÌ?**

```python
checkpoint = {
    'epoch': epoch,                              # Epoch hiện tại
    'model_state_dict': model.state_dict(),      # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'scheduler_state_dict': scheduler.state_dict(),  # Scheduler state  
    'scaler_state_dict': scaler.state_dict(),        # AMP scaler state
    'best_val_acc': best_val_acc,                # Best validation accuracy
    'best_val_loss': best_val_loss,              # Best validation loss (Advanced)
    'early_stop_counter': early_stop_counter,    # Early stopping counter (Advanced)
    'train_loss': train_loss,                    # Training loss epoch hiện tại
    'train_acc': train_acc,                      # Training accuracy
    'val_loss': val_loss,                        # Validation loss
    'val_acc': val_acc,                          # Validation accuracy
    'history': training_history                  # Toàn bộ lịch sử training
}
```

→ **Có thể resume CHÍNH XÁC từ epoch bị gián đoạn!**

---

## 🚀 **CÁCH SỬ DỤNG**

### **1. Training lần đầu (Fresh Start)**

```python
# Cell 5 (Standard) hoặc Cell 7 (Advanced)
RESUME_FROM_CHECKPOINT = True  # ✅ Luôn bật (mặc định)
SAVE_CHECKPOINT_EVERY = 1      # Lưu mỗi 1 epoch

# Run training → Checkpoint sẽ tự động lưu mỗi epoch
```

**Output:**
```
🆕 Starting fresh training (no checkpoint found)

Epoch 1/10
...
💾 Saved checkpoint at epoch 1
🎉 Saved best model! Val Acc: 0.9234

Epoch 2/10
...
💾 Saved checkpoint at epoch 2
```

---

### **2. Training bị gián đoạn → Resume**

**Scenario:** Training bị crash/timeout tại epoch 5

```python
# Không cần làm gì! Code tự động detect checkpoint
RESUME_FROM_CHECKPOINT = True  # ← Đã bật sẵn

# Run lại Cell 5/7 → Tự động resume
```

**Output:**
```
============================================================
🔄 RESUMING FROM CHECKPOINT...
============================================================
✅ Resumed from epoch 4
✅ Best Val Acc so far: 0.9234
✅ Will continue from epoch 5
============================================================

Epoch 5/10  ← Tiếp tục từ đây!
...
```

---

### **3. Training bị timeout trên Kaggle**

**Vấn đề:** Kaggle limit 12h/session → Chưa train xong

**Giải pháp:**

**Session 1 (12h):**
```python
NUM_EPOCHS = 10
# Train đến hết 12h → Auto save checkpoint tại epoch 8
```

**Session 2 (tiếp tục):**
```python
# Restart kernel
# Run lại cells → Resume từ epoch 8
# Train 2 epochs còn lại → Done!
```

---

## 🔧 **CẤU HÌNH CHECKPOINT**

### **Thay đổi tần suất lưu checkpoint:**

```python
# Cell 5 (Standard Training)
SAVE_CHECKPOINT_EVERY = 2  # Lưu mỗi 2 epochs thay vì 1

# Cell 7 (Advanced Training)  
SAVE_CHECKPOINT_EVERY_ADV = 3  # Lưu mỗi 3 epochs
```

**Recommend:**
- Fast Training: `SAVE_CHECKPOINT_EVERY = 1` (mỗi epoch ~10 min)
- Full Training: `SAVE_CHECKPOINT_EVERY = 1` (mỗi epoch ~1-2h)

---

### **Tắt resume (train từ đầu):**

```python
RESUME_FROM_CHECKPOINT = False  # ← Tắt resume

# Lưu ý: Checkpoint cũ vẫn còn, chỉ không load thôi
# Muốn xóa checkpoint: xóa folder checkpoints/
```

---

## 📊 **MONITORING CHECKPOINTS**

### **Check checkpoint có tồn tại không:**

```python
import os

checkpoint_path = '/kaggle/working/saved_models/standard/checkpoints/latest_checkpoint.pth'
if os.path.exists(checkpoint_path):
    print("✅ Checkpoint found!")
    
    # Load để xem thông tin
    checkpoint = torch.load(checkpoint_path)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best Val Acc: {checkpoint['best_val_acc']:.4f}")
else:
    print("❌ No checkpoint")
```

---

### **Download checkpoint về local:**

Từ Kaggle notebook output tab:
```
/kaggle/working/
├── saved_models/
│   ├── standard/
│   │   ├── best_model.pth              ← Download này để inference
│   │   └── checkpoints/
│   │       └── latest_checkpoint.pth   ← Download để resume local
```

**Restore trên local:**
```python
# Local machine
checkpoint = torch.load('latest_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Continue training...
```

---

## ⚠️ **LƯU Ý QUAN TRỌNG**

### **1. Checkpoint vs Best Model**

| File | Mục đích | Kích thước | Khi nào dùng |
|------|----------|------------|--------------|
| `best_model.pth` | Inference | ~80MB | Deploy model |
| `latest_checkpoint.pth` | Resume training | ~320MB | Resume interrupted training |

**💡 Tip:** 
- Chỉ download `best_model.pth` để inference
- Download `latest_checkpoint.pth` nếu muốn train tiếp

---

### **2. Disk Space**

**Checkpoint size:**
- Standard model: ~320MB per checkpoint
- Advanced model: ~320MB per checkpoint
- Total: ~640MB (nếu train cả 2)

**Kaggle workspace:** 20GB → Đủ dư!

**Nếu muốn tiết kiệm:**
```python
SAVE_CHECKPOINT_EVERY = 5  # Lưu ít hơn
```

---

### **3. Early Stopping + Checkpoint**

**Advanced model có early stopping:**
```python
# Nếu early stop tại epoch 12:
if early_stop_counter >= EARLY_STOP_PATIENCE:
    print("🛑 EARLY STOPPING")
    break  # ← Stop training
    
# Checkpoint tại epoch 12 vẫn được lưu!
# Resume sẽ tiếp tục từ epoch 12 (nhưng sẽ stop ngay)
```

**Nếu muốn tiếp tục train sau early stop:**
```python
# Option 1: Tăng patience
EARLY_STOP_PATIENCE = 10  # Thay vì 5

# Option 2: Reset early stop counter
checkpoint = torch.load(checkpoint_path)
checkpoint['early_stop_counter'] = 0  # Reset
torch.save(checkpoint, checkpoint_path)
```

---

## 🎯 **USE CASES**

### **Use Case 1: Kaggle Timeout**
```
Session 1 (12h):
  ├─ Epoch 1-8 ✅
  ├─ Save checkpoint at epoch 8
  └─ Timeout

Session 2 (continue):
  ├─ Resume from epoch 8
  ├─ Epoch 9-10 ✅
  └─ Done!
```

---

### **Use Case 2: Experiment với Hyperparameters**

```python
# Train 5 epochs để test
NUM_EPOCHS = 5
# ... training ...
# Val acc = 0.92 → OK!

# Load checkpoint và train thêm
NUM_EPOCHS = 15  # Train thêm 10 epochs nữa
RESUME_FROM_CHECKPOINT = True
# → Tiếp tục từ epoch 5 đến 15
```

---

### **Use Case 3: OOM Error ở epoch giữa chừng**

```
Epoch 1-3: OK
Epoch 4: OOM Error! ← Crash

Solution:
1. Restart kernel
2. Giảm batch size: 32 → 24
3. Resume từ checkpoint epoch 3
4. Continue training với batch size mới
```

---

## 📋 **TROUBLESHOOTING**

### **Lỗi: "RuntimeError: checkpoint mismatch"**

**Nguyên nhân:** Model architecture thay đổi giữa sessions

**Giải pháp:**
```python
# Xóa checkpoint cũ
!rm -rf /kaggle/working/saved_models/standard/checkpoints/*
!rm -rf /kaggle/working/saved_models/advanced/checkpoints/*

# Train lại từ đầu
RESUME_FROM_CHECKPOINT = False
```

---

### **Lỗi: "Checkpoint found but resume failed"**

**Debug:**
```python
try:
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint keys:", checkpoint.keys())
    print("Epoch:", checkpoint.get('epoch'))
except Exception as e:
    print(f"Error: {e}")
    # Có thể checkpoint bị corrupt → Xóa và train lại
```

---

## ✅ **BEST PRACTICES**

1. **✅ Luôn bật resume:**
   ```python
   RESUME_FROM_CHECKPOINT = True  # Default
   ```

2. **✅ Save mỗi epoch cho Fast Training:**
   ```python
   SAVE_CHECKPOINT_EVERY = 1  # Epoch chỉ 10 min
   ```

3. **✅ Download checkpoint khi training xong:**
   - Best model → Deploy
   - Latest checkpoint → Backup

4. **✅ Monitor disk space:**
   ```python
   !df -h /kaggle/working  # Check disk usage
   ```

5. **✅ Test resume trước khi train lâu:**
   ```python
   # Train 2 epochs
   NUM_EPOCHS = 2
   # Stop, restart kernel
   # Resume → Check nó có work không
   ```

---

## 🎁 **BONUS: Manual Checkpoint Loading**

```python
# Load checkpoint thủ công để inspect
checkpoint_path = '/kaggle/working/saved_models/standard/checkpoints/latest_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=== CHECKPOINT INFO ===")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Val Acc: {checkpoint['best_val_acc']:.4f}")
print(f"Train Acc: {checkpoint['train_acc']:.4f}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")

# View history
if 'history' in checkpoint:
    import pandas as pd
    df = pd.DataFrame(checkpoint['history'])
    print("\nTraining History:")
    print(df)
```

---

## 🚀 **TL;DR**

**Checkpoint system:**
```python
# ✅ Auto-save mỗi epoch
# ✅ Auto-resume when restart
# ✅ Không mất data
# ✅ Just run - it works!
```

**Không cần làm gì cả - Mọi thứ đã auto!** 🎉
