# 📓 Giải thích các Cell trong Notebook

## Cell 1: Mount Google Drive
**Mục đích:** Kết nối Google Drive với Colab để có thể truy cập files trên Drive

**Tại sao cần:**
- Project của bạn được lưu trên Google Drive
- Colab cần mount Drive để đọc/ghi files
- Cho phép lưu checkpoint và logs vào Drive

---

## Cell 2: Copy project từ Drive ⭐ **QUAN TRỌNG**

**Mục đích:** Copy toàn bộ project từ Google Drive vào Colab runtime để có thể chạy code

**Tại sao cần:**
1. **Colab runtime là môi trường tạm thời:**
   - Mỗi lần mở Colab, bạn có một runtime mới (trống rỗng)
   - Code và data không tự động có sẵn
   - Cần copy từ Drive vào runtime

2. **Project cần có trong Colab để chạy:**
   - Code Python (`src/`, `configs/`, `main.py`)
   - Dataset đã processed (`processed_data/`)
   - Checkpoint (nếu có) - sẽ tải riêng ở Cell 5

3. **Workflow:**
   ```
   Google Drive (lưu trữ lâu dài)
        ↓
   Cell 2: Copy vào Colab runtime
        ↓
   Colab runtime (chạy code)
        ↓
   Cell 7: Training
        ↓
   Tự động lưu lại Drive (checkpoint, logs)
   ```

**Lưu ý:**
- Nếu project đã tồn tại trong runtime, sẽ xóa và copy lại (đảm bảo code mới nhất)
- Quá trình copy có thể mất vài phút nếu project lớn
- Sau khi copy, tất cả code và data đã sẵn sàng trong `/content/DeepFake-Detection`

---

## Cell 3: Cài đặt packages
**Mục đích:** Cài đặt các thư viện Python cần thiết (torch, timm, opencv, ...)

**Tại sao cần:**
- Colab runtime mới không có các package của bạn
- Cần cài đặt để code có thể chạy

---

## Cell 4: Cấu hình cho Colab
**Mục đích:** 
- Copy `config_colab.py` → `config.py` (cấu hình cho Colab)
- Tự động điều chỉnh batch size cho GPU T4

**Tại sao cần:**
- Config local khác với config Colab (đường dẫn, batch size, ...)
- Cần dùng config phù hợp với Colab

---

## Cell 5: Tải checkpoint từ Drive
**Mục đích:** Tải checkpoint đã lưu từ Drive (nếu muốn tiếp tục training từ epoch 2)

**Tại sao cần:**
- Checkpoint được lưu trên Drive (không mất khi runtime disconnect)
- Cần tải về runtime để tiếp tục training
- Nếu không có checkpoint, training sẽ bắt đầu từ đầu

---

## Cell 6: Kiểm tra cấu hình
**Mục đích:** Kiểm tra mọi thứ đã sẵn sàng chưa trước khi training

**Kiểm tra:**
- ✅ GPU và VRAM
- ✅ Batch size phù hợp với GPU
- ✅ Dataset có đầy đủ không
- ✅ Checkpoint có sẵn không
- ✅ Google Drive đã mount chưa

---

## Cell 7: Bắt đầu Training
**Mục đích:** Chạy training script

**Sau khi chạy:**
- Training sẽ tự động lưu checkpoint vào Drive sau mỗi epoch
- Logs sẽ tự động sync vào Drive
- Không cần làm gì thêm!

---

## 🔄 Tóm tắt workflow

```
1. Mount Drive (Cell 1)
   ↓
2. Copy project từ Drive → Colab (Cell 2) ⭐
   ↓
3. Cài packages (Cell 3)
   ↓
4. Cấu hình (Cell 4)
   ↓
5. Tải checkpoint (Cell 5) - Tùy chọn
   ↓
6. Kiểm tra (Cell 6)
   ↓
7. Training (Cell 7)
   ↓
   Tự động lưu vào Drive
```

---

## ❓ Câu hỏi thường gặp

**Q: Tại sao phải copy project mỗi lần?**
A: Vì Colab runtime là tạm thời, mỗi lần mở lại là môi trường mới. Cần copy từ Drive (nơi lưu trữ lâu dài).

**Q: Có thể bỏ qua Cell 2 không?**
A: Không! Nếu không copy, code và data sẽ không có trong Colab, không thể chạy training.

**Q: Copy mất bao lâu?**
A: Tùy kích thước project:
- Code: ~10-50MB → vài giây
- Dataset processed_data: ~10-50GB → 5-30 phút

**Q: Có cách nào nhanh hơn không?**
A: Có thể dùng symlink hoặc mount trực tiếp, nhưng copy là cách an toàn và ổn định nhất.

---

**Tóm lại: Cell 2 là BẮT BUỘC để copy project từ Drive vào Colab runtime!**

