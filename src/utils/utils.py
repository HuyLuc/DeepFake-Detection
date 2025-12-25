# src/utils/utils.py (phiên bản cập nhật)

import torch
import os
import shutil
import glob
from typing import Dict, Any, Optional, Tuple, Union

# Import config để các hàm tiện ích có thể truy cập
from configs import config

# --- CÁC HÀM CŨ ---
def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool, 
    filename: str = 'checkpoint.pth.tar', 
    best_filename: str = 'model_best.pth.tar', 
    model_save_dir: Optional[str] = None
) -> None:
    if model_save_dir is None:
        model_save_dir = config.MODEL_SAVE_DIR
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(model_save_dir, best_filename)
        shutil.copyfile(checkpoint_path, best_path)
        print(f"🎉 Saved new best model to {best_path}")

def load_checkpoint(
    checkpoint_path: str, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Tải checkpoint với xử lý lỗi đầy đủ.
    
    Args:
        checkpoint_path: Đường dẫn đến file checkpoint
        model: Model cần load weights vào
        optimizer: Optimizer (optional)
    
    Returns:
        Tuple (model, optimizer, start_epoch, best_val_acc)
    """
    if not os.path.exists(checkpoint_path):
        print(f"🟡 Checkpoint not found at '{checkpoint_path}'. Starting from scratch.")
        return model, optimizer, 0, 0.0
    
    try:
        print(f"✅ Loading checkpoint from '{checkpoint_path}'")
        # Sử dụng weights_only=False để tương thích với các checkpoint cũ
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Kiểm tra checkpoint có đầy đủ thông tin không
        if 'state_dict' not in checkpoint:
            raise ValueError(f"Checkpoint không hợp lệ: thiếu 'state_dict'")
        
        # Kiểm tra số lớp có khớp không (nếu có thông tin trong checkpoint)
        model_state = model.state_dict()
        checkpoint_state = checkpoint['state_dict']
        
        # Kiểm tra key đầu tiên để xác định cấu trúc
        model_keys = set(model_state.keys())
        checkpoint_keys = set(checkpoint_state.keys())
        
        # Nếu có key không khớp, cảnh báo nhưng vẫn thử load
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"⚠️ Cảnh báo: Một số keys trong model không có trong checkpoint: {list(missing_keys)[:5]}...")
        if unexpected_keys:
            print(f"⚠️ Cảnh báo: Một số keys trong checkpoint không có trong model: {list(unexpected_keys)[:5]}...")
        
        # Load state dict với strict=False để bỏ qua các key không khớp
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if optimizer and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"⚠️ Cảnh báo: Không thể load optimizer state: {e}. Sử dụng optimizer mới.")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
        return model, optimizer, start_epoch, best_val_acc
        
    except Exception as e:
        print(f"❌ Lỗi khi load checkpoint: {e}")
        print(f"🟡 Bắt đầu training từ đầu (không load checkpoint)")
        return model, optimizer, 0, 0.0

# --- CÁC HÀM MỚI TỪ CONFIG CŨ ---
def verify_data_structure() -> None:
    """Kiểm tra cấu trúc dữ liệu có đúng không."""
    print("=== KIỂM TRA CẤU TRÚC DỮ LIỆU ===")
    print(f"Data root: {config.DATA_ROOT} | Exists: {os.path.exists(config.DATA_ROOT)}")
    print("\n--- VIDEO GỐC ---")
    for orig_type, dir_path in config.ORIGINAL_DIRS.items():
        count = len(glob.glob(os.path.join(dir_path, '*.mp4'))) if os.path.exists(dir_path) else 0
        print(f"{orig_type.upper()}: {count} videos | Exists: {os.path.exists(dir_path)}")
    print("\n--- VIDEO GIẢ MẠO ---")
    for method, dir_path in config.MANIPULATION_DIRS.items():
        count = len(glob.glob(os.path.join(dir_path, '*.mp4'))) if os.path.exists(dir_path) else 0
        print(f"{method}: {count} videos | Exists: {os.path.exists(dir_path)}")