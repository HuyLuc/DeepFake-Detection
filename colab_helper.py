"""
Helper script cho Google Colab
Chạy script này để setup và quản lý training trên Colab
"""

import os
import shutil
import sys

def setup_colab():
    """Setup môi trường Colab"""
    print("=" * 60)
    print("🚀 SETUP GOOGLE COLAB")
    print("=" * 60)
    
    # 1. Kiểm tra và mount Drive
    drive_path = '/content/drive'
    if not os.path.exists(drive_path):
        print("\n📌 Bước 1: Mount Google Drive")
        print("Chạy lệnh sau trong Colab:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return False
    
    print("✅ Google Drive đã được mount")
    
    # 2. Kiểm tra config
    config_path = 'configs/config.py'
    config_colab_path = 'configs/config_colab.py'
    
    if os.path.exists(config_colab_path):
        if not os.path.exists(config_path) or not os.path.exists('configs/config_local_backup.py'):
            print("\n📌 Bước 2: Backup và cấu hình")
            if os.path.exists(config_path):
                shutil.copy(config_path, 'configs/config_local_backup.py')
                print("✅ Đã backup config cũ")
            
            shutil.copy(config_colab_path, config_path)
            print("✅ Đã cấu hình cho Colab")
        else:
            print("✅ Config đã được cấu hình")
    else:
        print("⚠️ Không tìm thấy config_colab.py")
        return False
    
    # 3. Kiểm tra GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("\n⚠️ Không có GPU. Vui lòng bật GPU trong Colab settings")
    except ImportError:
        print("\n⚠️ Chưa cài đặt PyTorch")
    
    print("\n" + "=" * 60)
    print("✅ Setup hoàn tất!")
    print("=" * 60)
    return True

def sync_checkpoint_to_drive():
    """Đồng bộ checkpoint vào Google Drive"""
    from configs import config
    
    if not hasattr(config, 'USE_DRIVE_FOR_CHECKPOINTS') or not config.USE_DRIVE_FOR_CHECKPOINTS:
        print("⚠️ Chưa bật USE_DRIVE_FOR_CHECKPOINTS trong config")
        return
    
    checkpoint_dir = config.MODEL_SAVE_DIR
    drive_dir = getattr(config, 'DRIVE_CHECKPOINT_DIR', None)
    
    if not drive_dir:
        print("⚠️ Không tìm thấy DRIVE_CHECKPOINT_DIR")
        return
    
    os.makedirs(drive_dir, exist_ok=True)
    
    files_to_sync = ['checkpoint.pth.tar', 'model_best.pth.tar']
    
    for file in files_to_sync:
        src = os.path.join(checkpoint_dir, file)
        dst = os.path.join(drive_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / (1024**2)
            print(f"✅ Đã sync {file} vào Drive ({size_mb:.2f} MB)")
        else:
            print(f"⚠️ Không tìm thấy {file}")

def sync_logs_to_drive():
    """Đồng bộ log files vào Google Drive"""
    from configs import config
    
    if not hasattr(config, 'USE_DRIVE_FOR_LOGS') or not config.USE_DRIVE_FOR_LOGS:
        print("⚠️ Chưa bật USE_DRIVE_FOR_LOGS trong config")
        return
    
    log_dir = config.EVALUATION_RESULTS_DIR
    drive_log_dir = getattr(config, 'DRIVE_LOG_DIR', None)
    
    if not drive_log_dir:
        print("⚠️ Không tìm thấy DRIVE_LOG_DIR")
        return
    
    os.makedirs(drive_log_dir, exist_ok=True)
    
    log_files = ['training.log', 'training_log.csv']
    
    for file in log_files:
        src = os.path.join(log_dir, file)
        dst = os.path.join(drive_log_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / (1024**2)
            print(f"✅ Đã sync {file} vào Drive ({size_mb:.2f} MB)")
        else:
            print(f"⚠️ Không tìm thấy {file}")

def sync_all_to_drive():
    """Đồng bộ tất cả (checkpoint + logs) vào Drive"""
    print("=" * 60)
    print("💾 ĐỒNG BỘ VÀO GOOGLE DRIVE")
    print("=" * 60)
    sync_checkpoint_to_drive()
    print()
    sync_logs_to_drive()
    print("=" * 60)
    print("✅ Hoàn tất đồng bộ!")
    print("=" * 60)

def sync_checkpoint_from_drive():
    """Tải checkpoint từ Google Drive"""
    from configs import config
    
    if not hasattr(config, 'USE_DRIVE_FOR_CHECKPOINTS') or not config.USE_DRIVE_FOR_CHECKPOINTS:
        print("⚠️ Chưa bật USE_DRIVE_FOR_CHECKPOINTS trong config")
        return
    
    checkpoint_dir = config.MODEL_SAVE_DIR
    drive_dir = getattr(config, 'DRIVE_CHECKPOINT_DIR', None)
    
    if not drive_dir or not os.path.exists(drive_dir):
        print("⚠️ Không tìm thấy DRIVE_CHECKPOINT_DIR")
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    files_to_sync = ['checkpoint.pth.tar', 'model_best.pth.tar']
    
    for file in files_to_sync:
        src = os.path.join(drive_dir, file)
        dst = os.path.join(checkpoint_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Đã tải {file} từ Drive")
        else:
            print(f"⚠️ Không tìm thấy {file} trên Drive")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'setup':
            setup_colab()
        elif command == 'sync-to-drive':
            sync_checkpoint_to_drive()
        elif command == 'sync-logs':
            sync_logs_to_drive()
        elif command == 'sync-all':
            sync_all_to_drive()
        elif command == 'sync-from-drive':
            sync_checkpoint_from_drive()
        else:
            print("Usage: python colab_helper.py [setup|sync-to-drive|sync-logs|sync-all|sync-from-drive]")
    else:
        setup_colab()


