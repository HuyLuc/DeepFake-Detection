# src/architectures/standard/train.py
"""
🔵 STANDARD ARCHITECTURE - Training Script
EfficientNet-B4 đơn giản, phân loại từng frame độc lập.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
import os
import csv
import glob
from torch.amp import autocast, GradScaler
import psutil
import gc
import logging
import sys

# Hack để đảm bảo import được src.utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import từ các file khác trong dự án
from configs import config

# Import Dataset từ cùng thư mục
from .dataset import DeepfakeDataset

# Handle imports linh hoạt cho nhiều môi trường
try:
    from src.utils.utils import save_checkpoint, load_checkpoint, sync_logs_to_drive
except ModuleNotFoundError:
    try:
        from utils.utils import save_checkpoint, load_checkpoint, sync_logs_to_drive
    except ModuleNotFoundError:
        # Fallback: Thêm path thủ công
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.utils.utils import save_checkpoint, load_checkpoint, sync_logs_to_drive

# Thiết lập logging với UTF-8 encoding để tránh lỗi Unicode trên Windows
import sys
if sys.platform == 'win32':
    # Fix encoding cho Windows console
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(config.EVALUATION_RESULTS_DIR, 'training.log'),
            encoding='utf-8'  # Dùng UTF-8 cho file log
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Kiểm tra tài nguyên hệ thống và tối ưu cấu hình"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB ({memory.percent:.1f}%)")
        print(f"💾 RAM khả dụng: {memory.available/1e9:.1f}GB")
        
        # Tự động điều chỉnh NUM_WORKERS dựa trên RAM
        available_ram_gb = memory.available / 1e9
        if available_ram_gb < 4:
            recommended_workers = 0
        elif available_ram_gb < 8:
            recommended_workers = 1
        else:
            recommended_workers = 2
        print(f"🔧 Khuyến nghị NUM_WORKERS: {recommended_workers}")
        
    except ImportError:
        print("💾 Không thể kiểm tra RAM (psutil không có)")
        recommended_workers = 1
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory/1e9
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 VRAM total: {gpu_memory_gb:.1f}GB")
        
        # Tự động điều chỉnh batch size dựa trên VRAM
        if gpu_memory_gb < 3:
            recommended_batch = 4
        elif gpu_memory_gb < 6:
            recommended_batch = 8
        else:
            recommended_batch = 16
        print(f"🔧 Khuyến nghị BATCH_SIZE: {recommended_batch}")
        
        return recommended_workers, recommended_batch
    else:
        print("🎮 GPU: Không có CUDA, sử dụng CPU")
        return 0, 2

def run_training():
    """Hàm chính điều phối toàn bộ quá trình huấn luyện và kiểm định."""
    
    print("--- 🚀 Bắt đầu quá trình huấn luyện ---")
    logger.info("="*50)
    logger.info("Bat dau qua trinh huan luyen")
    logger.info("="*50)
    
    # THÊM: CUDA optimizations (không ảnh hưởng model accuracy)
    if config.DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True      # Tăng tốc convolution
        torch.backends.cudnn.enabled = True        # Bật CuDNN
        print("🚀 CUDA optimizations enabled")
        
        # Warmup GPU để tránh cold start
        dummy_input = torch.randn(1, 3, *config.IMAGE_SIZE).cuda()
        print("🔥 GPU warmup completed")
        del dummy_input  # Cleanup
        torch.cuda.empty_cache()
    
    recommended_workers, recommended_batch = check_system_resources()  # Kiểm tra tài nguyên hệ thống
    
    # Tự động điều chỉnh config dựa trên hardware
    if hasattr(config, 'NUM_WORKERS'):
        actual_workers = min(config.NUM_WORKERS, recommended_workers)
    else:
        actual_workers = recommended_workers
    
    print(f"Thiết bị sử dụng: {config.DEVICE}")
    print(f"Mô hình: {config.MODEL_NAME}")
    print(f"Workers được sử dụng: {actual_workers}")
    print(f"Tham số: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}, LR={config.LEARNING_RATE}")
    
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Workers: {actual_workers}")
    logger.info(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}")

    # --- Thiết lập file log ---
    log_file_path = os.path.join(config.EVALUATION_RESULTS_DIR, 'training_log.csv')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
    
    # --- 1. Chuẩn bị Dữ liệu - SỬ DỤNG DEEPFAKE-SPECIFIC AUGMENTATION ---
    print("\n--- 🎨 Đang thiết lập Data Augmentation ---")
    
    # Import từ module augmentation mới
    from src.data_processing.deepfake_augmentation import (
        get_deepfake_train_transforms, 
        get_deepfake_val_transforms
    )
    
    # Lấy USE_DEEPFAKE_AUGMENTATION từ config, mặc định là True nếu không có
    use_deepfake_aug = getattr(config, 'USE_DEEPFAKE_AUGMENTATION', True)
    
    if use_deepfake_aug:
        print("✅ Sử dụng Deepfake-specific Augmentation:")
        print("   - JPEG Compression (mô phỏng compression artifacts)")
        print("   - Gaussian Noise (mô phỏng camera chất lượng thấp)")
        print("   - Adaptive Blur (mô phỏng video mất nét)")
        print("   - Face Cutout (khuyến khích model học nhiều features)")
        
        data_transforms = {
            'train': get_deepfake_train_transforms(
                image_size=config.IMAGE_SIZE, 
                use_deepfake_aug=True
            ),
            'val': get_deepfake_val_transforms(image_size=config.IMAGE_SIZE)
        }
    else:
        print("⚠️ Sử dụng augmentation cơ bản (không có Deepfake-specific)")
        from torchvision import transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.05, scale=(0.02, 0.1)),
            ]),
            'val': transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    print("\n--- Đang tải dữ liệu ---")
    train_dir = os.path.join(config.PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(config.PROCESSED_DATA_DIR, 'val')
    train_dataset = DeepfakeDataset(data_dir=train_dir, transform=data_transforms['train'])
    val_dataset = DeepfakeDataset(data_dir=val_dir, transform=data_transforms['val'])

    # SỬA: Kiểm tra và áp dụng Oversampling cho train dataset
    print("\n--- ⚖️ Đang thiết lập Data Balancing ---")
    use_oversampling = getattr(config, 'USE_OVERSAMPLING', True)
    
    # SỬA: Tối ưu DataLoader cho tốc độ (không ảnh hưởng model)
    pin_memory_setting = True if config.DEVICE == "cuda" else False
    prefetch_factor = getattr(config, 'PREFETCH_FACTOR', 2) if actual_workers > 0 else None
    
    if use_oversampling:
        from src.utils.balanced_dataset import get_balanced_dataloader
        
        oversampling_method = getattr(config, 'OVERSAMPLING_METHOD', 'oversampling')
        oversample_ratio = getattr(config, 'OVERSAMPLE_RATIO', 1.3)
        
        print(f"✅ Sử dụng Data Balancing:")
        print(f"   - Method: {oversampling_method}")
        print(f"   - Oversample ratio: {oversample_ratio}")
        
        train_loader = get_balanced_dataloader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=actual_workers,
            pin_memory=pin_memory_setting,
            method=oversampling_method,
            oversample_ratio=oversample_ratio
        )
    else:
        print("⚠️ Không sử dụng oversampling (dùng class weights thay thế)")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=actual_workers,
            pin_memory=pin_memory_setting,
            persistent_workers=True if actual_workers > 0 else False,
            prefetch_factor=prefetch_factor,
            drop_last=False
        )
    
    # Validation loader giữ nguyên (không cần oversampling)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=actual_workers,
        pin_memory=pin_memory_setting,
        persistent_workers=True if actual_workers > 0 else False,
        prefetch_factor=prefetch_factor
    )
    
    print(f"\n📊 Kích thước dataset:")
    print(f"   Tập huấn luyện: {len(train_dataset)} mẫu (gốc)")
    if use_oversampling:
        print(f"   Tập huấn luyện: {len(train_loader.dataset)} mẫu (sau oversampling)")
    print(f"   Tập kiểm định: {len(val_dataset)} mẫu")
    print(f"   Số batch/epoch: {len(train_loader)} (train), {len(val_loader)} (val)")
    
    # --- THÊM MỚI: Gradient Accumulation cho GPU nhỏ ---
    # Đọc accumulation_steps từ config
    accumulation_steps = getattr(config, 'ACCUMULATION_STEPS', 1)
    effective_batch_size = config.BATCH_SIZE * accumulation_steps
    print(f"Actual batch size: {config.BATCH_SIZE}")
    print(f"Effective batch size (with accumulation): {effective_batch_size}")

    # --- 2. Xây dựng Mô hình, Optimizer, Loss ---
    print("\n--- Đang tính toán trọng số lớp ---")
    
    # Giả định các lớp được sắp xếp theo thứ tự alphabet: ['FAKE', 'REAL']
    # Cần đảm bảo thứ tự này khớp với train_dataset.classes
    class_names = train_dataset.classes
    
    # Bug fix: Kiểm tra số lượng lớp trước khi truy cập index
    if len(class_names) < 2:
        raise ValueError(
            f"Dataset phải có ít nhất 2 lớp (FAKE và REAL). "
            f"Hiện tại chỉ tìm thấy {len(class_names)} lớp: {class_names}. "
            f"Vui lòng kiểm tra lại cấu trúc thư mục dữ liệu."
        )
    
    if class_names[0] != 'FAKE' or class_names[1] != 'REAL':
        print("Cảnh báo: Thứ tự lớp không như dự kiến ('FAKE', 'REAL'). Trọng số có thể bị sai.")

    # Đếm số lượng file ảnh trong mỗi lớp của tập train
    num_fake_samples = len(glob.glob(os.path.join(train_dir, 'FAKE', '**', '*.png'), recursive=True))
    num_real_samples = len(glob.glob(os.path.join(train_dir, 'REAL', '**', '*.png'), recursive=True))
    
    if num_fake_samples == 0 or num_real_samples == 0:
        print("Lỗi: Không tìm thấy mẫu cho một trong các lớp. Không thể tính trọng số.")
        class_weights = None
    else:
        total_samples = num_fake_samples + num_real_samples
        # Công thức: weight = total / (n_classes * n_samples_of_class)
        weight_fake = total_samples / (2 * num_fake_samples)
        weight_real = total_samples / (2 * num_real_samples)
        
        class_weights = torch.tensor([weight_fake, weight_real], dtype=torch.float32)

        print(f"Số lượng mẫu: FAKE={num_fake_samples}, REAL={num_real_samples}")
        print(f"Trọng số lớp được tính toán (FAKE, REAL): [{weight_fake:.2f}, {weight_real:.2f}]")
    # ---------------------------------------------------------------------

    # --- 2. Xây dựng Mô hình, Optimizer, Loss ---
    print("\n--- Đang xây dựng mô hình ---")
    model = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=len(train_dataset.classes))
    model = model.to(config.DEVICE)
    
    # Thêm weight_decay để giảm overfitting (L2 regularization)
    weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)
    print(f"Weight decay (L2 regularization): {weight_decay}")
    
    # --- THÊM MỚI: Mixed Precision Scaler (TẮT để tránh NaN) ---
    use_mixed_precision = getattr(config, 'MIXED_PRECISION', True) and config.DEVICE == 'cuda'
    # Sử dụng API mới để tránh FutureWarning
    scaler = GradScaler('cuda') if use_mixed_precision else None
    print(f"Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    
    # --- THÊM MỚI: Gradient Clipping ---
    use_grad_clipping = getattr(config, 'GRADIENT_CLIPPING', True)
    max_grad_norm = getattr(config, 'MAX_GRAD_NORM', 1.0)
    print(f"Gradient clipping: {'Enabled' if use_grad_clipping else 'Disabled'}")
    if use_grad_clipping:
        print(f"Max gradient norm: {max_grad_norm}")
    
    # --- THAY ĐỔI: Truyền trọng số vào hàm loss ---
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
        print("Áp dụng trọng số lớp vào hàm CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Không áp dụng trọng số lớp.")
    # ----------------------------------------------
    
    # Learning rate scheduler: giảm LR khi validation accuracy không cải thiện
    # ĐIỀU CHỈNH: Patience=2 (giảm từ 3) để trigger sớm hơn, tránh overfitting
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    print(f"Learning rate scheduler: ReduceLROnPlateau (patience=2, factor=0.1) - đã điều chỉnh để tránh overfitting")

    # --- 3. Tải Checkpoint (nếu có) và Backup ---
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 'checkpoint.pth.tar')
    
    # Backup checkpoint hiện tại trước khi tiếp tục
    if os.path.exists(checkpoint_path):
        import shutil
        backup_path = os.path.join(config.MODEL_SAVE_DIR, 'checkpoint_backup_epoch2.pth.tar')
        if not os.path.exists(backup_path):
            shutil.copy2(checkpoint_path, backup_path)
            print(f"✅ Đã backup checkpoint hiện tại tại: {backup_path}")
    
    model, optimizer, start_epoch, best_val_acc = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"📊 Tiếp tục training từ epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # --- THÊM MỚI: Early Stopping ---
    # ĐIỀU CHỈNH: Giảm patience từ 4 xuống 2 để dừng sớm hơn khi overfitting
    # Dựa trên phân tích: model bắt đầu overfitting từ epoch 5-6, cần dừng sớm hơn
    early_stopping_patience = 2
    early_stopping_counter = 0
    print(f"Early stopping patience: {early_stopping_patience} epochs (đã điều chỉnh để tránh overfitting nghiêm trọng)")

    # --- 4. Vòng lặp Huấn luyện và Kiểm định ---
    print("\n--- Bắt đầu vòng lặp huấn luyện ---")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 20)

        # THÊM: Aggressive memory cleanup chỉ khi cần (mỗi 2 epochs)
        if epoch % 2 == 0:  # Mỗi 2 epochs thay vì mỗi epoch
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Giai đoạn Huấn luyện với Mixed Precision và Progress Bar chi tiết
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        # THÊM: Progress bar với thông tin chi tiết hơn
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(config.DEVICE, non_blocking=True), labels.to(config.DEVICE, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Chia loss cho accumulation_steps
                    loss = loss / accumulation_steps
                    _, preds = torch.max(outputs, 1)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation - chỉ update sau accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # THÊM: Gradient clipping trước optimizer.step()
                    if use_grad_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training cho CPU với gradient accumulation
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Chia loss
                _, preds = torch.max(outputs, 1)
                loss.backward()
                
                # Chỉ update sau accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # THÊM: Gradient clipping cho CPU training
                    if use_grad_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * accumulation_steps  # Nhân lại để đúng scale
            running_corrects += torch.sum(preds == labels.data)
            
            # THÊM: Update progress bar với loss real-time
            if batch_idx % 100 == 0:  # Mỗi 100 batches
                current_loss = running_loss / ((batch_idx + 1) * config.BATCH_SIZE)
                current_acc = running_corrects.double() / ((batch_idx + 1) * config.BATCH_SIZE)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Giai đoạn Kiểm định với tối ưu
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch_idx, (inputs, labels) in enumerate(pbar_val):
                inputs, labels = inputs.to(config.DEVICE, non_blocking=True), labels.to(config.DEVICE, non_blocking=True)
                
                # THÊM: Mixed precision cho validation (không ảnh hưởng accuracy)
                if scaler:
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                if batch_idx % 50 == 0:
                    current_acc = val_corrects.double() / ((batch_idx + 1) * config.BATCH_SIZE)
                    pbar_val.set_postfix({'Acc': f'{current_acc:.4f}'})

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Ghi log (giữ nguyên logic)
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr}"])

        # Lưu Checkpoint và Early Stopping
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            early_stopping_counter = 0  # Reset counter khi có improvement
            print(f"🎉 New best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs")
            logger.info(f"No improvement for {early_stopping_counter} epochs")
        
        save_checkpoint({
            'epoch': epoch, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'best_val_acc': best_val_acc,
        }, is_best=is_best)
        
        # Tự động sync logs vào Drive sau mỗi epoch (nếu bật)
        if hasattr(config, 'AUTO_SYNC_EVERY_EPOCH') and config.AUTO_SYNC_EVERY_EPOCH:
            sync_logs_to_drive()
        
        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered after {early_stopping_patience} epochs without improvement")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            logger.warning(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            break
        
    print("\n--- ✅ Hoàn tất huấn luyện! ---")
    logger.info("="*50)
    logger.info(f"Hoan tat huan luyen! Best validation accuracy: {best_val_acc:.4f}")
    logger.info("="*50)
    
    # Sync logs lần cuối sau khi training xong
    if hasattr(config, 'USE_DRIVE_FOR_LOGS') and config.USE_DRIVE_FOR_LOGS:
        sync_logs_to_drive()
        print("💾 Đã sync logs vào Google Drive")