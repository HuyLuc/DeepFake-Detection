# src/architectures/advanced/train.py
"""
🟢 ADVANCED ARCHITECTURE - Training Script
Training cho Temporal và Ensemble models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import csv
import logging
import sys

# Hack để đảm bảo import được src.models và src.utils
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configs
from configs import config

# Import models từ cùng thư mục
from .temporal_model import TemporalDeepfakeModel, LightweightTemporalModel
from .ensemble_model import EnsembleDeepfakeModel, TemporalEnsembleModel, create_model

# Import datasets
try:
    from .temporal_dataset import TemporalDeepfakeDataset, create_temporal_dataloaders
except ImportError:
    # Fallback cho imports
    from src.architectures.advanced.temporal_dataset import TemporalDeepfakeDataset, create_temporal_dataloaders

# Import utils
try:
    from src.utils.utils import save_checkpoint, load_checkpoint
except ModuleNotFoundError:
    try:
        from utils.utils import save_checkpoint, load_checkpoint
    except ModuleNotFoundError:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.utils.utils import save_checkpoint, load_checkpoint

# Import augmentation
from src.data_processing.deepfake_augmentation import (
    get_deepfake_train_transforms,
    get_deepfake_val_transforms
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_temporal_training(
    model_type: str = 'temporal_ensemble',
    sequence_length: int = 10,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    resume_from: str = None
):
    """
    Hàm chính để train Temporal/Ensemble models.
    
    Args:
        model_type: Loại model ('temporal', 'ensemble', 'temporal_ensemble')
        sequence_length: Số frames mỗi sequence
        epochs: Số epochs
        batch_size: Batch size
        learning_rate: Learning rate
        resume_from: Path đến checkpoint để resume
    """
    
    print("=" * 60)
    print("🚀 TEMPORAL/ENSEMBLE MODEL TRAINING")
    print("=" * 60)
    
    device = torch.device(config.DEVICE)
    print(f"📱 Device: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # =========================================================================
    # 1. Setup Data
    # =========================================================================
    print("\n📂 Loading datasets...")
    
    train_dir = os.path.join(config.PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(config.PROCESSED_DATA_DIR, 'val')
    
    train_transform = get_deepfake_train_transforms(
        image_size=config.IMAGE_SIZE,
        use_deepfake_aug=True
    )
    val_transform = get_deepfake_val_transforms(image_size=config.IMAGE_SIZE)
    
    train_dataset = TemporalDeepfakeDataset(
        data_dir=train_dir,
        transform=train_transform,
        sequence_length=sequence_length,
        sampling_strategy='uniform'
    )
    
    val_dataset = TemporalDeepfakeDataset(
        data_dir=val_dir,
        transform=val_transform,
        sequence_length=sequence_length,
        sampling_strategy='uniform'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Train: {len(train_dataset)} videos ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset)} videos ({len(val_loader)} batches)")
    
    # =========================================================================
    # 2. Setup Model
    # =========================================================================
    print(f"\n🏗️ Building model: {model_type}...")
    
    num_classes = len(train_dataset.classes)
    print(f"   Classes: {train_dataset.classes}")
    
    if model_type == 'temporal':
        model = TemporalDeepfakeModel(
            backbone_name='efficientnet_b4',
            num_classes=num_classes,
            lstm_hidden_size=512,
            lstm_num_layers=2,
            bidirectional=True,
            pretrained=True
        )
    
    elif model_type == 'ensemble':
        model = EnsembleDeepfakeModel(
            backbone_configs=[
                {'name': 'efficientnet_b4', 'weight': 0.5},
                {'name': 'swin_tiny_patch4_window7_224', 'weight': 0.5}
            ],
            num_classes=num_classes,
            fusion_method='attention',
            pretrained=True
        )
    
    elif model_type == 'temporal_ensemble':
        model = TemporalEnsembleModel(
            backbone_configs=[
                {'name': 'efficientnet_b4', 'weight': 0.5},
                {'name': 'swin_tiny_patch4_window7_224', 'weight': 0.5}
            ],
            num_classes=num_classes,
            lstm_hidden_size=512,
            lstm_num_layers=2,
            fusion_method='concat',
            pretrained=True
        )
    
    elif model_type == 'lightweight':
        model = LightweightTemporalModel(
            num_classes=num_classes,
            pretrained=True
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    
    # =========================================================================
    # 3. Setup Training Components
    # =========================================================================
    print("\n⚙️ Setting up training...")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss với class weights
    train_labels = [v['label'] for v in train_dataset.videos]
    class_counts = [train_labels.count(i) for i in range(num_classes)]
    class_weights = torch.tensor([1.0 / c for c in class_counts], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"   Class weights: {class_weights.tolist()}")
    
    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    print(f"   Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # Resume từ checkpoint nếu có
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"   Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Resuming from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
    
    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("🏃 STARTING TRAINING")
    print("=" * 60)
    
    # Log file
    log_file = os.path.join(config.EVALUATION_RESULTS_DIR, f'training_{model_type}_log.csv')
    if not os.path.exists(log_file) or start_epoch == 0:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    # Early stopping
    early_stop_patience = 5
    early_stop_counter = 0
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*40}")
        
        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{train_loss/train_total:.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # =====================================================================
        # Validation Phase
        # =====================================================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for sequences, labels in pbar:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                if use_amp:
                    with autocast('cuda'):
                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_loss += loss.item() * labels.size(0)
                
                pbar.set_postfix({
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Log to file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f'{train_loss:.4f}', f'{train_acc:.4f}',
                           f'{val_loss:.4f}', f'{val_acc:.4f}', f'{current_lr:.6f}'])
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            early_stop_counter = 0
            print(f"   🎉 New best validation accuracy: {100*best_val_acc:.2f}%")
        else:
            early_stop_counter += 1
            print(f"   No improvement for {early_stop_counter} epochs")
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'model_type': model_type
        }
        
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f'{model_type}_checkpoint.pth.tar')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(config.MODEL_SAVE_DIR, f'{model_type}_best.pth.tar')
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model to {best_path}")
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping triggered after {early_stop_patience} epochs without improvement")
            break
    
    # =========================================================================
    # 5. Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"   Model type: {model_type}")
    print(f"   Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"   Model saved to: {os.path.join(config.MODEL_SAVE_DIR, f'{model_type}_best.pth.tar')}")
    
    return model, best_val_acc


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Temporal/Ensemble Models')
    parser.add_argument('--model', type=str, default='temporal_ensemble',
                       choices=['temporal', 'ensemble', 'temporal_ensemble', 'lightweight'],
                       help='Model type to train')
    parser.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    run_temporal_training(
        model_type=args.model,
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=args.resume
    )
