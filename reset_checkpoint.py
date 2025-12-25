#!/usr/bin/env python3
"""
Script để reset checkpoint về một epoch cụ thể.
Hỗ trợ reset cả checkpoint.pth.tar và model_best.pth.tar
"""

import torch
import os
import argparse
import shutil
from datetime import datetime

def reset_checkpoint(checkpoint_path, target_epoch=1, target_val_acc=None, backup=True):
    """
    Reset checkpoint về một epoch cụ thể.
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint cần reset
        target_epoch: Epoch muốn reset về (mặc định 1)
        target_val_acc: Best validation accuracy muốn set (optional)
        backup: Có tạo backup không (mặc định True)
    """
    if not os.path.exists(checkpoint_path):
        print(f'❌ Không tìm thấy checkpoint tại: {checkpoint_path}')
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print('=== THÔNG TIN CHECKPOINT HIỆN TẠI ===')
        print(f'Epoch: {checkpoint.get("epoch", "Unknown")}')
        print(f'Best val acc: {checkpoint.get("best_val_acc", "Unknown")}')
        
        # Tạo backup nếu cần
        if backup:
            backup_path = f"{checkpoint_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(checkpoint_path, backup_path)
            print(f'✅ Đã tạo backup tại: {backup_path}')
        
        # Reset checkpoint
        checkpoint['epoch'] = target_epoch
        if target_val_acc is not None:
            checkpoint['best_val_acc'] = target_val_acc
        
        # Save lại
        torch.save(checkpoint, checkpoint_path)
        print(f'✅ Đã reset checkpoint về epoch {target_epoch}')
        if target_val_acc is not None:
            print(f'✅ Đã set best_val_acc = {target_val_acc}')
        print(f'Training sẽ bắt đầu từ epoch {target_epoch + 1}')
        return True
        
    except Exception as e:
        print(f'❌ Lỗi khi reset checkpoint: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Reset checkpoint về một epoch cụ thể')
    parser.add_argument('--checkpoint', type=str, default='saved_models/checkpoint.pth.tar',
                        help='Đường dẫn đến checkpoint (mặc định: saved_models/checkpoint.pth.tar)')
    parser.add_argument('--best-model', action='store_true',
                        help='Reset model_best.pth.tar thay vì checkpoint.pth.tar')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Epoch muốn reset về (mặc định: 1)')
    parser.add_argument('--val-acc', type=float, default=None,
                        help='Best validation accuracy muốn set (optional)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Không tạo backup trước khi reset')
    
    args = parser.parse_args()
    
    # Xác định checkpoint path
    if args.best_model:
        checkpoint_path = 'saved_models/model_best.pth.tar'
    else:
        checkpoint_path = args.checkpoint
    
    # Reset checkpoint
    success = reset_checkpoint(
        checkpoint_path=checkpoint_path,
        target_epoch=args.epoch,
        target_val_acc=args.val_acc,
        backup=not args.no_backup
    )
    
    if success:
        print('\n✅ Hoàn tất!')
    else:
        print('\n❌ Thất bại!')
        exit(1)

if __name__ == '__main__':
    main()
