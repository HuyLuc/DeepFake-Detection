# main.py (phiên bản cập nhật với cấu trúc thư mục mới)
"""
Entry point cho DeepFake Detection Project.

Có 2 kiến trúc training:
- 🔵 Standard (train): EfficientNet-B4 đơn giản
- 🟢 Advanced (train_advanced): Temporal + Ensemble models
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Thêm thư mục project và src vào Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, project_dir)
sys.path.insert(0, src_dir)


def main():
    parser = argparse.ArgumentParser(description="DeepFake Detection Project")
    parser.add_argument('task', 
                        choices=['preprocess', 'train', 'train_advanced', 'evaluate', 'app'], 
                        help="Tác vụ cần thực hiện")
    
    # Arguments cho train_advanced
    parser.add_argument('--model', type=str, default='temporal_ensemble',
                       choices=['temporal', 'ensemble', 'temporal_ensemble', 'lightweight'],
                       help='🟢 Model type cho train_advanced')
    parser.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Số epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path checkpoint để resume')
    
    args = parser.parse_args()

    # =========================================================================
    # PREPROCESSING (Dùng chung cho cả 2 kiến trúc)
    # =========================================================================
    if args.task == 'preprocess':
        from src.data_processing.preprocess import run_preprocessing
        print("🚀 Bắt đầu tác vụ: Tiền xử lý dữ liệu...")
        run_preprocessing()
        print("✅ Hoàn tất Tiền xử lý dữ liệu.")

    # =========================================================================
    # 🔵 STANDARD TRAINING (Kiến trúc 1)
    # =========================================================================
    elif args.task == 'train':
        from src.architectures.standard.train import run_training
        print("� Bắt đầu tác vụ: Huấn luyện mô hình STANDARD...")
        run_training()

    # =========================================================================
    # 🟢 ADVANCED TRAINING (Kiến trúc 2)
    # =========================================================================
    elif args.task == 'train_advanced':
        from src.architectures.advanced.train import run_temporal_training
        print(f"� Bắt đầu tác vụ: Huấn luyện mô hình ADVANCED ({args.model})...")
        run_temporal_training(
            model_type=args.model,
            sequence_length=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume
        )

    # =========================================================================
    # EVALUATE & APP
    # =========================================================================
    elif args.task == 'evaluate':
        from src.architectures.evaluate import run_evaluation
        print("🚀 Bắt đầu tác vụ: Đánh giá mô hình...")
        run_evaluation()

    elif args.task == 'app':
        from src.app.main_app import run_app
        print("🚀 Bắt đầu tác vụ: Chạy ứng dụng web...")
        run_app()


if __name__ == '__main__':
    main()

