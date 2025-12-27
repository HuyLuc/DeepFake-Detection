# main.py (phiên bản cập nhật, tối ưu hóa import)

import argparse
import sys
import os

# Thêm thư mục project và src vào Python path (tương thích với cả local và Colab)
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, project_dir)  # Thêm project root
sys.path.insert(0, src_dir)  # Thêm src directory

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Project Main Runner")
    parser.add_argument('task', choices=['preprocess', 'train', 'evaluate', 'app'], 
                        help="Tác vụ cần thực hiện: 'preprocess', 'train', 'evaluate', hoặc 'app'")
    args = parser.parse_args()

    # --- THAY ĐỔI QUAN TRỌNG: Import bên trong khối if ---

    if args.task == 'preprocess':
        # Chỉ import khi cần chạy preprocess
        from src.data_processing.preprocess import run_preprocessing
        print("🚀 Bắt đầu tác vụ: Tiền xử lý dữ liệu...")
        run_preprocessing()
        print("✅ Hoàn tất Tiền xử lý dữ liệu.")

    elif args.task == 'train':
        # Chỉ import khi cần chạy train
        # Import từ src.training.train vì project_dir đã được thêm vào path
        from src.training.train import run_training
        print("🚀 Bắt đầu tác vụ: Huấn luyện mô hình...")
        run_training()

    elif args.task == 'evaluate':
        # Chỉ import khi cần chạy evaluate
        from src.training.evaluate import run_evaluation
        print("🚀 Bắt đầu tác vụ: Đánh giá mô hình...")
        run_evaluation()

    elif args.task == 'app':
        # Chỉ import khi cần chạy app
        from src.app.main_app import run_app
        print("🚀 Bắt đầu tác vụ: Chạy ứng dụng web...")
        run_app()

if __name__ == '__main__':
    main()