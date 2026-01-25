# src/architectures/evaluate.py
"""
Script đánh giá mô hình trên tập test.
Dùng chung cho cả Standard và Advanced architectures.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

# Import từ các file khác trong dự án
from configs import config
from src.architectures.standard.dataset import DeepfakeDataset
from src.utils.utils import load_checkpoint

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.EVALUATION_RESULTS_DIR, 'evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Vẽ và lưu ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    print(f"✅ Ma trận nhầm lẫn đã được lưu tại: {save_path}")

def run_evaluation():
    """Hàm chính điều phối quá trình đánh giá mô hình trên tập test."""
    print("--- 🚀 Bắt đầu quá trình đánh giá ---")
    logger.info("="*50)
    logger.info("Bắt đầu quá trình đánh giá")
    logger.info("="*50)
    
    # --- 1. Chuẩn bị Dữ liệu Test ---
    # Chỉ cần transform cơ bản, không cần augmentation
    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = DeepfakeDataset(data_dir=os.path.join(config.PROCESSED_DATA_DIR, 'test'), 
                                   transform=test_transform)
    # Chỉ dùng pin_memory khi có GPU
    pin_memory_setting = True if config.DEVICE == "cuda" else False
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=config.NUM_WORKERS, pin_memory=pin_memory_setting)

    class_names = test_dataset.classes
    print(f"Các lớp: {class_names}")

    # --- 2. Tải Mô hình Tốt nhất ---
    print("\n--- Đang tải mô hình tốt nhất ---")
    model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=len(class_names))
    
    # Đường dẫn đến model tốt nhất đã lưu
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'model_best.pth.tar')
    
    # Chỉ tải model, không cần optimizer
    model, _, _, _ = load_checkpoint(best_model_path, model)
    model = model.to(config.DEVICE)
    model.eval()

    # --- 3. Chạy Dự đoán ---
    print("\n--- Đang chạy dự đoán trên tập test ---")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(config.DEVICE)
            # labels không cần to(device) vì sẽ dùng trên CPU
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 4. Tính toán và Hiển thị Kết quả ---
    print("\n--- Kết quả Đánh giá ---")
    
    # Độ chính xác tổng thể
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy trên tập test: {accuracy:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Báo cáo phân loại chi tiết (Precision, Recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nBáo cáo phân loại:")
    print(report)
    logger.info(f"\nClassification Report:\n{report}")

    # Lưu báo cáo vào file text
    report_path = os.path.join(config.EVALUATION_RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"\n✅ Báo cáo đã được lưu tại: {report_path}")

    # Vẽ và lưu ma trận nhầm lẫn
    cm_path = os.path.join(config.EVALUATION_RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    
    print("\n--- ✅ Hoàn tất đánh giá! ---")
    logger.info("="*50)
    logger.info(f"Hoàn tất đánh giá! Test Accuracy: {accuracy:.4f}")
    logger.info("="*50)