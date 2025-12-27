"""
Script để phân tích và visualize training progress
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from configs import config

def analyze_training():
    """Phân tích training log và hiển thị biểu đồ"""
    log_file = os.path.join(config.EVALUATION_RESULTS_DIR, 'training_log.csv')
    
    if not os.path.exists(log_file):
        print(f"Không tìm thấy file: {log_file}")
        return
    
    # Đọc dữ liệu
    df = pd.read_csv(log_file)
    
    if len(df) < 2:
        print("Cần ít nhất 2 epochs để phân tích")
        return
    
    print("=" * 60)
    print("📊 PHÂN TÍCH TRAINING PROGRESS")
    print("=" * 60)
    
    # Tính toán các chỉ số
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    print(f"\n📈 Epoch {int(latest['epoch'])}:")
    print(f"   Train Loss: {latest['train_loss']:.4f} (↓ {((previous['train_loss'] - latest['train_loss']) / previous['train_loss'] * 100):.1f}%)")
    print(f"   Train Acc:  {latest['train_acc']*100:.2f}% (↑ {((latest['train_acc'] - previous['train_acc']) * 100):.1f}%)")
    print(f"   Val Loss:   {latest['val_loss']:.4f} (↓ {((previous['val_loss'] - latest['val_loss']) / previous['val_loss'] * 100):.1f}%)")
    print(f"   Val Acc:    {latest['val_acc']*100:.2f}% (↑ {((latest['val_acc'] - previous['val_acc']) * 100):.1f}%)")
    
    # Tính gap
    gap = (latest['train_acc'] - latest['val_acc']) * 100
    print(f"\n🔍 Gap Train/Val: {gap:.2f}%")
    
    if gap < 2:
        print("   ✅ Gap nhỏ - Không overfitting")
    elif gap < 5:
        print("   ⚠️  Gap trung bình - Cần theo dõi")
    else:
        print("   ❌ Gap lớn - Có thể overfitting!")
    
    # Kiểm tra xu hướng
    if len(df) >= 3:
        val_accs = df['val_acc'].values[-3:]
        if val_accs[-1] < val_accs[-2] < val_accs[-3]:
            print("\n⚠️  CẢNH BÁO: Val accuracy giảm 2 epochs liên tiếp!")
        elif val_accs[-1] > val_accs[-2]:
            print("\n✅ Val accuracy đang cải thiện")
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss')
    axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-s', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(df['epoch'], df['train_acc']*100, 'b-o', label='Train Acc')
    axes[0, 1].plot(df['epoch'], df['val_acc']*100, 'r-s', label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([80, 100])
    
    # Gap
    gap_data = (df['train_acc'] - df['val_acc']) * 100
    axes[1, 0].plot(df['epoch'], gap_data, 'g-o')
    axes[1, 0].axhline(y=5, color='r', linestyle='--', label='Overfitting threshold (5%)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gap (%)')
    axes[1, 0].set_title('Train/Val Accuracy Gap')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['learning_rate'], 'm-o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    output_file = os.path.join(config.EVALUATION_RESULTS_DIR, 'training_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Biểu đồ đã lưu tại: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    analyze_training()


