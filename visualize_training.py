# visualize_training.py - Script để xem lịch sử training

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_history(log_file_path):
    """Vẽ biểu đồ lịch sử training từ CSV log"""
    
    if not os.path.exists(log_file_path):
        print(f"❌ Không tìm thấy file log: {log_file_path}")
        return
    
    # Đọc dữ liệu
    df = pd.read_csv(log_file_path)
    print("📊 LỊCH SỬ TRAINING:")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    # Tính toán improvements
    if len(df) > 1:
        print("\n📈 CẢI THIỆN QUA CÁC EPOCHS:")
        for i in range(1, len(df)):
            train_loss_change = ((df.iloc[i]['train_loss'] - df.iloc[i-1]['train_loss']) / df.iloc[i-1]['train_loss']) * 100
            val_loss_change = ((df.iloc[i]['val_loss'] - df.iloc[i-1]['val_loss']) / df.iloc[i-1]['val_loss']) * 100
            train_acc_change = ((df.iloc[i]['train_acc'] - df.iloc[i-1]['train_acc']) / df.iloc[i-1]['train_acc']) * 100
            val_acc_change = ((df.iloc[i]['val_acc'] - df.iloc[i-1]['val_acc']) / df.iloc[i-1]['val_acc']) * 100
            
            print(f"Epoch {i-1} → {i}:")
            print(f"  🔻 Train Loss: {train_loss_change:+.1f}%")
            print(f"  🔻 Val Loss: {val_loss_change:+.1f}%")
            print(f"  🔺 Train Acc: {train_acc_change:+.1f}%")
            print(f"  🔺 Val Acc: {val_acc_change:+.1f}%")
            print()
    
    # Vẽ biểu đồ
    # Sử dụng style tương thích với nhiều phiên bản matplotlib
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')  # Fallback to default style
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(df['epoch'], df['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(df['epoch'], df['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(df['epoch'], df['learning_rate'], 'g-^', label='Learning Rate', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined Loss vs Accuracy
    ax4_twin = ax4.twinx()
    ax4.plot(df['epoch'], df['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax4_twin.plot(df['epoch'], df['val_acc'], 'b-o', label='Val Accuracy', linewidth=2)
    ax4.set_title('Validation Loss vs Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss', color='red')
    ax4_twin.set_ylabel('Validation Accuracy', color='blue')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    output_path = 'evaluation_results/training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Đã lưu biểu đồ tại: {output_path}")
    
    # Hiển thị biểu đồ
    plt.show()

def print_latest_stats(log_file_path):
    """In thống kê mới nhất"""
    if not os.path.exists(log_file_path):
        return
        
    df = pd.read_csv(log_file_path)
    if len(df) == 0:
        return
        
    latest = df.iloc[-1]
    print(f"\n🏆 THỐNG KÊ MỚI NHẤT (Epoch {int(latest['epoch'])}):")
    print(f"  📉 Train Loss: {latest['train_loss']:.4f}")
    print(f"  📈 Train Accuracy: {latest['train_acc']:.4f} ({latest['train_acc']*100:.2f}%)")
    print(f"  📉 Val Loss: {latest['val_loss']:.4f}")
    print(f"  📈 Val Accuracy: {latest['val_acc']:.4f} ({latest['val_acc']*100:.2f}%)")
    print(f"  🎯 Learning Rate: {latest['learning_rate']}")

if __name__ == "__main__":
    log_file = "evaluation_results/training_log.csv"
    
    print("🚀 TRAINING HISTORY VISUALIZER")
    print("="*50)
    
    # In thống kê latest
    print_latest_stats(log_file)
    
    # Vẽ biểu đồ
    plot_training_history(log_file)
