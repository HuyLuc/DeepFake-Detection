# configs/config_colab.py (Phiên bản để chạy trên GOOGLE COLAB)

import torch
import os

# ==============================================================================
# --- 1. CẤU HÌNH ĐƯỜNG DẪN (PATH CONFIGURATION) ---
# ==============================================================================
# Colab sử dụng /content/ làm thư mục gốc
BASE_DIR = '/content/DeepFake-Detection'

# Đường dẫn đến dữ liệu - có thể từ Google Drive hoặc upload
# Option 1: Từ Google Drive (khuyến nghị)
DRIVE_MOUNT = '/content/drive'
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/DeepFake-Detection'

# Option 2: Từ /content (upload trực tiếp)
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'all')

# Các thư mục này sẽ được tự động tạo
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# ==============================================================================
# --- CẤU HÌNH GOOGLE DRIVE (AUTO-SYNC) ---
# ==============================================================================
# BẬT tự động lưu vào Google Drive (KHÔNG BAO GIỜ TẮT trên Colab!)
USE_DRIVE_FOR_CHECKPOINTS = True
USE_DRIVE_FOR_LOGS = True
USE_DRIVE_FOR_MODELS = True
AUTO_SYNC_EVERY_EPOCH = True  # Tự động sync sau mỗi epoch

# Đường dẫn trên Google Drive
DRIVE_CHECKPOINT_DIR = os.path.join(DRIVE_PROJECT_DIR, 'saved_models')
DRIVE_LOG_DIR = os.path.join(DRIVE_PROJECT_DIR, 'evaluation_results')
DRIVE_MODEL_DIR = os.path.join(DRIVE_PROJECT_DIR, 'saved_models')

# Tự động tạo các thư mục local
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Tự động tạo các thư mục trên Drive (nếu Drive đã mount)
if USE_DRIVE_FOR_CHECKPOINTS or USE_DRIVE_FOR_LOGS or USE_DRIVE_FOR_MODELS:
    try:
        if os.path.exists(DRIVE_MOUNT):
            os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(DRIVE_LOG_DIR, exist_ok=True)
            os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
            print(f"✅ Đã tạo thư mục trên Google Drive")
    except Exception as e:
        print(f"⚠️ Chưa mount Google Drive: {e}")
        print("   Vui lòng mount Drive trước khi training!")

# Đường dẫn đến các nguồn dữ liệu thô
ORIGINAL_DIRS = {
    'youtube': os.path.join(DATA_ROOT, "original_sequences", "youtube", "c23", "videos"),
    'actors': os.path.join(DATA_ROOT, "original_sequences", "actors", "c23", "videos")
}
MANIPULATION_DIRS = {
    'Deepfakes': os.path.join(DATA_ROOT, "manipulated_sequences", "Deepfakes", "c23", "videos"),
    'Face2Face': os.path.join(DATA_ROOT, "manipulated_sequences", "Face2Face", "c23", "videos"),
    'FaceSwap': os.path.join(DATA_ROOT, "manipulated_sequences", "FaceSwap", "c23", "videos"),
    'NeuralTextures': os.path.join(DATA_ROOT, "manipulated_sequences", "NeuralTextures", "c23", "videos"),
    'DeepFakeDetection': os.path.join(DATA_ROOT, "manipulated_sequences", "DeepFakeDetection", "c23", "videos"),
    'FaceShifter': os.path.join(DATA_ROOT, "manipulated_sequences", "FaceShifter", "c23", "videos")
}

# ==============================================================================
# --- 2. CẤU HÌNH TIỀN XỬ LÝ (PREPROCESSING CONFIGURATION) ---
# ==============================================================================
COMPRESSION_LEVEL = 'c23'
NUM_FRAMES_PER_VIDEO = 10
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# ==============================================================================
# --- 3. CẤU HÌNH HUẤN LUYỆN (TRAINING CONFIGURATION) ---
# ==============================================================================
# Colab luôn có GPU (T4 hoặc V100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'efficientnet_b4'

# --- TỐI ƯU CHO COLAB GPU (T4/V100 với 16GB VRAM) ---
IMAGE_SIZE = (224, 224)
NUM_EPOCHS = 7

# Colab GPU mạnh hơn nhiều so với local
if DEVICE == "cuda":
    # Tối ưu batch size cho T4 (16GB VRAM) và V100/A100
    # T4: Batch size 16-24 an toàn cho EfficientNet-B4
    # V100/A100: Có thể dùng 32-64
    BATCH_SIZE = 16  # An toàn cho T4, có thể tăng lên 24 nếu VRAM đủ
    NUM_WORKERS = 4  # Colab có 2 CPU cores, dùng 4 workers
    PIN_MEMORY = True
    MIXED_PRECISION = True  # Bật mixed precision để tăng tốc ~2x và tiết kiệm VRAM
    GRADIENT_CLIPPING = True
    MAX_GRAD_NORM = 1.0
    PREFETCH_FACTOR = 2
else:
    # Fallback cho CPU (không nên xảy ra trên Colab)
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PIN_MEMORY = False
    MIXED_PRECISION = False

# ĐIỀU CHỈNH ĐỂ TRÁNH OVERFITTING (dựa trên phân tích training log)
LEARNING_RATE = 0.0001  # Giảm từ 0.0005 → 0.0001 để học ổn định hơn, tránh overfitting
WEIGHT_DECAY = 1e-4  # Tăng từ 1e-5 → 1e-4 để tăng regularization, giảm overfitting
ACCUMULATION_STEPS = 1

# ==============================================================================
# --- 4. CẤU HÌNH ỨNG DỤNG WEB (APP CONFIGURATION) ---
# ==============================================================================
EVIDENCE_THRESHOLD = 0.80
CLASS_NAMES = ['FAKE', 'REAL']

MAX_VIDEO_SIZE_MB = 500
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

SKIP_FRAMES = 10
FACE_MARGIN = 20
FACE_DETECTION_CONFIDENCE = 0.5
FACE_DETECTION_MODEL = 1


