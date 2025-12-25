# configs/config.py (Phiên bản để chạy trên MÁY TÍNH CÁ NHÂN)

import torch
import os
import multiprocessing

# ==============================================================================
# --- 1. CẤU HÌNH ĐƯỜNG DẪN (PATH CONFIGURATION) ---
# ==============================================================================
# Lấy đường dẫn thư mục gốc của dự án (ví dụ: E:\DoAn\DeepFake-Detection)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến dữ liệu gốc - sử dụng BASE_DIR để tự động thích ứng
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'all')

# Các thư mục này sẽ được tự động tạo bên trong thư mục dự án của bạn
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# Tự động tạo các thư mục nếu chúng chưa tồn tại
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

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
NUM_FRAMES_PER_VIDEO = 10  # GIẢM từ 30 → 10 để giảm thời gian training ~66%
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# ==============================================================================
# --- 3. CẤU HÌNH HUẤN LUYỆN (TRAINING CONFIGURATION) ---
# ==============================================================================
# Tự động phát hiện GPU trên máy của bạn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'efficientnet_b4'

# --- TỐI ƯU TRIỆT ĐỂ CHO GPU MX130 (2GB VRAM) ---
IMAGE_SIZE = (224, 224)  # Giữ nguyên để tương thích với checkpoint
NUM_EPOCHS = 7  # Tăng lên vì mỗi epoch giờ nhanh hơn
if DEVICE == "cuda":
    # TĂNG BATCH SIZE để tăng tốc độ training đáng kể
    BATCH_SIZE = 16  # Tăng từ 4 → 16 (giảm 75% số iterations)
    NUM_WORKERS = 6  # Tăng từ 2 → 6 để data loading không bị bottleneck
    PIN_MEMORY = True  # Tăng tốc CPU->GPU transfer
    MIXED_PRECISION = True  # BẬT mixed precision để tăng tốc ~2x
    GRADIENT_CLIPPING = True  # THÊM gradient clipping
    MAX_GRAD_NORM = 1.0      # Giới hạn gradient norm
    PREFETCH_FACTOR = 4  # Tăng từ 2 → 4 để prefetch nhiều hơn
else:
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PIN_MEMORY = False
    MIXED_PRECISION = False

LEARNING_RATE = 0.001  # Tăng lại vì batch size lớn hơn

# Gradient accumulation - KHÔNG CẦN vì batch size đã đủ lớn
ACCUMULATION_STEPS = 1  # Tắt accumulation để tăng tốc độ

# ==============================================================================
# --- 4. CẤU HÌNH ỨNG DỤNG WEB (APP CONFIGURATION) ---
# ==============================================================================
EVIDENCE_THRESHOLD = 0.80
CLASS_NAMES = ['FAKE', 'REAL']  # Thứ tự lớp (phải khớp với dataset)

# Giới hạn file upload
MAX_VIDEO_SIZE_MB = 500  # Giới hạn kích thước video (MB)
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}  # Định dạng video cho phép

# Cấu hình xử lý video trong app
SKIP_FRAMES = 10  # Bỏ qua N-1 frame, xử lý 1 frame (để tiết kiệm tài nguyên)
FACE_MARGIN = 20  # Margin (pixels) khi cắt khuôn mặt từ frame

# Cấu hình preprocessing
FACE_DETECTION_CONFIDENCE = 0.5  # Ngưỡng confidence cho face detection
FACE_DETECTION_MODEL = 1  # Model selection cho MediaPipe (0 hoặc 1)
