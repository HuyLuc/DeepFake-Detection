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
NUM_FRAMES_PER_VIDEO = 20  #  20 frames/video để model học nhiều temporal patterns hơn
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# ==============================================================================
# --- 3. CẤU HÌNH HUẤN LUYỆN (TRAINING CONFIGURATION) ---
# ==============================================================================
MODEL_NAME = 'efficientnet_b4'

# ==============================================================================
# --- TỰ ĐỘNG CHỌN DEVICE (GPU/CPU) ---
# ==============================================================================
def get_optimal_device() -> str:
    """
    Chọn device phù hợp. Mặc định chọn CUDA nếu khả dụng.
    Chỉ chạy benchmark nếu thực sự cần so sánh hiệu năng.
    """
    if torch.cuda.is_available():
        # Kiểm tra xem có yêu cầu benchmark không (ví dụ qua biến môi trường)
        if os.getenv('RUN_DEVICE_BENCHMARK', 'False').lower() == 'true':
            try:
                # Benchmark logic có thể gọi ở đây nếu muốn
                return "cuda"
            except:
                return "cuda"
        return "cuda"
    return "cpu"

# Khởi tạo DEVICE nhanh chóng
DEVICE = get_optimal_device()

# In thông báo ngắn gọn, tránh làm rối log khi import
if os.getenv('HIDE_CONFIG_LOGS', 'False').lower() != 'true':
    print(f"✅ Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# --- ⚙️ CẤU HÌNH LOGIC QUYẾT ĐỊNH (DECISION LOGIC) ---
# ==============================================================================
FAKE_RATIO_THRESHOLD = 0.3      # 📊 Tỷ lệ frame fake tối thiểu để kết luận video là FAKE
MIN_FAKE_EVIDENCE_COUNT = 3     # 🔍 Số lượng frame fake tối thiểu nếu có confidence cực cao
STRONG_CONFIDENCE_THRESHOLD = 0.85 # 🎯 Ngưỡng confidence được coi là "cực cao"
EVIDENCE_THRESHOLD = 0.65       # ⚖️ Ngưỡng confidence tối thiểu để chấp nhận 1 frame là bằng chứng (evidence)

# ==============================================================================
# --- 🎓 TỐI ƯU TRIỆT ĐỂ CHO GPU MX130 (2GB VRAM) ---
# ==============================================================================
# 📐 NÂNG CẤP: Tăng độ phân giải từ 224x224 lên 380x380
# EfficientNet-B4 được thiết kế tối ưu cho 380x380
# Giúp model nhìn rõ các chi tiết nhỏ (artifacts) ở mép da, răng, mắt
IMAGE_SIZE = (380, 380)  # 🆙 Nâng cấp từ (224, 224) -> (380, 380)

NUM_EPOCHS = 10  # 🔄 Tăng từ 7 -> 10 vì model cần nhiều thời gian hơn với resolution cao
if DEVICE == "cuda":
    # ⚠️ GIẢM BATCH SIZE vì resolution cao hơn -> cần nhiều VRAM hơn
    BATCH_SIZE = 8  # Giảm từ 16 → 8 để tránh OOM với resolution 380x380
    NUM_WORKERS = 4  # 👷 Giảm từ 6 → 4 để cân bằng với batch size nhỏ hơn
    PIN_MEMORY = True  # ⚡ Tăng tốc CPU->GPU transfer
    MIXED_PRECISION = True  # 🎯 BẬT mixed precision để tăng tốc ~2x (QUAN TRỌNG với resolution cao)
    GRADIENT_CLIPPING = True  # ✂️ THÊM gradient clipping
    MAX_GRAD_NORM = 1.0      # 📏 Giới hạn gradient norm
    PREFETCH_FACTOR = 2  # 📦 Giảm từ 4 → 2 để tiết kiệm memory
else:
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PIN_MEMORY = False
    MIXED_PRECISION = False

# 🛡️ ĐIỀU CHỈNH ĐỂ TRÁNH OVERFITTING (dựa trên phân tích training log)
LEARNING_RATE = 0.0001  # 📉 Giảm từ 0.0005 → 0.0001 để học ổn định hơn, tránh overfitting
WEIGHT_DECAY = 1e-4  # 🔒 Tăng từ 1e-5 → 1e-4 để tăng regularization, giảm overfitting

# Gradient accumulation - KHÔNG CẦN vì batch size đã đủ lớn
ACCUMULATION_STEPS = 1  # ⏭️ Tắt accumulation để tăng tốc độ

# ==============================================================================
# --- 📊 CẤU HÌNH DATA AUGMENTATION & BALANCING (MỚI) ---
# ==============================================================================
# 🎨 Deepfake-specific Augmentation
USE_DEEPFAKE_AUGMENTATION = True  # Bật/tắt augmentation chuyên biệt cho Deepfake
ENABLE_COMPRESSION_AUG = True     # JPEG compression artifacts
ENABLE_NOISE_AUG = True            # Gaussian noise
ENABLE_BLUR_AUG = True             # Adaptive Gaussian blur
ENABLE_CUTOUT_AUG = True           # Face cutout

# ⚖️ Data Balancing (Oversampling)
USE_OVERSAMPLING = True            # Bật/tắt oversampling
OVERSAMPLING_METHOD = 'oversampling'  # 'oversampling' hoặc 'weighted_sampler'
OVERSAMPLE_RATIO = 1.3             # Tỷ lệ oversample cho lớp thiểu số (REAL)
                                   # 1.0 = cân bằng hoàn toàn
                                   # 1.3 = lớp REAL sẽ có 1.3x số mẫu của lớp FAKE
                                   # Giảm False Positive bằng cách cho model nhìn REAL nhiều hơn


# ==============================================================================
# --- 🌐 CẤU HÌNH ỨNG DỤNG WEB (APP CONFIGURATION) ---
# ==============================================================================
EVIDENCE_THRESHOLD = 0.80
CLASS_NAMES = ['FAKE', 'REAL']  # 🏷️ Thứ tự lớp (phải khớp với dataset)

# 📁 Giới hạn file upload
MAX_VIDEO_SIZE_MB = 500  # Giới hạn kích thước video (MB)
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}  # 🎬 Định dạng video cho phép

# ⚙️ Cấu hình xử lý video trong app
SKIP_FRAMES = 10  # ⏩ Bỏ qua N-1 frame, xử lý 1 frame (để tiết kiệm tài nguyên)
FACE_MARGIN = 20  # 📏 Margin (pixels) khi cắt khuôn mặt từ frame

# 👤 Cấu hình preprocessing
FACE_DETECTION_CONFIDENCE = 0.5  # 🎯 Ngưỡng confidence cho face detection
FACE_DETECTION_MODEL = 1  # 🤖 Model selection cho MediaPipe (0 hoặc 1)
