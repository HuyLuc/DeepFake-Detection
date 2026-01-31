# 🔍 DeepFake Detection - Kaggle Multi-GPU Training

Notebook hoàn chỉnh hỗ trợ **2 kiến trúc** với **Multi-GPU (T4 x2)**:

| | 🔵 Standard | 🟢 Advanced |
|---|---|---|
| **Model** | EfficientNet-B4 | EfficientNet + LSTM |
| **Batch size (2 GPU)** | 32-48 | 4-8 |
| **VRAM** | 30GB total | 30GB total |
| **Accuracy kỳ vọng** | ~98% | ~93-95% |

### 🚀 Chọn chế độ GPU phù hợp:

| Chế độ | Khi nào dùng | Tốc độ |
|--------|--------------|--------|
| **Single GPU** | Batch size nhỏ (≤16), debug | ⭐⭐⭐ Nhanh nhất với batch nhỏ |
| **DataParallel** | Batch size vừa (16-32) | ⭐⭐ Chậm hơn do sync overhead |
| **🔥 DDP** | Batch size lớn (≥32) | ⭐⭐⭐⭐ Nhanh nhất với batch lớn! |

---

## Cell 1: Kiểm tra GPU & Multi-GPU Setup

```python
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp

print("=" * 60)
print("🔍 KIỂM TRA MÔI TRƯỜNG")
print("=" * 60)

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")

# ============================================================
# 🚀 CHỌN CHẾ ĐỘ GPU - QUAN TRỌNG!
# ============================================================
# Chọn 1 trong 3 chế độ:
#   'single'       - Dùng 1 GPU (nhanh với batch nhỏ)
#   'dataparallel' - Dùng DataParallel (⭐ KHUYẾN NGHỊ cho Kaggle)
#   'ddp'          - Dùng DistributedDataParallel (phức tạp hơn)
# ============================================================
GPU_MODE = 'dataparallel'  # ⭐ Khuyến nghị cho Kaggle notebook
# ============================================================

# 💡 LƯU Ý:
# - 'dataparallel': Đơn giản, stable, nhanh hơn Single GPU ~40%
# - 'ddp': Nhanh nhất (~80%) NHƯNG cần setup multi-process (phức tạp trên Kaggle)
#   → Nếu chọn 'ddp' mà không setup, sẽ tự động fallback về 'dataparallel'


# Kiểm tra số GPU
NUM_GPUS = torch.cuda.device_count()
print(f"✅ Số GPU có sẵn: {NUM_GPUS}")

if torch.cuda.is_available():
    for i in range(NUM_GPUS):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("❌ Không có GPU!")
    GPU_MODE = 'single'

# Xác định cấu hình dựa trên GPU_MODE
if GPU_MODE == 'single' or NUM_GPUS == 1:
    USE_MULTI_GPU = False
    USE_DDP = False
    print(f"\n⚡ SINGLE GPU MODE")
elif GPU_MODE == 'dataparallel':
    USE_MULTI_GPU = True
    USE_DDP = False
    print(f"\n🚀 DATAPARALLEL MODE: {NUM_GPUS} GPUs")
    print("   ⚠️ Lưu ý: DataParallel có thể chậm hơn Single GPU với batch nhỏ")
elif GPU_MODE == 'ddp':
    USE_MULTI_GPU = True
    USE_DDP = True
    print(f"\n🔥 DDP MODE: {NUM_GPUS} GPUs (NHANH NHẤT!)")
    print("   ✅ DistributedDataParallel cho hiệu suất tối đa")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cài đặt packages
!pip install -q timm

print("\n✅ Cài đặt hoàn tất!")
print(f"📌 GPU Mode: {GPU_MODE.upper()}")
```

---

## Cell 2: Cấu hình đường dẫn Dataset

```python
import os

# ============================================================
# CẤU HÌNH DATASET
# ============================================================
DATASET_NAME = "train-df"  # Tên dataset của bạn
# ============================================================

base_input = f"/kaggle/input/{DATASET_NAME}"

if os.path.exists(os.path.join(base_input, "processed_data")):
    BASE_DATASET_DIR = os.path.join(base_input, "processed_data")
    print(f"📂 Phát hiện: {DATASET_NAME}/processed_data/")
else:
    BASE_DATASET_DIR = base_input
    print(f"📂 Sử dụng: {DATASET_NAME}/")

TRAIN_DIR = os.path.join(BASE_DATASET_DIR, "train")
VAL_DIR = os.path.join(BASE_DATASET_DIR, "val")
TEST_DIR = os.path.join(BASE_DATASET_DIR, "test")

# Output
BASE_OUTPUT_DIR = "/kaggle/working"
MODEL_SAVE_DIR_STANDARD = os.path.join(BASE_OUTPUT_DIR, "saved_models", "standard")
MODEL_SAVE_DIR_ADVANCED = os.path.join(BASE_OUTPUT_DIR, "saved_models", "advanced")
EVAL_RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "evaluation_results")

for d in [MODEL_SAVE_DIR_STANDARD, MODEL_SAVE_DIR_ADVANCED, EVAL_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"\n📂 Train: {TRAIN_DIR}")
print(f"📂 Val: {VAL_DIR}")
print(f"📂 Test: {TEST_DIR}")
```

---

## Cell 3: Kiểm tra dữ liệu

```python
import os
from glob import glob

print("=" * 60)
print("📊 KIỂM TRA DỮ LIỆU")
print("=" * 60)

for split, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    if os.path.exists(split_dir):
        fake_count = len(glob(os.path.join(split_dir, "FAKE", "**", "*.png"), recursive=True))
        real_count = len(glob(os.path.join(split_dir, "REAL", "**", "*.png"), recursive=True))
        print(f"\n📂 {split}: FAKE={fake_count}, REAL={real_count}, Total={fake_count+real_count}")
    else:
        print(f"\n❌ Không tìm thấy: {split_dir}")
```

---

## Cell 4: Dataset & Model Classes (Multi-GPU Ready) + 🔥 FULL AUGMENTATION

```python
# Suppress Pydantic warnings từ timm (không ảnh hưởng đến training)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from glob import glob
import os
import random
import numpy as np
import io
import timm

# 🚀 GPU Augmentation với Kornia (chạy trên GPU thay vì CPU)
try:
    import kornia.augmentation as K
    USE_KORNIA = True
    print("✅ Kornia available - GPU Augmentation enabled!")
except ImportError:
    USE_KORNIA = False
    print("⚠️ Kornia not found - using CPU Augmentation")

# ============================================================
# 🚀 FAST TRAINING MODE - TRAIN NHANH HƠN 10X!
# ============================================================
# ⚡ BẬT/TẮT Fast Training Mode:
#   True  = IMG 224, Minimal Aug → ~0.5s/it (NHANH!)
#   False = IMG 380, Full Aug    → ~5s/it (CHẬM nhưng robust)
# ============================================================
FAST_TRAINING = True  # 🔥 Đặt True để train nhanh!
# ============================================================

if FAST_TRAINING:
    IMAGE_SIZE = (224, 224)
    print("🚀 FAST TRAINING MODE: ON")
    print("   - Image size: 224x224")
    print("   - Minimal augmentation")
    print("   - Expected speed: ~0.5-1s/it (10x faster!)")
else:
    IMAGE_SIZE = (380, 380)
    print("🐢 FULL QUALITY MODE: ON")
    print("   - Image size: 380x380")
    print("   - Full augmentation")
    print("   - Expected speed: ~5s/it (slower but robust)")

# ⚡ BATCH SIZE TỐI ƯU (sẽ được điều chỉnh trong Cell 5/7)
BATCH_SIZE = 16          # Base batch size cho Single GPU
BATCH_SIZE_ADV = 2       # Temporal model cần nhiều VRAM hơn

# ============================================================
# 🔥 TỐI ƯU DATALOADER CHO MULTI-GPU
# ============================================================
import multiprocessing as mp

# Tính toán num_workers tự động dựa trên số GPU
NUM_CPUS = 2  # Kaggle có 2 CPU cores
NUM_GPUS_ACTIVE = NUM_GPUS if USE_MULTI_GPU else 1

# 🔥 CÔNG THỨC TỐI ƯU:
# - Single GPU: num_workers = 2
# - Multi-GPU: num_workers = min(4, NUM_CPUS * NUM_GPUS)
if USE_MULTI_GPU:
    NUM_WORKERS = min(4, NUM_CPUS * NUM_GPUS_ACTIVE)  # 4 workers cho 2 GPU
    PREFETCH_FACTOR = 4  # Prefetch nhiều hơn để GPU không bị đói data
else:
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2

PERSISTENT_WORKERS = True  # Giữ workers alive, giảm overhead

print(f"📊 GPU Configuration:")
print(f"   Active GPUs: {NUM_GPUS_ACTIVE}")
print(f"   Batch Size Standard: {BATCH_SIZE}")
print(f"   Batch Size Advanced: {BATCH_SIZE_ADV}")
print(f"\n🔥 DataLoader Optimization:")
print(f"   NUM_WORKERS: {NUM_WORKERS} {'(scaled for multi-GPU!)' if USE_MULTI_GPU else ''}")
print(f"   PREFETCH_FACTOR: {PREFETCH_FACTOR}")
print(f"   PERSISTENT_WORKERS: {PERSISTENT_WORKERS}")

if USE_MULTI_GPU:
    print(f"\n💡 Tip: Multi-GPU cần nhiều workers hơn để feed data đủ nhanh!")

# ============================================================
# 🔥 DEEPFAKE-SPECIFIC AUGMENTATIONS (QUAN TRỌNG!)
# ============================================================
# Các augmentation chuyên biệt cho Deepfake Detection
# Giúp giảm overfitting bằng cách mô phỏng các artifacts thực tế

class JPEGCompression:
    """
    Mô phỏng compression artifacts từ việc nén JPEG/Video.
    Deepfake thường bị mờ đi và xuất hiện artifacts khi nén.
    """
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=quality)
        output_buffer.seek(0)
        compressed_img = Image.open(output_buffer).convert('RGB')
        return compressed_img


class AdaptiveGaussianNoise:
    """
    Thêm Gaussian noise với cường độ thích ứng.
    Mô phỏng nhiễu từ camera chất lượng thấp.
    """
    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img).astype(np.float32) / 255.0
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(0, std, img_array.shape).astype(np.float32)
        noisy_img = np.clip(img_array + noise, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class AdaptiveGaussianBlur:
    """
    Blur thích ứng để mô phỏng sự mất nét trong video Deepfake.
    """
    def __init__(self, sigma_range=(0.1, 2.0), p=0.2):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return blurred_img


class FaceCutout:
    """
    Random cutout một phần nhỏ trên khuôn mặt.
    Giúp model học cách tập trung vào nhiều đặc điểm khác nhau.
    """
    def __init__(self, num_holes=1, max_h_size=0.15, max_w_size=0.15, p=0.3):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        h, w, c = img_array.shape
        
        for _ in range(self.num_holes):
            cutout_h = int(h * random.uniform(0.05, self.max_h_size))
            cutout_w = int(w * random.uniform(0.05, self.max_w_size))
            y = random.randint(0, h - cutout_h)
            x = random.randint(0, w - cutout_w)
            mean_color = img_array[y:y+cutout_h, x:x+cutout_w].mean(axis=(0, 1))
            img_array[y:y+cutout_h, x:x+cutout_w] = mean_color
        
        return Image.fromarray(img_array)


class MixedDeepfakeAugmentation:
    """
    Kết hợp nhiều phép augmentation chuyên biệt cho Deepfake.
    """
    def __init__(self, enable_compression=True, enable_noise=True, 
                 enable_blur=True, enable_cutout=True):
        self.augmentations = []
        
        if enable_compression:
            self.augmentations.append(JPEGCompression(quality_range=(30, 95), p=0.5))
        if enable_noise:
            self.augmentations.append(AdaptiveGaussianNoise(std_range=(0.01, 0.05), p=0.3))
        if enable_blur:
            self.augmentations.append(AdaptiveGaussianBlur(sigma_range=(0.1, 1.5), p=0.2))
        if enable_cutout:
            self.augmentations.append(FaceCutout(num_holes=1, max_h_size=0.15, max_w_size=0.15, p=0.3))
        
        print(f"🔥 MixedDeepfakeAugmentation: {len(self.augmentations)} augmentations enabled")
    
    def __call__(self, img):
        for aug in self.augmentations:
            img = aug(img)
        return img


# ============================================================
# TRANSFORMS - 🔥 VỚI FULL AUGMENTATION
# ============================================================
def get_base_transforms():
    """Transforms cơ bản - chỉ resize và normalize (cho validation/test)"""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_train_transforms():
    """
    🔥 Training transforms với 2 modes:
    - FAST_TRAINING=True:  Minimal aug → Train nhanh 10x
    - FAST_TRAINING=False: Full aug    → Robust nhưng chậm
    """
    if FAST_TRAINING:
        # 🚀 FAST MODE: Minimal augmentation
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        # 🐢 FULL MODE: Full augmentation (chậm hơn)
        return transforms.Compose([
            # 1. Resize
            transforms.Resize(IMAGE_SIZE),
            
            # 2. Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            
            # 3. 🔥 Deepfake-specific augmentations
            MixedDeepfakeAugmentation(
                enable_compression=True,
                enable_noise=True,
                enable_blur=True,
                enable_cutout=True
            ),
            
            # 4. Color augmentations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            
            # 5. Convert to Tensor
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            
            # 6. Random Erasing
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.1)),
        ])

def get_val_transforms():
    """Transforms cho validation - KHÔNG có augmentation"""
    return get_base_transforms()

if FAST_TRAINING:
    print("✅ Transforms: FAST MODE (Minimal augmentation)")
else:
    print("✅ Transforms: FULL MODE (Full augmentation)")

# ============================================================
# 🚀 GPU AUGMENTATION VỚI KORNIA (TÙY CHỌN)
# ============================================================
# ⚠️ Có thể TẮT vì đã có CPU augmentation đầy đủ
ENABLE_GPU_AUG = False

class GPUAugmentation(nn.Module):
    """Augmentation chạy trên GPU (optional, vì đã có CPU aug)"""
    def __init__(self):
        super().__init__()
        if USE_KORNIA and ENABLE_GPU_AUG:
            self.aug = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=10, p=0.3),
            )
            print("🚀 GPU Augmentation: ENABLED")
        else:
            self.aug = None
            print("⚡ GPU Augmentation: DISABLED (using CPU augmentation)")
    
    def forward(self, x):
        if self.aug is not None and self.training:
            return self.aug(x)
        return x

gpu_aug = GPUAugmentation() if (USE_KORNIA and ENABLE_GPU_AUG) else None

# ============================================================
# DATASET CLASS
# ============================================================
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            for video_dir in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_dir)
                if os.path.isdir(video_path):
                    for img_file in glob(os.path.join(video_path, "*.png")):
                        self.image_paths.append(img_file)
                        self.labels.append(class_idx)
        
        print(f"📂 Loaded {len(self.image_paths)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return self.__getitem__((idx + 1) % len(self))

# ============================================================
# TEMPORAL DATASET
# ============================================================
class TemporalDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=10):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.videos = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            for video_dir in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_dir)
                if os.path.isdir(video_path):
                    frames = sorted(glob(os.path.join(video_path, "*.png")))
                    if len(frames) >= sequence_length:
                        self.videos.append({'frames': frames, 'label': class_idx})
        
        print(f"📂 Loaded {len(self.videos)} videos from {data_dir}")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = video['frames']
        
        indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
        
        images = []
        for i in indices:
            try:
                img = Image.open(frames[i]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except:
                images.append(torch.zeros(3, *IMAGE_SIZE))
        
        return torch.stack(images), video['label']

# ============================================================
# MODELS
# ============================================================
def create_standard_model(num_classes=2, pretrained=True):
    model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
    print(f"✅ Created Standard Model: EfficientNet-B4")
    return model

class TemporalModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features
        
        self.lstm = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print("✅ Created Advanced Model: EfficientNet + LSTM")
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out)

# ============================================================
# 🚀 MULTI-GPU WRAPPER - TỰ ĐỘNG FALLBACK
# ============================================================
from torch.nn.parallel import DistributedDataParallel as DDP

# Global variables cho DDP
LOCAL_RANK = 0
WORLD_SIZE = NUM_GPUS

# ⚠️ KIỂM TRA DDP AVAILABILITY
DDP_AVAILABLE = False
if USE_DDP:
    try:
        # Kiểm tra xem distributed đã được init chưa
        if dist.is_available() and dist.is_initialized():
            DDP_AVAILABLE = True
            print("✅ DDP đã được initialize!")
        else:
            print("⚠️ DDP chưa được initialize, sẽ dùng DataParallel thay thế")
            print("   Để dùng DDP, cần chạy với torch.distributed.launch")
            USE_DDP = False
            USE_MULTI_GPU = True  # Fallback to DataParallel
    except:
        print("⚠️ DDP không khả dụng, dùng DataParallel")
        USE_DDP = False
        USE_MULTI_GPU = True

def wrap_model_multi_gpu(model, rank=0):
    """
    Wrap model với DataParallel hoặc DDP (auto-fallback)
    
    Args:
        model: PyTorch model
        rank: GPU rank (chỉ dùng cho DDP)
    Returns:
        Wrapped model
    """
    global USE_DDP  # Có thể thay đổi nếu fallback
    
    if not USE_MULTI_GPU:
        # Single GPU mode
        model = model.to(device)
        print(f"⚡ Model on Single GPU")
    elif USE_DDP and DDP_AVAILABLE:
        # DDP mode - kiểm tra lại một lần nữa
        try:
            model = model.to(f'cuda:{rank}')
            model = DDP(model, device_ids=[rank], output_device=rank)
            print(f"🔥 Model wrapped with DDP (GPU {rank})")
        except Exception as e:
            print(f"⚠️ DDP failed: {e}")
            print(f"   → Fallback to DataParallel")
            USE_DDP = False
            model = nn.DataParallel(model)
            model = model.to(device)
            print(f"🚀 Model wrapped with DataParallel ({NUM_GPUS} GPUs)")
    else:
        # DataParallel mode
        model = nn.DataParallel(model)
        model = model.to(device)
        print(f"🚀 Model wrapped with DataParallel ({NUM_GPUS} GPUs)")
    
    return model

def get_ddp_sampler(dataset, shuffle=True):
    """Tạo DistributedSampler cho DDP (nếu available)"""
    if USE_DDP and DDP_AVAILABLE:
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=shuffle)
    return None

print("\n✅ Dataset và Model classes đã sẵn sàng!")
print(f"📌 Chế độ: {'DDP' if (USE_DDP and DDP_AVAILABLE) else 'DataParallel' if USE_MULTI_GPU else 'Single GPU'}")

# 💡 LƯU Ý: Trong Kaggle notebook, DDP khó setup
# Khuyến nghị: Dùng DataParallel với batch size lớn thay thế
if GPU_MODE == 'ddp' and not DDP_AVAILABLE:
    print("\n" + "="*60)
    print("💡 LƯU Ý: DDP YÊU CẦU MULTI-PROCESS")
    print("="*60)
    print("Trong Kaggle notebook, dùng DataParallel đơn giản hơn:")
    print("  - Đặt GPU_MODE = 'dataparallel' trong Cell 1")
    print("  - Vẫn nhanh hơn Single GPU ~40%")
    print("  - Batch size tự động tăng lên 32-48")
    print("="*60)
```

---

## Cell 5: 🔵 STANDARD TRAINING (DDP/DataParallel/Single GPU)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import csv
import os

print("=" * 60)
print("🔵 STANDARD TRAINING - EfficientNet-B4")
print(f"🎯 Device: {device}")
print(f"� GPU Mode: {'DDP' if USE_DDP else 'DataParallel' if USE_MULTI_GPU else 'Single GPU'}")
print("=" * 60)

# ============================================================
# CẤU HÌNH - 🔥 TỐI ƯU CHO MULTI-GPU
# ============================================================
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# 🔥 BATCH SIZE TỐI ƯU CHO MULTI-GPU
# - Single GPU: 16
# - DataParallel (2 GPU): 32 (DataParallel tự chia batch)
if USE_MULTI_GPU:
    BATCH_SIZE_EFFECTIVE = 32  # DataParallel tự chia 32 cho 2 GPU
    print(f"🚀 DataParallel: Batch size = {BATCH_SIZE_EFFECTIVE} ({BATCH_SIZE_EFFECTIVE//NUM_GPUS} per GPU)")
else:
    BATCH_SIZE_EFFECTIVE = 16  # Single GPU
    print(f"⚡ Single GPU: Batch size = {BATCH_SIZE_EFFECTIVE}")
# ============================================================

# Datasets
train_dataset = DeepfakeDataset(TRAIN_DIR, transform=get_train_transforms())
val_dataset = DeepfakeDataset(VAL_DIR, transform=get_val_transforms())

# DataLoader - DataParallel KHÔNG cần DistributedSampler
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE_EFFECTIVE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE_EFFECTIVE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS
)

# Model với Multi-GPU/DDP
model = create_standard_model(num_classes=2, pretrained=True)
model = wrap_model_multi_gpu(model, rank=LOCAL_RANK)

# Class weights
num_fake = sum(1 for l in train_dataset.labels if l == 0)
num_real = sum(1 for l in train_dataset.labels if l == 1)
total = num_fake + num_real
class_weights = torch.tensor([total/(2*num_fake), total/(2*num_real)], device=device)
print(f"📊 Class weights: FAKE={class_weights[0]:.2f}, REAL={class_weights[1]:.2f}")

# Loss, optimizer, scaler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler('cuda')

# ============================================================
# 💾 CHECKPOINT SYSTEM - KHÔNG BỊ MẤT DỮ LIỆU!
# ============================================================
CHECKPOINT_DIR = os.path.join(MODEL_SAVE_DIR_STANDARD, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_FROM_CHECKPOINT = True  # ✅ Bật để tiếp tục training
SAVE_CHECKPOINT_EVERY = 1      # Lưu mỗi N epochs

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
best_model_path = os.path.join(MODEL_SAVE_DIR_STANDARD, 'best_model.pth')

# Log
log_path = os.path.join(EVAL_RESULTS_DIR, 'standard_training_log.csv')

# Variables để resume
start_epoch = 0
best_val_acc = 0.0
training_history = []

# 🔄 RESUME FROM CHECKPOINT (nếu có)
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
    print(f"\n{'='*60}")
    print("🔄 RESUMING FROM CHECKPOINT...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer & scheduler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load training state
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    training_history = checkpoint.get('history', [])
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
    print(f"✅ Best Val Acc so far: {best_val_acc:.4f}")
    print(f"✅ Will continue from epoch {start_epoch + 1}")
    print(f"{'='*60}\n")
else:
    print(f"\n🆕 Starting fresh training (no checkpoint found)")
    # Tạo log file mới
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
# ============================================================

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n{'='*40}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*40}")
    
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    # GPU augmentation mode
    if gpu_aug is not None:
        gpu_aug.train()
        gpu_aug.to(device)
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 🚀 GPU Augmentation (nếu có Kornia)
        if gpu_aug is not None:
            images = gpu_aug(images)
        
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += preds.eq(labels).sum().item()
        train_loss += loss.item() * labels.size(0)
        
        pbar.set_postfix({'Loss': f'{train_loss/train_total:.4f}', 'Acc': f'{train_correct/train_total:.4f}'})
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()
            val_loss += loss.item() * labels.size(0)
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    scheduler.step()
    
    print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch+1, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_loss:.4f}', f'{val_acc:.4f}'])
    
    # 💾 SAVE BEST MODEL
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Lưu state_dict của model gốc (không có DDP/DataParallel wrapper)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), best_model_path)
        print(f"🎉 Saved best model! Val Acc: {val_acc:.4f}")
    
    # 💾 SAVE CHECKPOINT (mỗi N epochs)
    if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': training_history + [{
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }]
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint at epoch {epoch + 1}")

print("\n" + "=" * 60)
print("✅ STANDARD TRAINING HOÀN TẤT!")
print(f"📊 Best Val Accuracy: {best_val_acc:.4f}")
print("=" * 60)
```

---

## Cell 6: Giải phóng GPU Memory (Chạy trước Advanced)

```python
# ============================================================
# GIẢI PHÓNG GPU MEMORY TRƯỚC KHI TRAIN ADVANCED
# ============================================================
import gc

# Xóa model cũ
if 'model' in dir():
    del model
if 'train_loader' in dir():
    del train_loader
if 'val_loader' in dir():
    del val_loader

gc.collect()
torch.cuda.empty_cache()

print(f"✅ Đã giải phóng GPU memory!")
for i in range(NUM_GPUS):
    mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"   GPU {i}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
```

---

## Cell 7: 🟢 ADVANCED TRAINING (Multi-GPU) + 🔥 ANTI-OVERFITTING

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import csv
import os

print("=" * 60)
print("🟢 ADVANCED TRAINING - Temporal Model + ANTI-OVERFITTING")
print(f"🎯 Device: {device}")
print(f"🚀 Multi-GPU: {USE_MULTI_GPU} ({NUM_GPUS} GPUs)")
print("=" * 60)

# ============================================================
# 🔥 CẤU HÌNH MỚI - GIẢM OVERFITTING + TỐI ƯU MULTI-GPU
# ============================================================
NUM_EPOCHS = 20                # Tăng epochs nhưng có Early Stopping
LEARNING_RATE = 0.00005        # 🔥 Giảm LR từ 0.0001 -> 0.00005
WEIGHT_DECAY = 1e-3            # 🔥 Tăng từ 1e-4 -> 1e-3

# 🔥 SEQUENCE_LENGTH TỐI ƯU CHO MULTI-GPU
# Giảm từ 10 -> 5 để:
# - Giảm CPU bottleneck (load ít frames hơn)
# - GPU utilization tăng (ít thời gian chờ data)
# - Training nhanh hơn ~2x
SEQUENCE_LENGTH = 5  # 🔥 Giảm từ 10 xuống 5 frames

# 🔥 Early Stopping config
EARLY_STOP_PATIENCE = 5        # Dừng nếu val_loss không giảm trong 5 epochs
MIN_DELTA = 0.001              # Cải thiện tối thiểu để reset patience
# ============================================================

print(f"🔧 Learning Rate: {LEARNING_RATE}")
print(f"🔧 Weight Decay: {WEIGHT_DECAY}")
print(f"🔥 Sequence Length: {SEQUENCE_LENGTH} frames (tối ưu cho multi-GPU)")
print(f"🔧 Early Stopping Patience: {EARLY_STOP_PATIENCE}")

# Datasets - SỬ DỤNG FULL AUGMENTATION
train_dataset_adv = TemporalDataset(TRAIN_DIR, transform=get_train_transforms(), sequence_length=SEQUENCE_LENGTH)
val_dataset_adv = TemporalDataset(VAL_DIR, transform=get_val_transforms(), sequence_length=SEQUENCE_LENGTH)

# ⚡ DataLoader tối ưu
train_loader_adv = DataLoader(
    train_dataset_adv, batch_size=BATCH_SIZE_ADV, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS
)
val_loader_adv = DataLoader(
    val_dataset_adv, batch_size=BATCH_SIZE_ADV, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS
)

# 🔥 Model với TĂNG DROPOUT
class TemporalModelV2(nn.Module):
    """Temporal Model với TĂNG DROPOUT để giảm overfitting"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features
        
        self.lstm = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5  # 🔥 Tăng từ 0.3 -> 0.5
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),           # 🔥 Tăng từ 0.5 -> 0.6
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),           # 🔥 Tăng từ 0.3 -> 0.5
            nn.Linear(256, num_classes)
        )
        
        print("✅ Created Advanced Model V2: EfficientNet + LSTM (TĂNG DROPOUT)")
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out)

# Tạo model mới với tăng dropout
model_adv = TemporalModelV2(num_classes=2, pretrained=True)
model_adv = wrap_model_multi_gpu(model_adv)

# Class weights
labels = [v['label'] for v in train_dataset_adv.videos]
num_fake = labels.count(0)
num_real = labels.count(1)
total = num_fake + num_real
class_weights_adv = torch.tensor([total/(2*num_fake), total/(2*num_real)], device=device)
print(f"📊 Class weights: FAKE={class_weights_adv[0]:.2f}, REAL={class_weights_adv[1]:.2f}")

# 🔥 Loss, optimizer với TĂNG REGULARIZATION
criterion_adv = nn.CrossEntropyLoss(weight=class_weights_adv, label_smoothing=0.1)  # 🔥 Label smoothing
optimizer_adv = optim.AdamW(model_adv.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler_adv = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adv, mode='min', patience=3, factor=0.5)  # 🔥 Thay CosineAnnealing
scaler_adv = GradScaler('cuda')

# ============================================================
# 💾 CHECKPOINT SYSTEM - ADVANCED MODEL
# ============================================================
CHECKPOINT_DIR_ADV = os.path.join(MODEL_SAVE_DIR_ADVANCED, 'checkpoints')
os.makedirs(CHECKPOINT_DIR_ADV, exist_ok=True)

RESUME_FROM_CHECKPOINT_ADV = True  # ✅ Bật để tiếp tục training
SAVE_CHECKPOINT_EVERY_ADV = 1      # Lưu mỗi N epochs

checkpoint_path_adv = os.path.join(CHECKPOINT_DIR_ADV, 'latest_checkpoint.pth')
best_model_path_adv = os.path.join(MODEL_SAVE_DIR_ADVANCED, 'best_temporal_model.pth')

# Log
log_path_adv = os.path.join(EVAL_RESULTS_DIR, 'advanced_training_log.csv')

# Variables để resume
start_epoch_adv = 0
best_val_acc_adv = 0.0
best_val_loss = float('inf')
early_stop_counter = 0
epochs_trained = 0
training_history_adv = []

# 🔄 RESUME FROM CHECKPOINT (nếu có)
if RESUME_FROM_CHECKPOINT_ADV and os.path.exists(checkpoint_path_adv):
    print(f"\n{'='*60}")
    print("🔄 RESUMING ADVANCED TRAINING FROM CHECKPOINT...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path_adv, map_location=device)
    
    # Load model state
    model_to_load = model_adv.module if hasattr(model_adv, 'module') else model_adv
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer & scheduler
    optimizer_adv.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_adv.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler_adv.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load training state
    start_epoch_adv = checkpoint['epoch'] + 1
    best_val_acc_adv = checkpoint['best_val_acc']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stop_counter = checkpoint.get('early_stop_counter', 0)
    training_history_adv = checkpoint.get('history', [])
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
    print(f"✅ Best Val Acc: {best_val_acc_adv:.4f}")
    print(f"✅ Best Val Loss: {best_val_loss:.4f}")
    print(f"✅ Early Stop Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
    print(f"✅ Will continue from epoch {start_epoch_adv + 1}")
    print(f"{'='*60}\n")
else:
    print(f"\n🆕 Starting fresh advanced training (no checkpoint found)")
    # Tạo log file mới
    with open(log_path_adv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
# ============================================================

for epoch in range(start_epoch_adv, NUM_EPOCHS):
    print(f"\n{'='*40}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Early Stop Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
    print(f"{'='*40}")
    
    # Train
    model_adv.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    pbar = tqdm(train_loader_adv, desc="Training")
    for sequences, labels in pbar:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer_adv.zero_grad()
        with autocast('cuda'):
            outputs = model_adv(sequences)
            loss = criterion_adv(outputs, labels)
        
        scaler_adv.scale(loss).backward()
        scaler_adv.unscale_(optimizer_adv)
        torch.nn.utils.clip_grad_norm_(model_adv.parameters(), max_norm=1.0)
        scaler_adv.step(optimizer_adv)
        scaler_adv.update()
        
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += preds.eq(labels).sum().item()
        train_loss += loss.item() * labels.size(0)
        
        pbar.set_postfix({'Loss': f'{train_loss/train_total:.4f}', 'Acc': f'{train_correct/train_total:.4f}'})
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    
    # Validation
    model_adv.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader_adv, desc="Validation"):
            sequences, labels = sequences.to(device), labels.to(device)
            with autocast('cuda'):
                outputs = model_adv(sequences)
                loss = criterion_adv(outputs, labels)
            
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()
            val_loss += loss.item() * labels.size(0)
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    # 🔥 Scheduler dựa trên val_loss
    scheduler_adv.step(val_loss)
    current_lr = optimizer_adv.param_groups[0]['lr']
    
    print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"📈 Learning Rate: {current_lr:.6f}")
    
    with open(log_path_adv, 'a', newline='') as f:
        csv.writer(f).writerow([epoch+1, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_loss:.4f}', f'{val_acc:.4f}'])
    
    # 💾 SAVE BEST MODEL
    if val_acc > best_val_acc_adv:
        best_val_acc_adv = val_acc
        model_to_save = model_adv.module if hasattr(model_adv, 'module') else model_adv
        torch.save(model_to_save.state_dict(), best_model_path_adv)
        print(f"🎉 Saved best model! Val Acc: {val_acc:.4f}")
    
    # 💾 SAVE CHECKPOINT (mỗi N epochs)
    if (epoch + 1) % SAVE_CHECKPOINT_EVERY_ADV == 0:
        model_to_save = model_adv.module if hasattr(model_adv, 'module') else model_adv
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer_adv.state_dict(),
            'scheduler_state_dict': scheduler_adv.state_dict(),
            'scaler_state_dict': scaler_adv.state_dict(),
            'best_val_acc': best_val_acc_adv,
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': training_history_adv + [{
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }]
        }
        torch.save(checkpoint, checkpoint_path_adv)
        print(f"💾 Saved checkpoint at epoch {epoch + 1}")
    
    # 🔥 Early Stopping check based on Val Loss
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⚠️ Val loss không cải thiện ({val_loss:.4f} >= {best_val_loss:.4f})")
        
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 EARLY STOPPING tại epoch {epoch+1}!")
            print(f"   Val loss không cải thiện trong {EARLY_STOP_PATIENCE} epochs liên tiếp.")
            epochs_trained = epoch + 1
            break
    
    epochs_trained = epoch + 1

print("\n" + "=" * 60)
print("✅ ADVANCED TRAINING HOÀN TẤT!")
print(f"📊 Epochs trained: {epochs_trained}/{NUM_EPOCHS}")
print(f"📊 Best Val Accuracy: {best_val_acc_adv:.4f}")
print(f"📊 Best Val Loss: {best_val_loss:.4f}")
print("=" * 60)
```

---

## Cell 8: Đánh giá Models

```python
import os
import json
from tqdm import tqdm

print("=" * 60)
print("📊 ĐÁNH GIÁ MODELS")
print("=" * 60)

results = {}

# Test Standard
if os.path.exists(os.path.join(MODEL_SAVE_DIR_STANDARD, 'best_model.pth')):
    print("\n🔵 Testing STANDARD model...")
    
    model_std = create_standard_model(num_classes=2, pretrained=False)
    model_std.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR_STANDARD, 'best_model.pth')))
    model_std = model_std.to(device)
    model_std.eval()
    
    test_dataset = DeepfakeDataset(TEST_DIR, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Standard"):
            images, labels = images.to(device), labels.to(device)
            outputs = model_std(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    
    results['standard'] = correct / total
    print(f"✅ Standard Test Accuracy: {results['standard']:.4f} ({results['standard']*100:.2f}%)")
    del model_std
    torch.cuda.empty_cache()

# Test Advanced
if os.path.exists(os.path.join(MODEL_SAVE_DIR_ADVANCED, 'best_temporal_model.pth')):
    print("\n🟢 Testing ADVANCED model...")
    
    model_adv_test = TemporalModel(num_classes=2, pretrained=False)
    model_adv_test.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR_ADVANCED, 'best_temporal_model.pth')))
    model_adv_test = model_adv_test.to(device)
    model_adv_test.eval()
    
    test_dataset_adv = TemporalDataset(TEST_DIR, transform=get_val_transforms(), sequence_length=SEQUENCE_LENGTH)
    test_loader_adv = DataLoader(test_dataset_adv, batch_size=BATCH_SIZE_ADV, shuffle=False, num_workers=NUM_WORKERS)
    
    correct, total = 0, 0
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader_adv, desc="Testing Advanced"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model_adv_test(sequences)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    
    results['advanced'] = correct / total
    print(f"✅ Advanced Test Accuracy: {results['advanced']:.4f} ({results['advanced']*100:.2f}%)")

# Summary
print("\n" + "=" * 60)
print("📊 KẾT QUẢ TỔNG HỢP")
print("=" * 60)
print("\n| Model | Test Accuracy |")
print("|-------|---------------|")
for name, acc in results.items():
    icon = "🔵" if name == "standard" else "🟢"
    print(f"| {icon} {name.capitalize()} | {acc:.4f} ({acc*100:.2f}%) |")

if len(results) == 2:
    diff = results['advanced'] - results['standard']
    print(f"\n📈 Advanced tốt hơn Standard: {diff*100:+.2f}%")

with open(os.path.join(EVAL_RESULTS_DIR, 'final_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("\n✅ HOÀN TẤT! Download models từ tab Output")
```

---

## Cell 9: 📊 Visualization - Biểu đồ & Bảng so sánh

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json

print("=" * 60)
print("📊 VISUALIZATION - BIỂU ĐỒ & BẢNG SO SÁNH")
print("=" * 60)

# Thiết lập style
plt.style.use('dark_background')
sns.set_palette("husl")

# ============================================================
# 1. ĐỌC TRAINING LOGS
# ============================================================
def load_training_log(log_path):
    """Load training log từ CSV"""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        return df
    return None

standard_log = load_training_log(os.path.join(EVAL_RESULTS_DIR, 'standard_training_log.csv'))
advanced_log = load_training_log(os.path.join(EVAL_RESULTS_DIR, 'advanced_training_log.csv'))

# ============================================================
# 2. BIỂU ĐỒ TRAINING CURVES
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('📊 Training & Validation Metrics', fontsize=16, fontweight='bold')

# Plot 1: Standard Model - Loss
if standard_log is not None:
    ax1 = axes[0, 0]
    ax1.plot(standard_log['epoch'], standard_log['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(standard_log['epoch'], standard_log['val_loss'], 'r--', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('🔵 Standard Model - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Plot 2: Standard Model - Accuracy
if standard_log is not None:
    ax2 = axes[0, 1]
    ax2.plot(standard_log['epoch'], standard_log['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(standard_log['epoch'], standard_log['val_acc'], 'r--', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('🔵 Standard Model - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

# Plot 3: Advanced Model - Loss
if advanced_log is not None:
    ax3 = axes[1, 0]
    ax3.plot(advanced_log['epoch'], advanced_log['train_loss'], 'g-', label='Train Loss', linewidth=2)
    ax3.plot(advanced_log['epoch'], advanced_log['val_loss'], 'orange', linestyle='--', label='Val Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('🟢 Advanced Model - Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: Advanced Model - Accuracy
if advanced_log is not None:
    ax4 = axes[1, 1]
    ax4.plot(advanced_log['epoch'], advanced_log['train_acc'], 'g-', label='Train Acc', linewidth=2)
    ax4.plot(advanced_log['epoch'], advanced_log['val_acc'], 'orange', linestyle='--', label='Val Acc', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('🟢 Advanced Model - Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: training_curves.png")

# ============================================================
# 3. SO SÁNH 2 MÔ HÌNH - VALIDATION ACCURACY
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

if standard_log is not None:
    ax.plot(standard_log['epoch'], standard_log['val_acc'], 'b-o', label='🔵 Standard (EfficientNet-B4)', linewidth=2, markersize=6)
if advanced_log is not None:
    ax.plot(advanced_log['epoch'], advanced_log['val_acc'], 'g-s', label='🟢 Advanced (EfficientNet + LSTM)', linewidth=2, markersize=6)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('📈 So sánh Validation Accuracy - Standard vs Advanced', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.0])

# Thêm best accuracy annotation
if standard_log is not None:
    best_std = standard_log['val_acc'].max()
    best_std_epoch = standard_log['val_acc'].idxmax() + 1
    ax.annotate(f'Best: {best_std:.4f}', xy=(best_std_epoch, best_std), 
                xytext=(best_std_epoch+0.5, best_std+0.02), fontsize=10, color='blue')
if advanced_log is not None:
    best_adv = advanced_log['val_acc'].max()
    best_adv_epoch = advanced_log['val_acc'].idxmax() + 1
    ax.annotate(f'Best: {best_adv:.4f}', xy=(best_adv_epoch, best_adv), 
                xytext=(best_adv_epoch+0.5, best_adv+0.02), fontsize=10, color='green')

plt.tight_layout()
plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: model_comparison.png")
```

---

## Cell 10: 📊 Confusion Matrix & Classification Report

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("=" * 60)
print("📊 CONFUSION MATRIX & CLASSIFICATION REPORT")
print("=" * 60)

def evaluate_model_detailed(model, data_loader, model_name, device):
    """Đánh giá chi tiết model và tạo confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Tính metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 {model_name} Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   True Positives:  {tp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Đánh giá Standard Model
all_metrics = {}

if os.path.exists(os.path.join(MODEL_SAVE_DIR_STANDARD, 'best_model.pth')):
    print("\n🔵 Evaluating STANDARD model...")
    model_std = create_standard_model(num_classes=2, pretrained=False)
    model_std.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR_STANDARD, 'best_model.pth')))
    model_std = model_std.to(device)
    model_std.eval()
    
    test_dataset = DeepfakeDataset(TEST_DIR, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    preds_std, labels_std, probs_std = evaluate_model_detailed(model_std, test_loader, "Standard", device)
    metrics_std = plot_confusion_matrix(labels_std, preds_std, "🔵 Standard (EfficientNet-B4)", 
                                        os.path.join(EVAL_RESULTS_DIR, 'confusion_matrix_standard.png'))
    all_metrics['standard'] = metrics_std
    
    print("\n📋 Classification Report - Standard:")
    print(classification_report(labels_std, preds_std, target_names=['FAKE', 'REAL']))
    
    del model_std
    torch.cuda.empty_cache()

# Đánh giá Advanced Model
if os.path.exists(os.path.join(MODEL_SAVE_DIR_ADVANCED, 'best_temporal_model.pth')):
    print("\n🟢 Evaluating ADVANCED model...")
    model_adv = TemporalModel(num_classes=2, pretrained=False)
    model_adv.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR_ADVANCED, 'best_temporal_model.pth')))
    model_adv = model_adv.to(device)
    model_adv.eval()
    
    test_dataset_adv = TemporalDataset(TEST_DIR, transform=get_val_transforms(), sequence_length=SEQUENCE_LENGTH)
    test_loader_adv = DataLoader(test_dataset_adv, batch_size=BATCH_SIZE_ADV, shuffle=False, num_workers=NUM_WORKERS)
    
    preds_adv, labels_adv, probs_adv = evaluate_model_detailed(model_adv, test_loader_adv, "Advanced", device)
    metrics_adv = plot_confusion_matrix(labels_adv, preds_adv, "🟢 Advanced (EfficientNet + LSTM)", 
                                        os.path.join(EVAL_RESULTS_DIR, 'confusion_matrix_advanced.png'))
    all_metrics['advanced'] = metrics_adv
    
    print("\n📋 Classification Report - Advanced:")
    print(classification_report(labels_adv, preds_adv, target_names=['FAKE', 'REAL']))

print("\n✅ Confusion matrices saved!")
```

---

## Cell 11: 📊 Bảng so sánh tổng hợp & ROC Curve

```python
print("=" * 60)
print("📊 BẢNG SO SÁNH TỔNG HỢP & ROC CURVE")
print("=" * 60)

# ============================================================
# BẢNG SO SÁNH
# ============================================================
if len(all_metrics) > 0:
    print("\n" + "=" * 70)
    print("📊 BẢNG SO SÁNH 2 MÔ HÌNH")
    print("=" * 70)
    print(f"{'Metric':<15} | {'🔵 Standard':<15} | {'🟢 Advanced':<15} | {'Difference':<15}")
    print("-" * 70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        std_val = all_metrics.get('standard', {}).get(metric, 0)
        adv_val = all_metrics.get('advanced', {}).get(metric, 0)
        diff = adv_val - std_val
        diff_str = f"{diff*100:+.2f}%" if diff != 0 else "0%"
        print(f"{metric.capitalize():<15} | {std_val:.4f}         | {adv_val:.4f}         | {diff_str}")
    
    print("=" * 70)

# ============================================================
# ROC CURVE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Standard ROC
if 'probs_std' in dir() and 'labels_std' in dir():
    fpr_std, tpr_std, _ = roc_curve(labels_std, probs_std)
    roc_auc_std = auc(fpr_std, tpr_std)
    ax.plot(fpr_std, tpr_std, 'b-', linewidth=2, label=f'🔵 Standard (AUC = {roc_auc_std:.4f})')

# Advanced ROC
if 'probs_adv' in dir() and 'labels_adv' in dir():
    fpr_adv, tpr_adv, _ = roc_curve(labels_adv, probs_adv)
    roc_auc_adv = auc(fpr_adv, tpr_adv)
    ax.plot(fpr_adv, tpr_adv, 'g-', linewidth=2, label=f'🟢 Advanced (AUC = {roc_auc_adv:.4f})')

# Random baseline
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('📈 ROC Curve - So sánh 2 Mô hình', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'roc_curve_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: roc_curve_comparison.png")

# ============================================================
# BAR CHART SO SÁNH
# ============================================================
if len(all_metrics) == 2:
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    std_values = [all_metrics['standard'][m.lower().replace('-', '')] for m in ['accuracy', 'precision', 'recall', 'f1']]
    adv_values = [all_metrics['advanced'][m.lower().replace('-', '')] for m in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, std_values, width, label='🔵 Standard', color='royalblue')
    bars2 = ax.bar(x + width/2, adv_values, width, label='🟢 Advanced', color='forestgreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('📊 So sánh Metrics - Standard vs Advanced', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị trên bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'metrics_comparison_bar.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: metrics_comparison_bar.png")

# ============================================================
# LƯU KẾT QUẢ
# ============================================================
final_results = {
    'standard': all_metrics.get('standard', {}),
    'advanced': all_metrics.get('advanced', {}),
    'comparison': {
        'accuracy_diff': all_metrics.get('advanced', {}).get('accuracy', 0) - all_metrics.get('standard', {}).get('accuracy', 0),
        'f1_diff': all_metrics.get('advanced', {}).get('f1', 0) - all_metrics.get('standard', {}).get('f1', 0),
    }
}

with open(os.path.join(EVAL_RESULTS_DIR, 'detailed_results.json'), 'w') as f:
    json.dump(final_results, f, indent=4)

print("\n" + "=" * 60)
print("✅ TẤT CẢ BIỂU ĐỒ ĐÃ ĐƯỢC LƯU!")
print("=" * 60)
print(f"\n📁 Các file đã tạo trong {EVAL_RESULTS_DIR}:")
print("   📈 training_curves.png")
print("   📈 model_comparison.png")
print("   📊 confusion_matrix_standard.png")
print("   📊 confusion_matrix_advanced.png")
print("   📈 roc_curve_comparison.png")
print("   📊 metrics_comparison_bar.png")
print("   📋 detailed_results.json")
```

---

## 📋 Tóm tắt cấu hình tối ưu:

### 🚀 CHỌN CHẾ ĐỘ GPU (Cell 1)

| Chế độ | Khi nào dùng | Batch Size | Training Speed |
|--------|--------------|------------|----------------|
| **Single GPU** | Debug, batch nhỏ (≤16) | 16 | Baseline (100%) ⭐⭐⭐ |
| **⭐ DataParallel** | **KHUYẾN NGHỊ** - 2 GPU trên Kaggle | 32 | **~140%** ⭐⭐⭐⭐ |
| **DDP** | Advanced (cần multi-process setup) | 48 | ~180% (nếu setup được) |

**Để dùng DataParallel (KHUYẾN NGHỊ):**
```python
# Cell 1: Đặt
GPU_MODE = 'dataparallel'  # ⭐ Đơn giản và hiệu quả!
```

**💡 Tại sao DataParallel thay vì DDP?**
- ✅ **Đơn giản**: Không cần setup multi-process
- ✅ **Stable**: Chạy ổn định trên Kaggle notebook
- ✅ **Nhanh hơn Single GPU ~40%**: Đủ tốt cho hầu hết cases
- ⚠️ DDP nhanh hơn (~80%) nhưng phức tạp trên notebook environment

### 📊 Cấu hình chi tiết

| Cấu hình | Fast Mode | Full Mode |
|----------|-----------|-----------|
| **🚀 FAST_TRAINING** | ✅ True | ❌ False |
| **IMAGE_SIZE** | 224x224 | 380x380 |
| **Augmentation** | Minimal | Full Deepfake |
| **Training Speed** | **~0.5s/it** 🔥 | ~5s/it |
| **1 Epoch Time** | **~10 min** | ~41h |
| **Use Case** | Quick baseline, debugging | Final robust model |

**🎯 Khuyến nghị:**
- **Bắt đầu với FAST_TRAINING=True** để test pipeline nhanh
- Sau khi model ổn → đổi sang False để train model robust hơn

### Advanced Model Specific:

| Cấu hình | Standard | Advanced |
|----------|----------|----------|
| **Batch (Single GPU)** | 16 | 2 |
| **Batch (DataParallel)** | 32 | 4 |
| **🔥 NUM_WORKERS** | 2 (Single) / 4 (Multi) | 2 (Single) / 4 (Multi) |
| **🔥 PREFETCH_FACTOR** | 2 (Single) / 4 (Multi) | 2 (Single) / 4 (Multi) |
| **🔥 SEQUENCE_LENGTH** | N/A | 5 frames (tối ưu!) |

**💡 Tối ưu Multi-GPU:**
- NUM_WORKERS tự động tăng 2→4 khi dùng 2 GPU
- PREFETCH_FACTOR tăng 2→4 để GPU không đói data
- SEQUENCE_LENGTH giảm 10→5 để giảm CPU bottleneck
- **FAST_TRAINING=True để train nhanh gấp 10x**

### 🔥 CẢI TIẾN MỚI - GIẢM OVERFITTING

| Cải tiến | Standard | Advanced |
|----------|----------|----------|
| **Data Augmentation** | ✅ Full | ✅ Full |
| **Early Stopping** | ❌ | ✅ Patience=5 |
| **Learning Rate** | 0.0001 | 🔥 0.00005 |
| **Weight Decay** | 1e-4 | 🔥 1e-3 |
| **Dropout (LSTM)** | 0.3 | 🔥 0.5 |
| **Dropout (Classifier)** | 0.5/0.3 | 🔥 0.6/0.5 |
| **Label Smoothing** | ❌ | ✅ 0.1 |
| **Scheduler** | Cosine | 🔥 ReduceLROnPlateau |

### 🔥 Deepfake-Specific Augmentations

| Augmentation | Mô tả | Probability |
|--------------|-------|-------------|
| **JPEGCompression** | Mô phỏng video compression artifacts | 50% |
| **GaussianNoise** | Mô phỏng camera noise | 30% |
| **GaussianBlur** | Mô phỏng motion blur | 20% |
| **FaceCutout** | Random che một phần mặt | 30% |
| **ColorJitter** | Thay đổi brightness/contrast | 100% |
| **RandomErasing** | Xóa ngẫu nhiên một vùng nhỏ | 5% |

### 📈 Kỳ vọng kết quả sau cải tiến

| Model | Trước | Sau (kỳ vọng) |
|-------|-------|---------------|
| **Standard** | 98.76% | 98-99% (giữ nguyên hoặc tốt hơn) |
| **Advanced** | 91.56% (overfitting) | 93-95% (ít overfitting hơn) |

---

## 🚀 HƯỚNG DẪN SỬ DỤNG 2 GPU HIỆU QUẢ

### ⚡ Quick Start với DDP (NHANH NHẤT):

1. **Cell 1**: Đặt `GPU_MODE = 'ddp'`
2. Code tự động:
   - ✅ Sử dụng DistributedDataParallel
   - ✅ DistributedSampler cho data loading
   - ✅ Batch size tối ưu (24/GPU = 48 total)
3. **Training speed**: Tăng ~80% so với Single GPU!

### 📖 Chi tiết & Troubleshooting:

Xem file [`DDP_KAGGLE_GUIDE.md`](./DDP_KAGGLE_GUIDE.md) để:
- So sánh chi tiết DDP vs DataParallel vs Single GPU
- Benchmarks thực tế
- Tips tối ưu hóa performance
- Xử lý lỗi DDP trên Kaggle

### ⚠️ Lưu ý quan trọng:

| Vấn đề | Giải pháp |
|--------|-----------|
| Training chậm với Single GPU | ✅ Đổi sang `GPU_MODE = 'dataparallel'` |
| OOM khi tăng batch size | Giảm batch size từ 32 → 24 hoặc 16 |
| Muốn tốc độ tối đa | Setup DDP (xem DDP_KAGGLE_GUIDE.md) |

---

## 🎯 TL;DR - BẮT ĐẦU NGAY

**Muốn train nhanh nhất với 2 GPU T4 trên Kaggle?**

```python
# Cell 1: GPU Setup
GPU_MODE = 'dataparallel'  # ⭐ Đơn giản và hiệu quả!

# Chạy tất cả cells như bình thường
# Model sẽ tự động optimize cho 2 GPU
```

**Kết quả:**
- ✅ Training speed tăng ~40%
- ✅ Batch size gấp đôi (16 → 32)
- ✅ Ít overfitting hơn nhờ batch lớn hơn
- ✅ Đơn giản, stable, không cần setup phức tạp!

**🔥 Muốn nhanh hơn nữa (~80%)?**
- Xem hướng dẫn DDP setup trong `DDP_KAGGLE_GUIDE.md`
- Lưu ý: Phức tạp hơn, cần multi-process setup

