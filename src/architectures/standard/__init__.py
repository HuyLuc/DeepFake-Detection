# src/architectures/standard/__init__.py
"""
🔵 KIẾN TRÚC STANDARD (Kiến trúc 1)

Mô tả: EfficientNet-B4 đơn giản, phân loại từng frame độc lập.

Ưu điểm:
- Nhanh, nhẹ
- Phù hợp GPU yếu (2GB VRAM)
- Dễ debug

Nhược điểm:
- Không học được temporal patterns
- Accuracy thấp hơn Advanced

Sử dụng:
    python main.py train
"""

from .model import create_standard_model
from .dataset import DeepfakeDataset
from .train import run_training
