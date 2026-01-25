# src/architectures/advanced/__init__.py
"""
🟢 KIẾN TRÚC ADVANCED (Kiến trúc 2)

Mô tả: Kết hợp EfficientNet + LSTM + Swin Transformer.
       Học được cả local features, global structure và temporal patterns.

Ưu điểm:
- Accuracy cao nhất (~95%)
- Phát hiện được flickering giữa các frames
- Robust với nhiều loại deepfake

Nhược điểm:
- Cần GPU mạnh (8GB+ VRAM)
- Training lâu hơn

Sử dụng:
    python main.py train_advanced --model temporal_ensemble

Các model có sẵn:
- temporal: EfficientNet + LSTM
- ensemble: EfficientNet + Swin Transformer
- temporal_ensemble: Cả hai (mạnh nhất)
- lightweight: Phiên bản nhẹ cho máy yếu
"""

from .temporal_model import TemporalDeepfakeModel, LightweightTemporalModel
from .ensemble_model import EnsembleDeepfakeModel, TemporalEnsembleModel, create_model
from .temporal_dataset import TemporalDeepfakeDataset, create_temporal_dataloaders
from .train import run_temporal_training
