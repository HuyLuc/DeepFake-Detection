# src/architectures/standard/model.py
"""
🔵 STANDARD ARCHITECTURE - Model
EfficientNet-B4 đơn giản.
"""

import timm
import torch.nn as nn


def create_standard_model(
    model_name: str = 'efficientnet_b4',
    num_classes: int = 2,
    pretrained: bool = True
) -> nn.Module:
    """
    Tạo model EfficientNet-B4 cho kiến trúc Standard.
    
    Args:
        model_name: Tên model trong timm
        num_classes: Số lớp phân loại (2: FAKE/REAL)
        pretrained: Sử dụng pretrained weights
    
    Returns:
        Model instance
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    print(f"✅ Created Standard Model: {model_name}")
    print(f"   - Pretrained: {pretrained}")
    print(f"   - Num classes: {num_classes}")
    
    return model
