# src/architectures/__init__.py
"""
Module chứa các kiến trúc model cho DeepFake Detection.

Có 2 kiến trúc chính:
1. Standard: EfficientNet-B4 đơn giản (nhanh, nhẹ)
2. Advanced: EfficientNet + LSTM + Swin Transformer (accuracy cao)
"""

from . import standard
from . import advanced
