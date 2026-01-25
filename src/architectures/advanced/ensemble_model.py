# src/models/ensemble_model.py
"""
Ensemble Model: Kết hợp EfficientNet + Swin Transformer.
- EfficientNet: Giỏi local features (mắt, miệng, artifact nhỏ)
- Swin Transformer: Giỏi global structure (tương quan giữa các vùng mặt)
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Tuple


class EnsembleDeepfakeModel(nn.Module):
    """
    Ensemble model kết hợp nhiều backbones cho DeepFake Detection.
    
    Kiến trúc:
        Input: (batch, C, H, W)
        ↓
        ┌─────────────────┬─────────────────┐
        │  EfficientNet   │  Swin Transformer│
        │  (Local feat)   │  (Global feat)   │
        └────────┬────────┴────────┬─────────┘
                 │                  │
                 ▼                  ▼
              Features          Features
                 │                  │
                 └────────┬─────────┘
                          ▼
                    Fusion Layer
                          ▼
                     Classifier
    """
    
    def __init__(
        self,
        backbone_configs: Optional[List[dict]] = None,
        num_classes: int = 2,
        fusion_method: str = 'concat',  # 'concat', 'attention', 'weighted_sum'
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone_configs: List các cấu hình backbone, mỗi config gồm:
                - name: Tên model trong timm (e.g., 'efficientnet_b4', 'swin_tiny_patch4_window7_224')
                - weight: Trọng số khi fusion (nếu dùng weighted_sum)
            num_classes: Số lớp phân loại
            fusion_method: Phương pháp kết hợp features ('concat', 'attention', 'weighted_sum')
            pretrained: Sử dụng pretrained weights
            dropout: Dropout rate trước classifier
        """
        super().__init__()
        
        # Default backbone configs nếu không được cung cấp
        if backbone_configs is None:
            backbone_configs = [
                {'name': 'efficientnet_b4', 'weight': 0.5},
                {'name': 'swin_tiny_patch4_window7_224', 'weight': 0.5}
            ]
        
        self.backbone_configs = backbone_configs
        self.fusion_method = fusion_method
        self.num_backbones = len(backbone_configs)
        
        # Tạo các backbones
        self.backbones = nn.ModuleList()
        self.feature_dims = []
        
        for config in backbone_configs:
            backbone = timm.create_model(
                config['name'],
                pretrained=pretrained,
                num_classes=0  # Remove classifier, chỉ lấy features
            )
            self.backbones.append(backbone)
            self.feature_dims.append(backbone.num_features)
            print(f"✅ Loaded backbone: {config['name']} (features: {backbone.num_features})")
        
        # Tổng feature dimension
        self.total_feature_dim = sum(self.feature_dims)
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_output_dim = self.total_feature_dim
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.attention_fc = nn.Linear(self.total_feature_dim, self.num_backbones)
            fusion_output_dim = max(self.feature_dims)  # Output có size = max feature dim
            # Project mỗi backbone về cùng dimension
            self.projection_layers = nn.ModuleList([
                nn.Linear(dim, fusion_output_dim) for dim in self.feature_dims
            ])
        elif fusion_method == 'weighted_sum':
            # Learnable weights
            self.backbone_weights = nn.Parameter(
                torch.tensor([config.get('weight', 1.0/self.num_backbones) for config in backbone_configs])
            )
            fusion_output_dim = max(self.feature_dims)
            self.projection_layers = nn.ModuleList([
                nn.Linear(dim, fusion_output_dim) for dim in self.feature_dims
            ])
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_output_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )
        
        print(f"\n📊 Ensemble Model Summary:")
        print(f"   - Backbones: {[c['name'] for c in backbone_configs]}")
        print(f"   - Feature dims: {self.feature_dims}")
        print(f"   - Fusion method: {fusion_method}")
        print(f"   - Fusion output dim: {fusion_output_dim}")
        print(f"   - Num classes: {num_classes}")
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features từ tất cả backbones.
        
        Args:
            x: Input tensor (batch, C, H, W)
        
        Returns:
            List of feature tensors, một tensor cho mỗi backbone
        """
        features = []
        for backbone in self.backbones:
            feat = backbone(x)
            features.append(feat)
        return features
    
    def fuse_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Kết hợp features từ các backbones.
        
        Args:
            features: List of feature tensors
        
        Returns:
            Fused feature tensor
        """
        if self.fusion_method == 'concat':
            # Simple concatenation
            return torch.cat(features, dim=1)
        
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            concat_features = torch.cat(features, dim=1)
            attention_weights = torch.softmax(self.attention_fc(concat_features), dim=1)
            
            # Project mỗi feature về cùng dimension
            projected = [proj(feat) for proj, feat in zip(self.projection_layers, features)]
            
            # Weighted sum dựa trên attention
            fused = torch.zeros_like(projected[0])
            for i, proj_feat in enumerate(projected):
                fused += attention_weights[:, i:i+1] * proj_feat
            
            return fused
        
        elif self.fusion_method == 'weighted_sum':
            # Learnable weighted sum
            weights = torch.softmax(self.backbone_weights, dim=0)
            
            # Project mỗi feature về cùng dimension
            projected = [proj(feat) for proj, feat in zip(self.projection_layers, features)]
            
            # Weighted sum
            fused = torch.zeros_like(projected[0])
            for i, proj_feat in enumerate(projected):
                fused += weights[i] * proj_feat
            
            return fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, C, H, W)
        
        Returns:
            logits: (batch, num_classes)
        """
        # 1. Extract features từ tất cả backbones
        features = self.extract_features(x)
        
        # 2. Fuse features
        fused = self.fuse_features(features)
        
        # 3. Classification
        logits = self.classifier(fused)
        
        return logits
    
    def get_backbone_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Lấy predictions từ từng backbone riêng lẻ (để analyze).
        
        Returns:
            List of prediction tensors
        """
        predictions = []
        features = self.extract_features(x)
        
        for feat in features:
            # Simple linear classifier cho mỗi backbone
            logits = self.classifier(feat)
            predictions.append(logits)
        
        return predictions


class TemporalEnsembleModel(nn.Module):
    """
    Kết hợp Ensemble + Temporal: 
    Multiple backbones + LSTM cho temporal modeling.
    
    Đây là model mạnh nhất, kết hợp cả local/global features và temporal information.
    """
    
    def __init__(
        self,
        backbone_configs: Optional[List[dict]] = None,
        num_classes: int = 2,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        fusion_method: str = 'concat',
        pretrained: bool = True
    ):
        super().__init__()
        
        if backbone_configs is None:
            backbone_configs = [
                {'name': 'efficientnet_b4', 'weight': 0.5},
                {'name': 'swin_tiny_patch4_window7_224', 'weight': 0.5}
            ]
        
        self.backbone_configs = backbone_configs
        self.fusion_method = fusion_method
        
        # Tạo ensemble extractor (không có classifier)
        self.backbones = nn.ModuleList()
        self.feature_dims = []
        
        for config in backbone_configs:
            backbone = timm.create_model(
                config['name'],
                pretrained=pretrained,
                num_classes=0
            )
            self.backbones.append(backbone)
            self.feature_dims.append(backbone.num_features)
        
        # Fusion dimension
        if fusion_method == 'concat':
            self.fused_dim = sum(self.feature_dims)
        else:
            self.fused_dim = max(self.feature_dims)
            self.projection_layers = nn.ModuleList([
                nn.Linear(dim, self.fused_dim) for dim in self.feature_dims
            ])
        
        # Temporal LSTM
        self.temporal = nn.LSTM(
            input_size=self.fused_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.3 if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention cho temporal aggregation
        temporal_dim = lstm_hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(0.5),
            nn.Linear(temporal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ TemporalEnsembleModel initialized:")
        print(f"   - Backbones: {[c['name'] for c in backbone_configs]}")
        print(f"   - LSTM: hidden={lstm_hidden_size}, layers={lstm_num_layers}")
        print(f"   - Output: {num_classes} classes")
    
    def extract_and_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features từ tất cả backbones và fuse."""
        features = [backbone(x) for backbone in self.backbones]
        
        if self.fusion_method == 'concat':
            return torch.cat(features, dim=1)
        else:
            projected = [proj(feat) for proj, feat in zip(self.projection_layers, features)]
            return torch.stack(projected, dim=0).mean(dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, C, H, W) hoặc (batch, C, H, W)
        
        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract and fuse features cho mỗi frame
        x = x.view(batch_size * seq_len, C, H, W)
        fused_features = self.extract_and_fuse(x)
        fused_features = fused_features.view(batch_size, seq_len, -1)
        
        # Temporal modeling
        temporal_out, _ = self.temporal(fused_features)
        
        # Attention aggregation
        attention_weights = torch.softmax(self.attention(temporal_out), dim=1)
        context = torch.sum(attention_weights * temporal_out, dim=1)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


def create_model(
    model_type: str = 'temporal',
    backbone_name: str = 'efficientnet_b4',
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function để tạo model.
    
    Args:
        model_type: 'simple', 'temporal', 'ensemble', 'temporal_ensemble'
        backbone_name: Tên backbone cho simple/temporal models
        num_classes: Số lớp
        pretrained: Dùng pretrained weights
        **kwargs: Các tham số khác cho từng loại model
    
    Returns:
        Model instance
    """
    if model_type == 'simple':
        # Standard single backbone model
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    
    elif model_type == 'temporal':
        from .temporal_model import TemporalDeepfakeModel
        model = TemporalDeepfakeModel(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    elif model_type == 'ensemble':
        model = EnsembleDeepfakeModel(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    elif model_type == 'temporal_ensemble':
        model = TemporalEnsembleModel(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
