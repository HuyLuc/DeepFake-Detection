# src/models/temporal_model.py
"""
Temporal Model: EfficientNet + LSTM/GRU để tận dụng thông tin thời gian.
Phát hiện flickering và sự không nhất quán giữa các frames.
"""

import torch
import torch.nn as nn
import timm


class TemporalDeepfakeModel(nn.Module):
    """
    Model kết hợp EfficientNet-B4 (feature extractor) + LSTM (temporal learning).
    
    Kiến trúc:
        Input: (batch, seq_len, C, H, W) - sequence of frames
        ↓
        EfficientNet-B4: Extract features cho từng frame
        ↓
        Features: (batch, seq_len, feature_dim)
        ↓
        LSTM: Học temporal patterns
        ↓
        FC: Classification (FAKE/REAL)
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b4',
        num_classes: int = 2,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        bidirectional: bool = True,
        freeze_backbone_layers: int = 0,
        use_gru: bool = False,
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Tên backbone (efficientnet_b4, efficientnet_b0, etc.)
            num_classes: Số lớp phân loại (2: FAKE/REAL)
            lstm_hidden_size: Kích thước hidden state của LSTM
            lstm_num_layers: Số layers của LSTM
            lstm_dropout: Dropout giữa các LSTM layers
            bidirectional: Sử dụng Bidirectional LSTM
            freeze_backbone_layers: Số layers đầu của backbone cần freeze
            use_gru: Sử dụng GRU thay vì LSTM
            pretrained: Sử dụng pretrained weights cho backbone
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.bidirectional = bidirectional
        self.use_gru = use_gru
        
        # 1. Backbone: EfficientNet để extract features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier head, chỉ lấy features
        )
        
        # Lấy feature dimension từ backbone
        self.feature_dim = self.backbone.num_features
        print(f"📐 Backbone feature dimension: {self.feature_dim}")
        
        # Freeze một số layers đầu nếu cần (để tiết kiệm VRAM và tránh overfitting)
        if freeze_backbone_layers > 0:
            self._freeze_backbone_layers(freeze_backbone_layers)
        
        # 2. Temporal module: LSTM hoặc GRU
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.temporal = rnn_class(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 3. Classifier
        temporal_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_output_size),
            nn.Dropout(0.5),
            nn.Linear(temporal_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Attention layer để weighted sum các timesteps
        self.attention = nn.Sequential(
            nn.Linear(temporal_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        print(f"✅ TemporalDeepfakeModel initialized:")
        print(f"   - Backbone: {backbone_name}")
        print(f"   - Temporal: {'GRU' if use_gru else 'LSTM'} ({lstm_num_layers} layers, hidden={lstm_hidden_size})")
        print(f"   - Bidirectional: {bidirectional}")
        print(f"   - Classifier output: {num_classes} classes")
    
    def _freeze_backbone_layers(self, num_layers: int):
        """Freeze một số layers đầu của backbone."""
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if frozen_count < num_layers:
                param.requires_grad = False
                frozen_count += 1
        print(f"🔒 Frozen {frozen_count} backbone parameters")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features từ một batch ảnh.
        
        Args:
            x: (batch, C, H, W)
        Returns:
            features: (batch, feature_dim)
        """
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor với shape (batch, seq_len, C, H, W)
               hoặc (batch, C, H, W) cho single frame mode
        
        Returns:
            logits: (batch, num_classes)
        """
        # Kiểm tra input shape
        if x.dim() == 4:
            # Single frame mode: (batch, C, H, W)
            # Thêm dimension seq_len = 1
            x = x.unsqueeze(1)
        
        batch_size, seq_len, C, H, W = x.shape
        
        # 1. Extract features cho từng frame
        # Reshape: (batch * seq_len, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.extract_features(x)
        
        # Reshape lại: (batch, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # 2. Temporal modeling với LSTM/GRU
        temporal_out, _ = self.temporal(features)
        # temporal_out: (batch, seq_len, hidden_size * num_directions)
        
        # 3. Attention-based aggregation
        # Tính attention weights cho mỗi timestep
        attention_weights = self.attention(temporal_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * temporal_out, dim=1)  # (batch, hidden_size * num_directions)
        
        # 4. Classification
        logits = self.classifier(context)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lấy attention weights để visualize frame nào quan trọng nhất.
        
        Returns:
            attention_weights: (batch, seq_len)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, C, H, W = x.shape
        
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.extract_features(x)
        features = features.view(batch_size, seq_len, -1)
        
        temporal_out, _ = self.temporal(features)
        attention_weights = self.attention(temporal_out)
        attention_weights = torch.softmax(attention_weights, dim=1).squeeze(-1)
        
        return attention_weights


class LightweightTemporalModel(nn.Module):
    """
    Phiên bản nhẹ hơn cho máy yếu hoặc inference nhanh.
    Sử dụng EfficientNet-B0 + GRU đơn giản.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # Backbone nhẹ
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # GRU đơn giản
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, C, H, W = x.shape
        
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        
        _, hidden = self.gru(features)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        logits = self.classifier(hidden)
        return logits
