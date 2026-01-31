# src/architectures/advanced/model.py
"""
Advanced Temporal Model: EfficientNet-B4 + LSTM
Phát hiện deepfake dựa trên sequence of frames
"""

import torch
import torch.nn as nn
import timm


class TemporalModel(nn.Module):
    """
    Temporal Model với TĂNG DROPOUT để giảm overfitting
    
    Architecture:
        - Backbone: EfficientNet-B4 (feature extraction)
        - Temporal: Bidirectional LSTM (2 layers)
        - Classifier: Fully connected layers với dropout
    
    Input: (batch, sequence_length, channels, height, width)
    Output: (batch, num_classes)
    """
    
    def __init__(self, num_classes=2, pretrained=False, sequence_length=5):
        """
        Args:
            num_classes: Số lượng classes (2: FAKE/REAL)
            pretrained: Có sử dụng pretrained weights không (False khi load from checkpoint)
            sequence_length: Số frames trong 1 sequence (default: 5)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Backbone: EfficientNet-B4 (extract features từ mỗi frame)
        self.backbone = timm.create_model(
            'efficientnet_b4', 
            pretrained=pretrained, 
            num_classes=0  # Không có classifier, chỉ lấy features
        )
        backbone_dim = self.backbone.num_features  # 1792 for EfficientNet-B4
        
        # LSTM: Process sequence of features
        self.lstm = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,       # Input shape: (batch, seq, feature)
            bidirectional=True,     # 2 directions → output size = 512*2
            dropout=0.5             # Dropout giữa các LSTM layers
        )
        
        # Classifier: FC layers với dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),                    # Dropout cao để giảm overfitting
            nn.Linear(512 * 2, 256),            # 512*2 vì bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ TemporalModel initialized:")
        print(f"   - Backbone: EfficientNet-B4 ({backbone_dim} features)")
        print(f"   - LSTM: 2 layers, bidirectional, hidden_size=512")
        print(f"   - Sequence length: {sequence_length} frames")
        print(f"   - Output: {num_classes} classes")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor of shape (batch, sequence_length, channels, height, width)
               Ví dụ: (2, 5, 3, 224, 224)
        
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape để feed vào backbone:
        # (batch, seq, C, H, W) → (batch*seq, C, H, W)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features từ mỗi frame
        features = self.backbone(x)  # (batch*seq, backbone_dim)
        
        # Reshape lại để feed vào LSTM:
        # (batch*seq, features) → (batch, seq, features)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM forward
        # lstm_out shape: (batch, seq, hidden_size*2)
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Lấy output của frame cuối cùng
        # lstm_out[:, -1, :] shape: (batch, hidden_size*2)
        final_features = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(final_features)  # (batch, num_classes)
        
        return logits


def create_temporal_model(num_classes=2, pretrained=False, sequence_length=5):
    """
    Factory function để tạo TemporalModel
    
    Args:
        num_classes: Số lượng classes
        pretrained: Có sử dụng pretrained EfficientNet không
        sequence_length: Số frames trong sequence
    
    Returns:
        TemporalModel instance
    """
    model = TemporalModel(
        num_classes=num_classes,
        pretrained=pretrained,
        sequence_length=sequence_length
    )
    return model


# Test nếu run file này trực tiếp
if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing TemporalModel...")
    print("="*60)
    
    # Tạo model
    model = create_temporal_model(num_classes=2, pretrained=False, sequence_length=5)
    
    # Tạo input giả
    batch_size = 2
    sequence_length = 5
    channels = 3
    height = width = 224
    
    dummy_input = torch.randn(batch_size, sequence_length, channels, height, width)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Softmax để lấy probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    print(f"\nProbabilities:\n{probs}")
    
    # Predictions
    preds = torch.argmax(probs, dim=1)
    print(f"\nPredictions (class indices): {preds}")
    
    print("\n✅ TemporalModel test passed!")
