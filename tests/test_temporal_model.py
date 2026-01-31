# tests/test_temporal_model.py
"""
Test cho TemporalModel (Advanced Model)
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architectures.advanced.model import TemporalModel, create_temporal_model


class TestTemporalModel:
    """Test suite cho TemporalModel"""
    
    @pytest.fixture
    def model(self):
        """Fixture: Tạo model instance"""
        return create_temporal_model(num_classes=2, pretrained=False, sequence_length=5)
    
    def test_model_creation(self, model):
        """Test 1: Model được tạo thành công"""
        assert model is not None
        assert isinstance(model, TemporalModel)
        print("✅ Test 1 passed: Model creation")
    
    def test_model_architecture(self, model):
        """Test 2: Kiểm tra architecture components"""
        # Check backbone
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'classifier')
        
        # Check LSTM config
        assert model.lstm.input_size == model.backbone.num_features
        assert model.lstm.hidden_size == 512
        assert model.lstm.num_layers == 2
        assert model.lstm.bidirectional == True
        
        print("✅ Test 2 passed: Architecture components")
    
    def test_forward_pass_shape(self, model):
        """Test 3: Forward pass với dummy input"""
        batch_size = 2
        sequence_length = 5
        channels = 3
        height = width = 224
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, sequence_length, channels, height, width)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (batch_size, 2)  # (batch, num_classes)
        print(f"✅ Test 3 passed: Forward pass shape = {output.shape}")
    
    def test_output_range(self, model):
        """Test 4: Output logits có thể convert thành probabilities"""
        batch_size = 1
        dummy_input = torch.randn(batch_size, 5, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Check probabilities sum to 1
        prob_sum = probs.sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6)
        
        # Check probabilities in range [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()
        
        print(f"✅ Test 4 passed: Probabilities = {probs}")
    
    def test_different_batch_sizes(self, model):
        """Test 5: Model hoạt động với different batch sizes"""
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            dummy_input = torch.randn(batch_size, 5, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape[0] == batch_size
        
        print("✅ Test 5 passed: Different batch sizes")
    
    def test_model_eval_mode(self, model):
        """Test 6: Model chuyển sang eval mode"""
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False
        
        print("✅ Test 6 passed: Train/Eval mode switching")
    
    def test_model_parameters_exist(self, model):
        """Test 7: Model có parameters"""
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check có gradient
        total_params = sum(p.numel() for p in params)
        print(f"✅ Test 7 passed: Total parameters = {total_params:,}")
    
    def test_gradient_flow(self, model):
        """Test 8: Gradient flow trong training mode"""
        model.train()
        
        # Create dummy input và target
        dummy_input = torch.randn(2, 5, 3, 224, 224)
        dummy_target = torch.tensor([0, 1])  # FAKE, REAL
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
        
        print(f"✅ Test 8 passed: Gradient flow OK, Loss = {loss.item():.4f}")


def test_factory_function():
    """Test 9: Factory function create_temporal_model()"""
    model = create_temporal_model(num_classes=2, pretrained=False, sequence_length=5)
    assert isinstance(model, TemporalModel)
    print("✅ Test 9 passed: Factory function")


def test_model_state_dict():
    """Test 10: Model state_dict can be saved/loaded"""
    model1 = create_temporal_model(num_classes=2, pretrained=False)
    
    # Get state dict
    state_dict = model1.state_dict()
    assert len(state_dict) > 0
    
    # Create new model and load state dict
    model2 = create_temporal_model(num_classes=2, pretrained=False)
    model2.load_state_dict(state_dict)
    
    # Verify weights are the same
    for key in state_dict.keys():
        assert torch.allclose(model1.state_dict()[key], model2.state_dict()[key])
    
    print("✅ Test 10 passed: State dict save/load")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Running TemporalModel Tests...")
    print("="*60 + "\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
