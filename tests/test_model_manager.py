# tests/test_model_manager.py
"""
Test cho ModelManager
"""

import pytest
import torch
import sys
import os
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.app.services.model_manager import ModelManager


class TestModelManager:
    """Test suite cho ModelManager"""
    
    @pytest.fixture(scope="class")
    def manager(self):
        """Fixture: Tạo ModelManager instance (singleton)"""
        return ModelManager()
    
    @pytest.fixture
    def dummy_image(self):
        """Fixture: Tạo dummy PIL Image"""
        # Random RGB image 224x224
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    
    @pytest.fixture
    def dummy_images(self):
        """Fixture: Tạo list of dummy images (for video)"""
        images = []
        for _ in range(10):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images.append(Image.fromarray(arr))
        return images
    
    def test_singleton_pattern(self, manager):
        """Test 1: ModelManager là singleton"""
        manager2 = ModelManager()
        assert manager is manager2
        print("✅ Test 1 passed: Singleton pattern")
    
    def test_initialization(self, manager):
        """Test 2: ModelManager initialized correctly"""
        assert manager is not None
        assert manager.standard_model is not None
        assert manager.advanced_model is not None
        assert manager.device is not None
        print(f"✅ Test 2 passed: Initialized on {manager.device}")
    
    def test_models_in_eval_mode(self, manager):
        """Test 3: Models ở eval mode"""
        assert manager.standard_model.training == False
        assert manager.advanced_model.training == False
        print("✅ Test 3 passed: Models in eval mode")
    
    def test_transforms_exist(self, manager):
        """Test 4: Transforms được config"""
        assert manager.image_transform is not None
        print("✅ Test 4 passed: Transforms configured")
    
    def test_class_names(self, manager):
        """Test 5: Class names"""
        assert manager.class_names == ['FAKE', 'REAL']
        print("✅ Test 5 passed: Class names")
    
    def test_predict_image_standard(self, manager, dummy_image):
        """Test 6: Predict single image với Standard model"""
        result = manager.predict_image_standard(dummy_image)
        
        # Check result structure
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'model' in result
        
        # Check values
        assert result['verdict'] in ['FAKE', 'REAL']
        assert 0 <= result['confidence'] <= 1
        assert result['model'] == 'standard'
        assert 'FAKE' in result['probabilities']
        assert 'REAL' in result['probabilities']
        
        # Check probabilities sum to 1
        prob_sum = result['probabilities']['FAKE'] + result['probabilities']['REAL']
        assert abs(prob_sum - 1.0) < 1e-6
        
        print(f"✅ Test 6 passed: Standard prediction = {result['verdict']} ({result['confidence']:.4f})")
    
    def test_predict_image_advanced(self, manager, dummy_image):
        """Test 7: Predict single image với Advanced model"""
        result = manager.predict_image_advanced(dummy_image)
        
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['model'] == 'advanced'
        assert result['verdict'] in ['FAKE', 'REAL']
        
        print(f"✅ Test 7 passed: Advanced prediction = {result['verdict']} ({result['confidence']:.4f})")
    
    def test_predict_video_standard(self, manager, dummy_images):
        """Test 8: Predict video với Standard model"""
        result = manager.predict_video_standard(dummy_images)
        
        # Check result structure
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'timeline' in result
        assert 'stats' in result
        assert result['model'] == 'standard'
        
        # Check stats
        assert result['stats']['total_frames'] == len(dummy_images)
        assert result['stats']['fake_count'] >= 0
        assert result['stats']['real_count'] >= 0
        assert 0 <= result['stats']['fake_ratio'] <= 1
        
        # Check timeline
        assert len(result['timeline']) == len(dummy_images)
        for item in result['timeline']:
            assert 'frame' in item
            assert 'verdict' in item
            assert 'confidence' in item
        
        print(f"✅ Test 8 passed: Video prediction = {result['verdict']}")
        print(f"   Stats: {result['stats']['fake_count']} FAKE, {result['stats']['real_count']} REAL")
    
    def test_predict_video_advanced(self, manager, dummy_images):
        """Test 9: Predict video với Advanced model"""
        result = manager.predict_video_advanced(dummy_images)
        
        assert 'verdict' in result
        assert 'timeline' in result
        assert 'stats' in result
        assert result['model'] == 'advanced'
        
        # Advanced model uses sequences, so timeline length different
        assert len(result['timeline']) <= len(dummy_images)
        
        print(f"✅ Test 9 passed: Advanced video prediction")
        print(f"   Sequences analyzed: {result['stats']['sequences_analyzed']}")
    
    def test_predict_ensemble_image(self, manager, dummy_image):
        """Test 10: Ensemble prediction cho image"""
        result = manager.predict_ensemble(dummy_image, is_video=False)
        
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'models_comparison' in result
        assert result['model'] == 'ensemble'
        
        # Check models comparison
        assert 'standard' in result['models_comparison']
        assert 'advanced' in result['models_comparison']
        
        print(f"✅ Test 10 passed: Ensemble image prediction")
        print(f"   Standard: {result['models_comparison']['standard']['verdict']}")
        print(f"   Advanced: {result['models_comparison']['advanced']['verdict']}")
        print(f"   Ensemble: {result['verdict']}")
    
    def test_predict_ensemble_video(self, manager, dummy_images):
        """Test 11: Ensemble prediction cho video"""
        result = manager.predict_ensemble(dummy_images, is_video=True)
        
        assert 'verdict' in result
        assert 'models_comparison' in result
        assert result['model'] == 'ensemble'
        
        print(f"✅ Test 11 passed: Ensemble video prediction = {result['verdict']}")
    
    def test_prepare_image(self, manager, dummy_image):
        """Test 12: Image preprocessing"""
        tensor = manager._prepare_image(dummy_image)
        
        # Check shape: (1, 3, 380, 380)
        assert tensor.shape == (1, 3, 380, 380)
        assert tensor.device == manager.device
        
        print(f"✅ Test 12 passed: Image preprocessing, shape = {tensor.shape}")
    
    def test_prepare_sequence(self, manager, dummy_images):
        """Test 13: Sequence preprocessing"""
        # Take first 5 images
        sequence = dummy_images[:5]
        tensor = manager._prepare_sequence(sequence)
        
        # Check shape: (1, 5, 3, 380, 380)
        assert tensor.shape == (1, 5, 3, 380, 380)
        assert tensor.device == manager.device
        
        print(f"✅ Test 13 passed: Sequence preprocessing, shape = {tensor.shape}")
    
    def test_consistency_multiple_predictions(self, manager, dummy_image):
        """Test 14: Multiple predictions cho cùng image should be consistent"""
        result1 = manager.predict_image_standard(dummy_image)
        result2 = manager.predict_image_standard(dummy_image)
        
        # Results should be identical (model in eval mode, no dropout)
        assert result1['verdict'] == result2['verdict']
        assert abs(result1['confidence'] - result2['confidence']) < 1e-6
        
        print("✅ Test 14 passed: Prediction consistency")


def test_model_paths_config():
    """Test 15: Model paths configuration"""
    manager = ModelManager()
    
    assert hasattr(manager, 'standard_model_path')
    assert hasattr(manager, 'advanced_model_path')
    assert 'standard' in manager.standard_model_path
    assert 'advanced' in manager.advanced_model_path
    
    print("✅ Test 15 passed: Model paths configured")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Running ModelManager Tests...")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
