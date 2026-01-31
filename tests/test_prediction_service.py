# tests/test_prediction_service.py
"""
Test cho PredictionService (Integration tests)
"""

import pytest
import sys
import os
from PIL import Image
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.app.services.prediction_service import PredictionService


class TestPredictionService:
    """Test suite cho PredictionService"""
    
    @pytest.fixture(scope="class")
    def service(self):
        """Fixture: Tạo PredictionService instance"""
        return PredictionService()
    
    @pytest.fixture
    def temp_image_file(self, tmp_path):
        """Fixture: Tạo temporary image file"""
        temp_path = tmp_path / "test_image.jpg"
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(temp_path)
        return str(temp_path)
    
    def test_initialization(self, service):
        """Test 1: Service initialized correctly"""
        assert service is not None
        assert service.model_manager is not None
        assert service.file_processor is not None
        print("✅ Test 1 passed: Service initialization")
    
    def test_model_manager_loaded(self, service):
        """Test 2: ModelManager được load"""
        assert service.model_manager.standard_model is not None
        assert service.model_manager.advanced_model is not None
        print("✅ Test 2 passed: Models loaded")
    
    def test_file_processor_configured(self, service):
        """Test 3: FileProcessor configured"""
        assert service.file_processor.skip_frames == 5
        assert service.file_processor.face_margin == 20
        print("✅ Test 3 passed: FileProcessor configured")
    
    def test_predict_invalid_file_type(self, service, temp_image_file):
        """Test 4: Invalid file_type raises error"""
        with pytest.raises(ValueError, match="Invalid file_type"):
            service.predict(temp_image_file, file_type='invalid')
        print("✅ Test 4 passed: Invalid file_type handled")
    
    def test_predict_image_standard(self, service, temp_image_file):
        """Test 5: Predict image với Standard model"""
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='standard'
        )
        
        # Check result structure
        assert 'processing_time' in result
        
        # Có thể success hoặc fail (no face)
        if result.get('success'):
            assert 'verdict' in result
            assert 'confidence' in result
            assert 'model_used' in result
            assert result['model_used'] == 'standard'
            print(f"✅ Test 5 passed: Standard prediction = {result['verdict']}")
        else:
            assert 'error' in result
            print(f"⚠️  Test 5: No face detected (expected for random image)")
    
    def test_predict_image_advanced(self, service, temp_image_file):
        """Test 6: Predict image với Advanced model"""
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='advanced'
        )
        
        assert 'processing_time' in result
        
        if result.get('success'):
            assert result['model_used'] == 'advanced'
            print(f"✅ Test 6 passed: Advanced prediction = {result['verdict']}")
        else:
            print(f"⚠️  Test 6: No face detected")
    
    def test_predict_image_ensemble(self, service, temp_image_file):
        """Test 7: Predict image với Ensemble"""
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='ensemble'
        )
        
        assert 'processing_time' in result
        
        if result.get('success'):
            assert result['model_used'] == 'ensemble'
            # Ensemble should have models_comparison
            if 'models_comparison' in result:
                assert 'standard' in result['models_comparison']
                assert 'advanced' in result['models_comparison']
            print(f"✅ Test 7 passed: Ensemble prediction = {result['verdict']}")
        else:
            print(f"⚠️  Test 7: No face detected")
    
    def test_predict_image_no_face(self, service, tmp_path):
        """Test 8: Predict image không có face"""
        # Tạo solid color image (definitely no face)
        temp_path = tmp_path / "no_face.jpg"
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        img = Image.fromarray(arr)
        img.save(temp_path)
        
        result = service.predict(
            str(temp_path),
            file_type='image',
            model_choice='standard'
        )
        
        # Should return error về no face
        if not result.get('success'):
            assert 'error' in result
            assert 'face' in result['error'].lower()
            print("✅ Test 8 passed: No face error handled")
        else:
            print("⚠️  Test 8: Face detected in solid color (false positive)")
    
    def test_predict_with_options(self, service, temp_image_file):
        """Test 9: Predict với options"""
        options = {
            'show_timeline': True,
            'threshold': 0.85
        }
        
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='standard',
            options=options
        )
        
        assert 'processing_time' in result
        print("✅ Test 9 passed: Options passed successfully")
    
    def test_processing_time_tracked(self, service, temp_image_file):
        """Test 10: Processing time được track"""
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='standard'
        )
        
        assert 'processing_time' in result
        assert isinstance(result['processing_time'], (int, float))
        assert result['processing_time'] >= 0
        
        print(f"✅ Test test10 passed: Processing time = {result['processing_time']:.2f}s")
    
    def test_predict_image_details(self, service, temp_image_file):
        """Test 11: Result details cho image"""
        result = service.predict(
            temp_image_file,
            file_type='image',
            model_choice='standard'
        )
        
        assert 'details' in result
        assert 'face_detected' in result['details']
        
        if result.get('success'):
            assert result['details']['face_detected'] == True
            assert 'face_size' in result['details']
            print(f"✅ Test 11 passed: Image details included")
        else:
            assert result['details']['face_detected'] == False
            print("✅ Test 11 passed: No face details")
    
    def test_error_handling_invalid_file(self, service):
        """Test 12: Error handling cho invalid file path"""
        with pytest.raises(Exception):
            service.predict(
                "/nonexistent/file.jpg",
                file_type='image',
                model_choice='standard'
            )
        print("✅ Test 12 passed: Invalid file error handling")
    
    def test_all_model_choices(self, service, temp_image_file):
        """Test 13: Tất cả model choices"""
        model_choices = ['standard', 'advanced', 'ensemble']
        
        for choice in model_choices:
            try:
                result = service.predict(
                    temp_image_file,
                    file_type='image',
                    model_choice=choice
                )
                assert 'processing_time' in result
                print(f"✅ Model choice '{choice}' works")
            except Exception as e:
                pytest.fail(f"Model choice '{choice}' failed: {e}")
        
        print("✅ Test 13 passed: All model choices work")
    
    def test_invalid_model_choice(self, service, temp_image_file):
        """Test 14: Invalid model choice"""
        with pytest.raises(ValueError, match="Invalid model_choice"):
            service.predict(
                temp_image_file,
                file_type='image',
                model_choice='invalid_model'
            )
        print("✅ Test 14 passed: Invalid model choice handled")


def test_service_can_be_recreated():
    """Test 15: Service có thể được recreated"""
    service1 = PredictionService()
    service2 = PredictionService()
    
    # Both should work (ModelManager is singleton, but service is not)
    assert service1 is not service2
    assert service1.model_manager is service2.model_manager  # Singleton
    
    print("✅ Test 15 passed: Service can be recreated")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Running PredictionService Tests...")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
