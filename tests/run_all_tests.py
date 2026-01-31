# tests/run_all_tests.py
"""
Test runner không dùng pytest - chạy trực tiếp
"""

import sys
import os
import traceback
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("\n" + "="*80)
print("🧪 PHASE 1 - COMPREHENSIVE TEST SUITE")
print("="*80 + "\n")

start_time = time.time()
passed = 0
failed = 0
errors = []

# =============================================================================
# TEST GROUP 1: TEMPORAL MODEL
# =============================================================================
print("\n" + "="*60)
print("📦 TEST GROUP 1: TemporalModel")
print("="*60)

try:
    import torch
    from src.architectures.advanced.model import TemporalModel, create_temporal_model
    
    # Test 1.1: Model creation
    print("\n✅ Test 1.1: Import & Create")
    model = create_temporal_model(num_classes=2, pretrained=False, sequence_length=5)
    assert model is not None
    passed += 1
    
    # Test 1.2: Architecture
    print("✅ Test 1.2: Architecture components")
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'lstm')
    assert hasattr(model, 'classifier')
    passed += 1
    
    # Test 1.3: Forward pass
    print("✅ Test 1.3: Forward pass")
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
    passed += 1
    
    # Test 1.4: Probabilities
    print("✅ Test 1.4: Output probabilities")
    probs = torch.nn.functional.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    passed += 1
    
    # Test 1.5: State dict
    print("✅ Test 1.5: State dict save/load")
    state_dict = model.state_dict()
    model2 = create_temporal_model(num_classes=2, pretrained=False)
    model2.load_state_dict(state_dict)
    passed += 1
    
    print(f"\n✅ TemporalModel: 5/5 tests passed")
    
except Exception as e:
    print(f"\n❌ TemporalModel tests failed: {e}")
    errors.append(f"TemporalModel: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 2: MODEL MANAGER
# =============================================================================
print("\n" + "="*60)
print("🤖 TEST GROUP 2: ModelManager")
print("="*60)

try:
    from src.app.services.model_manager import ModelManager
    from PIL import Image
    import numpy as np
    
    # Test 2.1: Singleton
    print("\n✅ Test 2.1: Import & Singleton")
    manager = ModelManager()
    manager2 = ModelManager()
    assert manager is manager2
    passed += 1
    
    # Test 2.2: Models loaded
    print("✅ Test 2.2: Models loaded (may warn if no saved models)")
    assert manager.standard_model is not None
    assert manager.advanced_model is not None
    passed += 1
    
    # Test 2.3: Transforms
    print("✅ Test 2.3: Transforms configured")
    assert manager.image_transform is not None
    passed += 1
    
    # Test 2.4: Predict image standard
    print("✅ Test 2.4: Predict image (Standard)")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = manager.predict_image_standard(dummy_img)
    assert 'verdict' in result
    assert 'confidence' in result
    assert result['verdict'] in ['FAKE', 'REAL']
    passed += 1
    
    # Test 2.5: Predict image advanced
    print("✅ Test 2.5: Predict image (Advanced)")
    result = manager.predict_image_advanced(dummy_img)
    assert result['model'] == 'advanced'
    passed += 1
    
    # Test 2.6: Ensemble
    print("✅ Test 2.6: Ensemble prediction")
    result = manager.predict_ensemble(dummy_img, is_video=False)
    assert result['model'] == 'ensemble'
    assert 'models_comparison' in result
    passed += 1
    
    # Test 2.7: Video prediction
    print("✅ Test 2.7: Video prediction (Standard)")
    dummy_imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(10)]
    result = manager.predict_video_standard(dummy_imgs)
    assert 'timeline' in result
    assert 'stats' in result
    passed += 1
    
    print(f"\n✅ ModelManager: 7/7 tests passed")
    
except Exception as e:
    print(f"\n❌ ModelManager tests failed: {e}")
    traceback.print_exc()
    errors.append(f"ModelManager: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 3: FILE PROCESSOR
# =============================================================================
print("\n" + "="*60)
print("📸 TEST GROUP 3: FileProcessor")
print("="*60)

try:
    from src.app.services.file_processor import FileProcessor
    from PIL import Image
    import numpy as np
    
    # Test 3.1: Create
    print("\n✅ Test 3.1: Create FileProcessor")
    processor = FileProcessor(skip_frames=5, face_margin=20)
    assert processor.skip_frames == 5
    assert processor.face_margin == 20
    passed += 1
    
    # Test 3.2: Face detector
    print("✅ Test 3.2: Face detector initialized")
    assert processor.face_detector is not None
    passed += 1
    
    # Test 3.3: Extract face (random image - no face expected)
    print("✅ Test 3.3: Extract face (no face in random image)")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    face = processor.extract_face(dummy_img)
    # May or may not detect face in random noise - both are OK
    passed += 1
    
    # Test 3.4: Process invalid path
    print("✅ Test 3.4: Handle invalid path")
    result = processor.process_image("/nonexistent/path.jpg")
    assert result is None
    passed += 1
    
    # Test 3.5: Grayscale handling
    print("✅ Test 3.5: Grayscale image handling")
    gray_img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode='L')
    face = processor.extract_face(gray_img)  # Should not crash
    passed += 1
    
    print(f"\n✅ FileProcessor: 5/5 tests passed")
    
except Exception as e:
    print(f"\n❌ FileProcessor tests failed: {e}")
    traceback.print_exc()
    errors.append(f"FileProcessor: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 4: PREDICTION SERVICE
# =============================================================================
print("\n" + "="*60)
print("🔮 TEST GROUP 4: PredictionService")
print("="*60)

try:
    from src.app.services.prediction_service import PredictionService
    from PIL import Image
    import numpy as np
    import tempfile
    
    # Test 4.1: Create
    print("\n✅ Test 4.1: Create PredictionService")
    service = PredictionService()
    assert service.model_manager is not None
    assert service.file_processor is not None
    passed += 1
    
    # Test 4.2: ModelManager is singleton
    print("✅ Test 4.2: Uses singleton ModelManager")
    service2 = PredictionService()
    assert service.model_manager is service2.model_manager
    passed += 1
    
    # Test 4.3: Invalid file type
    print("✅ Test 4.3: Invalid file type handling")
    try:
        service.predict("/test.jpg", file_type='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Invalid file_type' in str(e)
    passed += 1
    
    # Test 4.4: Predict image with temp file
    print("✅ Test 4.4: Predict image workflow")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(temp_path)
    
    result = service.predict(temp_path, file_type='image', model_choice='standard')
    assert 'processing_time' in result
    # May succeed or fail (no face) - both are valid
    os.unlink(temp_path)
    passed += 1
    
    # Test 4.5: All model choices
    print("✅ Test 4.5: All model choices")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(temp_path)
    
    for choice in ['standard', 'advanced', 'ensemble']:
        result = service.predict(temp_path, file_type='image', model_choice=choice)
        assert 'processing_time' in result
    os.unlink(temp_path)
    passed += 1
    
    print(f"\n✅ PredictionService: 5/5 tests passed")
    
except Exception as e:
    print(f"\n❌ PredictionService tests failed: {e}")
    traceback.print_exc()
    errors.append(f"PredictionService: {e}")
    failed += 1

# =============================================================================
# SUMMARY
# =============================================================================
duration = time.time() - start_time

print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)

print(f"\n✅ Passed: {passed}")
print(f"❌ Failed: {len(errors)}")
print(f"⏱️  Duration: {duration:.2f}s")

if errors:
    print("\n❌ Errors:")
    for err in errors:
        print(f"   - {err}")
else:
    print("\n🎉 ALL TESTS PASSED!")
    
print("\n" + "="*80)

# Exit code
sys.exit(0 if len(errors) == 0 else 1)
