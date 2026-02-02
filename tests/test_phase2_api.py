# tests/test_phase2_api.py
"""
Tests cho Phase 2: API Layer, Database, Export
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("\n" + "="*80)
print("🧪 PHASE 2 - API LAYER TESTS")
print("="*80 + "\n")

passed = 0
failed = 0
errors = []

# =============================================================================
# TEST GROUP 1: DATABASE
# =============================================================================
print("="*60)
print("💾 TEST GROUP 1: Database")
print("="*60)

try:
    # Test 1.1: Import database models
    print("\n✅ Test 1.1: Import database models")
    from src.app.models.database import db, PredictionHistory, init_db
    passed += 1
    
    # Test 1.2: Create Flask app and init database
    print("✅ Test 1.2: Initialize database")
    from flask import Flask
    
    test_app = Flask(__name__)
    test_app.config['TESTING'] = True
    test_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    test_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(test_app)
    with test_app.app_context():
        db.create_all()
    passed += 1
    
    # Test 1.3: Create PredictionHistory record
    print("✅ Test 1.3: Create PredictionHistory record")
    with test_app.app_context():
        history = PredictionHistory(
            file_name='test.jpg',
            file_type='image',
            model_used='standard',
            verdict='FAKE',
            confidence=0.95,
            fake_probability=0.95,
            real_probability=0.05,
            processing_time=1.5
        )
        db.session.add(history)
        db.session.commit()
        
        assert history.id is not None
        assert history.verdict == 'FAKE'
    passed += 1
    
    # Test 1.4: Query records
    print("✅ Test 1.4: Query records")
    with test_app.app_context():
        count = PredictionHistory.query.count()
        assert count >= 1
    passed += 1
    
    # Test 1.5: to_dict()
    print("✅ Test 1.5: to_dict() method")
    with test_app.app_context():
        h = PredictionHistory.query.first()
        d = h.to_dict()
        assert 'id' in d
        assert 'verdict' in d
        assert 'confidence' in d
    passed += 1
    
    print(f"\n✅ Database: 5/5 tests passed")
    
except Exception as e:
    print(f"\n❌ Database tests failed: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Database: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 2: HISTORY SERVICE
# =============================================================================
print("\n" + "="*60)
print("📜 TEST GROUP 2: HistoryService")
print("="*60)

try:
    from src.app.services.history_service import HistoryService
    
    # Test 2.1: Import and create
    print("\n✅ Test 2.1: Import HistoryService")
    passed += 1
    
    # Test 2.2: Methods exist
    print("✅ Test 2.2: Methods exist")
    assert hasattr(HistoryService, 'save_prediction')
    assert hasattr(HistoryService, 'get_history')
    assert hasattr(HistoryService, 'get_by_id')
    assert hasattr(HistoryService, 'delete_prediction')
    assert hasattr(HistoryService, 'get_statistics')
    passed += 1
    
    print(f"\n✅ HistoryService: 2/2 tests passed")
    
except Exception as e:
    print(f"\n❌ HistoryService tests failed: {e}")
    errors.append(f"HistoryService: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 3: EXPORT SERVICE
# =============================================================================
print("\n" + "="*60)
print("📤 TEST GROUP 3: ExportService")
print("="*60)

try:
    from src.app.services.export_service import ExportService
    
    # Test 3.1: Create ExportService
    print("\n✅ Test 3.1: Create ExportService")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_service = ExportService(export_dir=tmpdir)
        assert os.path.exists(export_service.json_dir)
        assert os.path.exists(export_service.pdf_dir)
    passed += 1
    
    # Test 3.2: Export JSON
    print("✅ Test 3.2: Export JSON")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_service = ExportService(export_dir=tmpdir)
        
        test_data = {
            'verdict': 'FAKE',
            'confidence': 0.92,
            'probabilities': {'FAKE': 0.92, 'REAL': 0.08},
            'model_used': 'standard'
        }
        
        filepath = export_service.export_json(test_data, 'test_export')
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert 'prediction' in data
        assert data['prediction']['verdict'] == 'FAKE'
    passed += 1
    
    # Test 3.3: Export method
    print("✅ Test 3.3: Export method (JSON format)")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_service = ExportService(export_dir=tmpdir)
        filepath = export_service.export(test_data, format='json')
        assert os.path.exists(filepath)
    passed += 1
    
    print(f"\n✅ ExportService: 3/3 tests passed")
    
except Exception as e:
    print(f"\n❌ ExportService tests failed: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"ExportService: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 4: API ROUTES
# =============================================================================
print("\n" + "="*60)
print("🌐 TEST GROUP 4: API Routes")
print("="*60)

try:
    from src.app.api.routes import api, allowed_file, get_file_type
    
    # Test 4.1: Import API blueprint
    print("\n✅ Test 4.1: Import API blueprint")
    assert api is not None
    assert api.name == 'api'
    passed += 1
    
    # Test 4.2: allowed_file function
    print("✅ Test 4.2: allowed_file function")
    assert allowed_file('test.jpg') == True
    assert allowed_file('test.png', 'image') == True
    assert allowed_file('test.mp4', 'video') == True
    assert allowed_file('test.exe') == False
    assert allowed_file('test.jpg', 'video') == False
    passed += 1
    
    # Test 4.3: get_file_type function
    print("✅ Test 4.3: get_file_type function")
    assert get_file_type('photo.jpg') == 'image'
    assert get_file_type('video.mp4') == 'video'
    assert get_file_type('document.pdf') == None
    passed += 1
    
    print(f"\n✅ API Routes: 3/3 tests passed")
    
except Exception as e:
    print(f"\n❌ API Routes tests failed: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"API Routes: {e}")
    failed += 1

# =============================================================================
# TEST GROUP 5: APP FACTORY
# =============================================================================
print("\n" + "="*60)
print("🏭 TEST GROUP 5: App Factory")
print("="*60)

try:
    from src.app.app_v2 import create_app
    
    # Test 5.1: Create app
    print("\n✅ Test 5.1: Create app")
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'
    })
    assert app is not None
    passed += 1
    
    # Test 5.2: App has blueprints registered
    print("✅ Test 5.2: Blueprints registered")
    assert 'api' in app.blueprints
    assert 'export_api' in app.blueprints
    passed += 1
    
    # Test 5.3: Test client
    print("✅ Test 5.3: Test client works")
    with app.test_client() as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
    passed += 1
    
    # Test 5.4: Models endpoint
    print("✅ Test 5.4: Models endpoint")
    with app.test_client() as client:
        response = client.get('/api/models')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        assert len(data['models']) == 3
    passed += 1
    
    # Test 5.5: Supported formats endpoint
    print("✅ Test 5.5: Supported formats endpoint")
    with app.test_client() as client:
        response = client.get('/api/supported-formats')
        assert response.status_code == 200
        data = response.get_json()
        assert 'images' in data['formats']
        assert 'videos' in data['formats']
    passed += 1
    
    print(f"\n✅ App Factory: 5/5 tests passed")
    
except Exception as e:
    print(f"\n❌ App Factory tests failed: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"App Factory: {e}")
    failed += 1

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("📊 PHASE 2 TEST SUMMARY")
print("="*80)

print(f"\n✅ Passed: {passed}")
print(f"❌ Failed: {len(errors)}")

if errors:
    print("\n❌ Errors:")
    for err in errors:
        print(f"   - {err}")
else:
    print("\n🎉 ALL PHASE 2 TESTS PASSED!")

print("\n" + "="*80)

sys.exit(0 if len(errors) == 0 else 1)
