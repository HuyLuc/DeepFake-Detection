# tests/run_phase1_tests.py
"""
Master test runner cho Phase 1 components
Chạy tất cả tests và tạo summary report
"""

import subprocess
import sys
import os

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_tests():
    """Run all Phase 1 tests"""
    
    print("="*80)
    print("🧪 PHASE 1 COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    test_files = [
        ('test_temporal_model.py', '📦 TemporalModel (Advanced Architecture)'),
        ('test_model_manager.py', '🤖 ModelManager (Singleton, Predictions)'),
        ('test_file_processor.py', '📸 FileProcessor (Face Extraction, Video)'),
        ('test_prediction_service.py', '🔮 PredictionService (Integration)')
    ]
    
    results = {}
    
    for test_file, description in test_files:
        print(f"\n{'='*80}")
        print(f"Running: {description}")
        print(f"File: {test_file}")
        print(f"{'='*80}\n")
        
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        try:
            # Run pytest với verbose output
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short', '--color=yes'],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                capture_output=False,
                text=True
            )
            
            results[test_file] = (result.returncode == 0)
            
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            results[test_file] = False
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} - {test_file}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Phase 1 is ready!")
    else:
        print(f"⚠️  {total - passed} test suite(s) failed. Please check errors above.")
    
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
