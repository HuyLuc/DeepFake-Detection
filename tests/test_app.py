"""
Unit tests cho src/app/main_app.py
"""

import unittest
import os
import tempfile
from io import BytesIO
from PIL import Image

# Import các hàm cần test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.app.main_app import validate_video_file


class TestApp(unittest.TestCase):
    """Test cases cho Flask app functions"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_file(self, filename: str, size_mb: float = 1.0) -> object:
        """Tạo mock file object cho testing"""
        class MockFile:
            def __init__(self, name, size_bytes):
                self.filename = name
                self._size = size_bytes
                self._pos = 0
            
            def tell(self):
                return self._size
            
            def seek(self, pos, whence=0):
                if whence == 0:
                    self._pos = pos
                elif whence == 1:
                    self._pos += pos
                elif whence == 2:
                    self._pos = self._size + pos
                return self._pos
            
            def read(self):
                return b'fake video content'
        
        size_bytes = int(size_mb * 1024 * 1024)
        return MockFile(filename, size_bytes)
    
    def test_validate_video_file_valid(self):
        """Test validate_video_file với file hợp lệ"""
        file = self.create_mock_file('test.mp4', size_mb=10.0)
        
        is_valid, error_msg = validate_video_file(file)
        
        self.assertTrue(is_valid, "Valid file should pass validation")
        self.assertIsNone(error_msg, "No error message for valid file")
    
    def test_validate_video_file_invalid_extension(self):
        """Test validate_video_file với extension không hợp lệ"""
        file = self.create_mock_file('test.txt', size_mb=10.0)
        
        is_valid, error_msg = validate_video_file(file)
        
        self.assertFalse(is_valid, "Invalid extension should fail")
        self.assertIsNotNone(error_msg, "Should have error message")
        self.assertIn('định dạng', error_msg.lower() or '')
    
    def test_validate_video_file_too_large(self):
        """Test validate_video_file với file quá lớn"""
        # Tạo file > 500MB (MAX_VIDEO_SIZE_MB)
        file = self.create_mock_file('test.mp4', size_mb=600.0)
        
        is_valid, error_msg = validate_video_file(file)
        
        self.assertFalse(is_valid, "File too large should fail")
        self.assertIsNotNone(error_msg, "Should have error message")
    
    def test_validate_video_file_no_filename(self):
        """Test validate_video_file với file không có tên"""
        class MockFileNoName:
            filename = None
            def tell(self): return 0
            def seek(self, *args): return 0
            def read(self): return b''
        
        file = MockFileNoName()
        
        is_valid, error_msg = validate_video_file(file)
        
        self.assertFalse(is_valid, "File without filename should fail")
        self.assertIsNotNone(error_msg)


if __name__ == '__main__':
    unittest.main()

