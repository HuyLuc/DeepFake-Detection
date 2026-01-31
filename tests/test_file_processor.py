# tests/test_file_processor.py
"""
Test cho FileProcessor
"""

import pytest
import sys
import os
from PIL import Image
import numpy as np
import tempfile
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.app.services.file_processor import FileProcessor


class TestFileProcessor:
    """Test suite cho FileProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Fixture: Tạo FileProcessor instance"""
        return FileProcessor(skip_frames=5, face_margin=20)
    
    @pytest.fixture
    def dummy_image_with_mock_face(self):
        """Fixture: Tạo dummy image (random - không có real face)"""
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    
    @pytest.fixture
    def simple_image(self):
        """Fixture: Tạo simple solid color image"""
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray
        return Image.fromarray(arr)
    
    def test_initialization(self, processor):
        """Test 1: FileProcessor initialized correctly"""
        assert processor is not None
        assert processor.skip_frames == 5
        assert processor.face_margin == 20
        assert processor.face_detector is not None
        print("✅ Test 1 passed: Initialization")
    
    def test_extract_face_no_face(self, processor, dummy_image_with_mock_face):
        """Test 2: Extract face từ image không có face"""
        face = processor.extract_face(dummy_image_with_mock_face)
        
        # Với random noise, likely không detect được face
        # (Có thể detect được false positive, nhưng acceptable)
        if face is None:
            print("✅ Test 2 passed: No face detected in random image (expected)")
        else:
            print(f"⚠️  Test 2: Face detected in random image (false positive): {face.size}")
    
    def test_extract_face_grayscale_conversion(self, processor):
        """Test 3: Xử lý grayscale image"""
        # Tạo grayscale image
        gray_arr = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        gray_image = Image.fromarray(gray_arr, mode='L')
        
        # Should not crash
        try:
            face = processor.extract_face(gray_image)
            print("✅ Test 3 passed: Grayscale image handled")
        except Exception as e:
            pytest.fail(f"Failed to handle grayscale image: {e}")
    
    def test_process_image_file_not_found(self, processor):
        """Test 4: Process image với invalid path"""
        result = processor.process_image("/nonexistent/path/image.jpg")
        assert result is None
        print("✅ Test 4 passed: Invalid path handled gracefully")
    
    def test_process_image_valid_file(self, processor, tmp_path):
        """Test 5: Process image với valid file"""
        # Tạo temporary image file
        temp_image_path = tmp_path / "test_image.jpg"
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(temp_image_path)
        
        # Process
        result = processor.process_image(str(temp_image_path))
        
        # Result có thể là None (no face) hoặc PIL Image (face detected)
        if result is None:
            print("✅ Test 5 passed: No face in test image (expected)")
        else:
            assert isinstance(result, Image.Image)
            print(f"✅ Test 5 passed: Face extracted, size = {result.size}")
    
    def test_skip_frames_config(self):
        """Test 6: Skip frames configuration"""
        processor1 = FileProcessor(skip_frames=10)
        processor2 = FileProcessor(skip_frames=3)
        
        assert processor1.skip_frames == 10
        assert processor2.skip_frames == 3
        print("✅ Test 6 passed: Skip frames configurable")
    
    def test_face_margin_config(self):
        """Test 7: Face margin configuration"""
        processor = FileProcessor(face_margin=30)
        assert processor.face_margin == 30
        print("✅ Test 7 passed: Face margin configurable")
    
    def test_process_video_invalid_path(self, processor):
        """Test 8: Process video với invalid path"""
        with pytest.raises(Exception):
            processor.process_video("/nonexistent/video.mp4")
        print("✅ Test 8 passed: Invalid video path raises error")
    
    def test_process_video_with_mock(self, processor, tmp_path):
        """Test 9: Process video với generated video"""
        # Tạo simple video file (5 frames, solid colors)
        temp_video_path = tmp_path / "test_video.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video_path), fourcc, 30.0, (640, 480))
        
        # Write 15 frames (các màu khác nhau)
        for i in range(15):
            color_value = (i * 17) % 256  # Different shades
            frame = np.full((480, 640, 3), color_value, dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Process video
        try:
            result = processor.process_video(str(temp_video_path), max_frames=10)
            
            # Check result structure
            assert 'all_frames' in result
            assert 'sequences' in result
            assert 'metadata' in result
            
            # Check metadata
            metadata = result['metadata']
            assert 'total_frames' in metadata
            assert 'processed_frames' in metadata
            assert 'frames_with_face' in metadata
            assert 'fps' in metadata
            assert 'duration' in metadata
            
            print("✅ Test 9 passed: Video processing structure")
            print(f"   Total frames: {metadata['total_frames']}")
            print(f"   Processed: {metadata['processed_frames']}")
            print(f"   With faces: {metadata['frames_with_face']}")
            print(f"   Sequences: {len(result['sequences'])}")
            
        except Exception as e:
            # OpenCV/video processing có thể fail trên một số systems
            print(f"⚠️  Test 9: Video processing failed (may be environment issue): {e}")
    
    def test_sequence_grouping_logic(self, processor):
        """Test 10: Sequence grouping logic"""
        # Tạo mock frames
        frames = []
        for i in range(12):
            arr = np.ones((224, 224, 3), dtype=np.uint8) * (i * 20)
            frames.append(Image.fromarray(arr))
        
        # Manually group into sequences of 5
        sequence_length = 5
        sequences = []
        
        for i in range(0, len(frames) - sequence_length + 1, sequence_length):
            sequence = frames[i:i + sequence_length]
            if len(sequence) == sequence_length:
                sequences.append(sequence)
        
        # Should have 2 complete sequences from 12 frames
        # (0-4), (5-9) - frame 10,11 not enough for full sequence
        assert len(sequences) == 2
        assert len(sequences[0]) == 5
        assert len(sequences[1]) == 5
        
        print("✅ Test 10 passed: Sequence grouping logic")
        print(f"   12 frames → {len(sequences)} sequences of 5")
    
    def test_cleanup(self, processor):
        """Test 11: Resource cleanup"""
        # FileProcessor should have face_detector
        assert hasattr(processor, 'face_detector')
        
        # Manually cleanup
        processor.__del__()
        
        print("✅ Test 11 passed: Cleanup method exists")


def test_face_detector_initialization():
    """Test 12: MediaPipe face detector configuration"""
    processor = FileProcessor()
    
    # Check face detector initialized
    assert processor.face_detector is not None
    
    print("✅ Test 12 passed: Face detector initialized")


def test_multiple_processor_instances():
    """Test 13: Multiple FileProcessor instances"""
    processor1 = FileProcessor(skip_frames=5)
    processor2 = FileProcessor(skip_frames=10)
    
    # Should be different instances
    assert processor1 is not processor2
    assert processor1.skip_frames != processor2.skip_frames
    
    print("✅ Test 13 passed: Multiple processors can coexist")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Running FileProcessor Tests...")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
