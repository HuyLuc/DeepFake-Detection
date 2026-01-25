"""
Unit tests cho src/architectures/standard/dataset.py
"""

import unittest
import os
import tempfile
import shutil
from PIL import Image
import numpy as np

# Import các hàm cần test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.architectures.standard.dataset import DeepfakeDataset


class TestDeepfakeDataset(unittest.TestCase):
    """Test cases cho DeepfakeDataset"""
    
    def setUp(self):
        """Setup test fixtures - tạo cấu trúc thư mục test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Tạo cấu trúc: temp_dir/REAL/video1/frame_*.png
        #                temp_dir/FAKE/video2/frame_*.png
        self.real_dir = os.path.join(self.temp_dir, 'REAL', 'video1')
        self.fake_dir = os.path.join(self.temp_dir, 'FAKE', 'video2')
        
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        
        # Tạo một số ảnh test
        for i in range(3):
            # Tạo ảnh RGB đơn giản
            img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
            img.save(os.path.join(self.real_dir, f'frame_{i}.png'))
            img.save(os.path.join(self.fake_dir, f'frame_{i}.png'))
    
    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_initialization(self):
        """Test khởi tạo dataset"""
        dataset = DeepfakeDataset(self.temp_dir)
        
        self.assertEqual(len(dataset), 6, "Should have 6 images (3 REAL + 3 FAKE)")
        self.assertEqual(len(dataset.classes), 2, "Should have 2 classes")
        self.assertIn('FAKE', dataset.classes)
        self.assertIn('REAL', dataset.classes)
    
    def test_dataset_getitem(self):
        """Test __getitem__ method"""
        dataset = DeepfakeDataset(self.temp_dir)
        
        # Test lấy item đầu tiên
        image, label = dataset[0]
        
        self.assertIsNotNone(image, "Image should not be None")
        self.assertIsInstance(label, int, "Label should be integer")
        self.assertIn(label, [0, 1], "Label should be 0 or 1")
    
    def test_dataset_len(self):
        """Test __len__ method"""
        dataset = DeepfakeDataset(self.temp_dir)
        
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.__len__(), 6)
    
    def test_dataset_with_transform(self):
        """Test dataset với transform"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        dataset = DeepfakeDataset(self.temp_dir, transform=transform)
        image, label = dataset[0]
        
        # Image should be a tensor after transform
        import torch
        self.assertIsInstance(image, torch.Tensor, 
                             "Image should be Tensor after transform")
        self.assertEqual(image.shape[0], 3, "Should have 3 channels (RGB)")
    
    def test_dataset_class_to_idx(self):
        """Test class_to_idx mapping"""
        dataset = DeepfakeDataset(self.temp_dir)
        
        self.assertIn('FAKE', dataset.class_to_idx)
        self.assertIn('REAL', dataset.class_to_idx)
        self.assertEqual(len(dataset.class_to_idx), 2)


if __name__ == '__main__':
    unittest.main()

