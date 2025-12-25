"""
Unit tests cho src/utils/utils.py
"""

import unittest
import os
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

# Import các hàm cần test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.utils import save_checkpoint, load_checkpoint, verify_data_structure


class TestUtils(unittest.TestCase):
    """Test cases cho các utility functions"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model = nn.Linear(10, 2)
        self.test_optimizer = torch.optim.Adam(self.test_model.parameters())
        
    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_checkpoint(self):
        """Test save_checkpoint function"""
        state = {
            'epoch': 1,
            'state_dict': self.test_model.state_dict(),
            'optimizer': self.test_optimizer.state_dict(),
            'best_val_acc': 0.95
        }
        
        # Test save checkpoint
        save_checkpoint(state, is_best=False, 
                       filename='test_checkpoint.pth.tar',
                       model_save_dir=self.temp_dir)
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth.tar')
        self.assertTrue(os.path.exists(checkpoint_path), 
                       "Checkpoint file should be created")
    
    def test_save_best_checkpoint(self):
        """Test save_checkpoint với is_best=True"""
        state = {
            'epoch': 1,
            'state_dict': self.test_model.state_dict(),
            'optimizer': self.test_optimizer.state_dict(),
            'best_val_acc': 0.95
        }
        
        save_checkpoint(state, is_best=True,
                       filename='test_checkpoint.pth.tar',
                       best_filename='test_best.pth.tar',
                       model_save_dir=self.temp_dir)
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth.tar')
        best_path = os.path.join(self.temp_dir, 'test_best.pth.tar')
        
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(os.path.exists(best_path), 
                       "Best model file should be created")
    
    def test_load_checkpoint_success(self):
        """Test load_checkpoint với checkpoint hợp lệ"""
        # Tạo checkpoint
        state = {
            'epoch': 5,
            'state_dict': self.test_model.state_dict(),
            'optimizer': self.test_optimizer.state_dict(),
            'best_val_acc': 0.98
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth.tar')
        torch.save(state, checkpoint_path)
        
        # Load checkpoint
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        model, optimizer, start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        self.assertEqual(start_epoch, 6, "Start epoch should be epoch + 1")
        self.assertEqual(best_val_acc, 0.98, "Best val acc should match")
        self.assertIsNotNone(model)
        self.assertIsNotNone(optimizer)
    
    def test_load_checkpoint_not_found(self):
        """Test load_checkpoint khi file không tồn tại"""
        checkpoint_path = os.path.join(self.temp_dir, 'nonexistent.pth.tar')
        
        model, optimizer, start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, self.test_model, self.test_optimizer
        )
        
        self.assertEqual(start_epoch, 0, "Should start from 0 if checkpoint not found")
        self.assertEqual(best_val_acc, 0.0, "Should have 0.0 best val acc if checkpoint not found")
    
    def test_load_checkpoint_invalid(self):
        """Test load_checkpoint với checkpoint không hợp lệ"""
        # Tạo checkpoint không hợp lệ (thiếu state_dict)
        invalid_state = {
            'epoch': 1,
            'best_val_acc': 0.95
            # Thiếu 'state_dict'
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'invalid.pth.tar')
        torch.save(invalid_state, checkpoint_path)
        
        # Load should handle error gracefully
        model, optimizer, start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, self.test_model, self.test_optimizer
        )
        
        # Should return default values on error
        self.assertEqual(start_epoch, 0)
        self.assertEqual(best_val_acc, 0.0)


if __name__ == '__main__':
    unittest.main()

