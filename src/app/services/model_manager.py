# src/app/services/model_manager.py
"""
ModelManager: Quản lý cả 2 models (Standard + Advanced)
Singleton pattern - Load models 1 lần duy nhất khi app khởi động
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional
import os
import logging

# Import TemporalModel
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from src.architectures.advanced.model import TemporalModel

logger = logging.getLogger(__name__)

# ============================================================================
# HYBRID VERDICT SYSTEM CONSTANTS
# ============================================================================
# Thresholds for hybrid verdict (FAKE/SUSPICIOUS/REAL)
FAKE_RATIO_THRESHOLD_HIGH = 0.50      # >= 50% FAKE -> FAKE
FAKE_RATIO_THRESHOLD_MEDIUM = 0.20    # >= 20% FAKE -> SUSPICIOUS
FAKE_RATIO_THRESHOLD_LOW = 0.10       # >= 10% with high confidence -> careful
MAX_FAKE_CONFIDENCE_CRITICAL = 0.90   # Any frame >= 90% FAKE confidence is critical
MAX_FAKE_CONFIDENCE_WARNING = 0.85    # Any frame >= 85% FAKE confidence is warning


def compute_hybrid_verdict(timeline: List[Dict], fake_ratio: float) -> Tuple[str, float, str]:
    """
    Compute video verdict using Hybrid approach:
    - Combines FAKE ratio AND maximum FAKE confidence
    - Returns verdict with explanation
    
    Args:
        timeline: List of frame predictions [{frame, verdict, confidence}, ...]
        fake_ratio: Ratio of FAKE frames (0.0 - 1.0)
    
    Returns:
        (verdict, confidence, explanation)
        - verdict: 'FAKE', 'SUSPICIOUS', or 'REAL'
        - confidence: Overall confidence score
        - explanation: Human-readable explanation
    """
    if not timeline:
        return 'REAL', 0.0, 'Không có frames để phân tích'
    
    # Calculate max FAKE confidence from timeline
    fake_confidences = [
        item['confidence'] for item in timeline 
        if item['verdict'] == 'FAKE'
    ]
    max_fake_confidence = max(fake_confidences) if fake_confidences else 0.0
    
    # Calculate average confidence for verdict
    all_confidences = [item['confidence'] for item in timeline]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # =========== HYBRID LOGIC ===========
    
    # Case 1: Clear FAKE (>= 50% FAKE ratio)
    if fake_ratio >= FAKE_RATIO_THRESHOLD_HIGH:
        explanation = f'Rõ ràng FAKE: {fake_ratio*100:.1f}% frames là giả mạo'
        return 'FAKE', avg_confidence, explanation
    
    # Case 2: Partial deepfake detected (>= 10% FAKE with critical confidence >= 90%)
    if fake_ratio >= FAKE_RATIO_THRESHOLD_LOW and max_fake_confidence >= MAX_FAKE_CONFIDENCE_CRITICAL:
        explanation = f'Phát hiện deepfake một phần: {fake_ratio*100:.1f}% frames FAKE với độ tin cậy cao ({max_fake_confidence*100:.1f}%)'
        return 'FAKE', max_fake_confidence, explanation
    
    # Case 3: SUSPICIOUS (20-50% FAKE ratio)
    if fake_ratio >= FAKE_RATIO_THRESHOLD_MEDIUM:
        explanation = f'Nghi ngờ: {fake_ratio*100:.1f}% frames có dấu hiệu giả mạo'
        return 'SUSPICIOUS', avg_confidence, explanation
    
    # Case 4: SUSPICIOUS (any frame with >= 85% FAKE confidence)
    if max_fake_confidence >= MAX_FAKE_CONFIDENCE_WARNING:
        explanation = f'Cảnh báo: Có frame với độ FAKE cao ({max_fake_confidence*100:.1f}%)'
        return 'SUSPICIOUS', max_fake_confidence, explanation
    
    # Case 5: REAL (no significant FAKE signals)
    real_ratio = 1 - fake_ratio
    explanation = f'Video thật: {real_ratio*100:.1f}% frames là thật'
    return 'REAL', avg_confidence, explanation


class ModelManager:
    """
    Singleton class để quản lý cả 2 deepfake detection models
    
    Models:
        - Standard: EfficientNet-B4 (single frame)
        - Advanced: EfficientNet-B4 + LSTM (temporal - sequence of frames)
    
    Usage:
        manager = ModelManager()
        result = manager.predict_image_standard(pil_image)
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern - chỉ tạo 1 instance duy nhất"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ModelManager - chỉ chạy 1 lần"""
        if ModelManager._initialized:
            return
        
        logger.info("="*60)
        logger.info("🤖 Initializing ModelManager...")
        logger.info("="*60)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"📍 Device: {self.device}")
        
        # Model paths
        self.standard_model_path = 'saved_models/standard/best_model.pth'
        self.advanced_model_path = 'saved_models/advanced/best_temporal_model.pth'
        
        # Models
        self.standard_model = None
        self.advanced_model = None
        
        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Config
        self.sequence_length = 5  # Số frames cho Advanced model
        self.class_names = ['FAKE', 'REAL']
        
        # Load models
        self.load_models()
        
        ModelManager._initialized = True
        logger.info("✅ ModelManager initialized successfully!")
        logger.info("="*60)
    
    def load_models(self):
        """Load cả 2 models vào memory"""
        try:
            # 1. Load Standard Model
            logger.info("\n📦 Loading Standard Model...")
            self.standard_model = timm.create_model(
                'efficientnet_b4',
                pretrained=False,
                num_classes=2
            )
            
            if os.path.exists(self.standard_model_path):
                state_dict = torch.load(self.standard_model_path, map_location=self.device)
                self.standard_model.load_state_dict(state_dict)
                logger.info(f"✅ Standard Model loaded from {self.standard_model_path}")
            else:
                logger.warning(f"⚠️  Standard model not found at {self.standard_model_path}")
                logger.warning("   Using untrained model (for testing only)")
            
            self.standard_model.to(self.device)
            self.standard_model.eval()
            
            # 2. Load Advanced Model
            logger.info("\n📦 Loading Advanced Model...")
            self.advanced_model = TemporalModel(
                num_classes=2,
                pretrained=False,
                sequence_length=self.sequence_length
            )
            
            if os.path.exists(self.advanced_model_path):
                state_dict = torch.load(self.advanced_model_path, map_location=self.device)
                self.advanced_model.load_state_dict(state_dict)
                logger.info(f"✅ Advanced Model loaded from {self.advanced_model_path}")
            else:
                logger.warning(f"⚠️  Advanced model not found at {self.advanced_model_path}")
                logger.warning("   Using untrained model (for testing only)")
            
            self.advanced_model.to(self.device)
            self.advanced_model.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            raise
    
    def get_model(self, model_type: str = 'standard') -> torch.nn.Module:
        """
        Get model reference for external use (e.g., Grad-CAM)
        
        Args:
            model_type: 'standard' or 'advanced'
            
        Returns:
            PyTorch model instance
        """
        if model_type == 'standard':
            return self.standard_model
        elif model_type == 'advanced':
            return self.advanced_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_device(self) -> torch.device:
        """Get current device (cpu/cuda)"""
        return self.device
    
    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare single image for inference
        
        Args:
            image: PIL Image
        
        Returns:
            Tensor of shape (1, 3, 224, 224)
        """
        return self.image_transform(image).unsqueeze(0).to(self.device)
    
    def _prepare_sequence(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Prepare sequence of images for Advanced model
        
        Args:
            images: List of PIL Images (length = sequence_length)
        
        Returns:
            Tensor of shape (1, sequence_length, 3, 224, 224)
        """
        tensors = [self.image_transform(img) for img in images]
        sequence = torch.stack(tensors).unsqueeze(0).to(self.device)  # (1, seq_len, 3, H, W)
        return sequence
    
    def predict_image_standard(self, image: Image.Image) -> Dict:
        """
        Predict single image với Standard model
        
        Args:
            image: PIL Image của face đã được crop
        
        Returns:
            {
                'verdict': 'FAKE' or 'REAL',
                'confidence': float (0-1),
                'probabilities': {'FAKE': float, 'REAL': float},
                'model': 'standard'
            }
        """
        try:
            # Prepare input
            image_tensor = self._prepare_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.standard_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get predictions
            confidence, pred_idx = torch.max(probabilities, 0)
            verdict = self.class_names[pred_idx.item()]
            
            result = {
                'verdict': verdict,
                'confidence': confidence.item(),
                'probabilities': {
                    'FAKE': probabilities[0].item(),
                    'REAL': probabilities[1].item()
                },
                'model': 'standard'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_image_standard: {e}", exc_info=True)
            raise
    
    def predict_image_advanced(self, image: Image.Image) -> Dict:
        """
        Predict single image với Advanced model
        Treat single image như 1 sequence với cùng frame repeated
        
        Args:
            image: PIL Image của face đã được crop
        
        Returns:
            {
                'verdict': 'FAKE' or 'REAL',
                'confidence': float,
                'probabilities': {'FAKE': float, 'REAL': float},
                'model': 'advanced'
            }
        """
        try:
            # Repeat image để tạo sequence
            images = [image] * self.sequence_length
            sequence_tensor = self._prepare_sequence(images)
            
            # Inference
            with torch.no_grad():
                outputs = self.advanced_model(sequence_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get predictions
            confidence, pred_idx = torch.max(probabilities, 0)
            verdict = self.class_names[pred_idx.item()]
            
            result = {
                'verdict': verdict,
                'confidence': confidence.item(),
                'probabilities': {
                    'FAKE': probabilities[0].item(),
                    'REAL': probabilities[1].item()
                },
                'model': 'advanced'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_image_advanced: {e}", exc_info=True)
            raise
    
    def predict_video_standard(self, frames: List[Image.Image]) -> Dict:
        """
        Predict video với Standard model (frame-by-frame)
        
        Args:
            frames: List of PIL Images (faces đã được crop)
        
        Returns:
            {
                'verdict': 'FAKE' or 'REAL',
                'confidence': float,
                'timeline': [{'frame': int, 'verdict': str, 'confidence': float}, ...],
                'stats': {
                    'total_frames': int,
                    'fake_count': int,
                    'real_count': int,
                    'fake_ratio': float
                },
                'model': 'standard'
            }
        """
        try:
            timeline = []
            fake_count = 0
            real_count = 0
            
            for idx, frame in enumerate(frames):
                result = self.predict_image_standard(frame)
                
                timeline.append({
                    'frame': idx + 1,
                    'verdict': result['verdict'],
                    'confidence': result['confidence']
                })
                
                if result['verdict'] == 'FAKE':
                    fake_count += 1
                else:
                    real_count += 1
            
            # Aggregate results
            total_frames = len(frames)
            fake_ratio = fake_count / total_frames if total_frames > 0 else 0
            
            # HYBRID VERDICT SYSTEM
            final_verdict, verdict_confidence, verdict_explanation = compute_hybrid_verdict(
                timeline, fake_ratio
            )
            
            logger.info(f"📊 Video verdict: {final_verdict} ({verdict_confidence*100:.1f}%) - {verdict_explanation}")
            
            return {
                'verdict': final_verdict,
                'confidence': verdict_confidence,
                'verdict_explanation': verdict_explanation,
                'timeline': timeline,
                'stats': {
                    'total_frames': total_frames,
                    'fake_count': fake_count,
                    'real_count': real_count,
                    'fake_ratio': fake_ratio
                },
                'model': 'standard'
            }
            
        except Exception as e:
            logger.error(f"Error in predict_video_standard: {e}", exc_info=True)
            raise
    
    def predict_video_advanced(self, frames: List[Image.Image]) -> Dict:
        """
        Predict video với Advanced model (sequences)
        Group frames into sequences of 5
        
        Args:
            frames: List of PIL Images (faces)
        
        Returns:
            Same format as predict_video_standard
        """
        try:
            timeline = []
            fake_count = 0
            real_count = 0
            
            # Group frames into sequences
            num_sequences = len(frames) // self.sequence_length
            
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * self.sequence_length
                end_idx = start_idx + self.sequence_length
                sequence = frames[start_idx:end_idx]
                
                # Prepare and predict
                sequence_tensor = self._prepare_sequence(sequence)
                
                with torch.no_grad():
                    outputs = self.advanced_model(sequence_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                confidence, pred_idx = torch.max(probabilities, 0)
                verdict = self.class_names[pred_idx.item()]
                
                # Record result for middle frame of sequence
                middle_frame = start_idx + self.sequence_length // 2
                timeline.append({
                    'frame': middle_frame + 1,
                    'verdict': verdict,
                    'confidence': confidence.item()
                })
                
                if verdict == 'FAKE':
                    fake_count += 1
                else:
                    real_count += 1
            
            # Aggregate
            total_sequences = num_sequences
            fake_ratio = fake_count / total_sequences if total_sequences > 0 else 0
            
            # HYBRID VERDICT SYSTEM
            final_verdict, verdict_confidence, verdict_explanation = compute_hybrid_verdict(
                timeline, fake_ratio
            )
            
            logger.info(f"📊 Video verdict (advanced): {final_verdict} ({verdict_confidence*100:.1f}%) - {verdict_explanation}")
            
            return {
                'verdict': final_verdict,
                'confidence': verdict_confidence,
                'verdict_explanation': verdict_explanation,
                'timeline': timeline,
                'stats': {
                    'total_frames': len(frames),
                    'sequences_analyzed': total_sequences,
                    'fake_count': fake_count,
                    'real_count': real_count,
                    'fake_ratio': fake_ratio
                },
                'model': 'advanced'
            }
            
        except Exception as e:
            logger.error(f"Error in predict_video_advanced: {e}", exc_info=True)
            raise
    
    def predict_ensemble(self, input_data, is_video=False) -> Dict:
        """
        Ensemble prediction: Combine cả 2 models
        
        Args:
            input_data: PIL Image (if is_video=False) or List[PIL.Image] (if is_video=True)
            is_video: Boolean flag
        
        Returns:
            {
                'verdict': 'FAKE' or 'REAL',
                'confidence': float,
                'models_comparison': {
                    'standard': {...},
                    'advanced': {...}
                },
                'model': 'ensemble'
            }
        """
        try:
            if is_video:
                # Video ensemble
                standard_result = self.predict_video_standard(input_data)
                advanced_result = self.predict_video_advanced(input_data)
            else:
                # Image ensemble
                standard_result = self.predict_image_standard(input_data)
                advanced_result = self.predict_image_advanced(input_data)
            
            # Combine: Average probabilities
            fake_prob_avg = (
                standard_result['probabilities']['FAKE'] +
                advanced_result['probabilities']['FAKE']
            ) / 2
            
            real_prob_avg = (
                standard_result['probabilities']['REAL'] +
                advanced_result['probabilities']['REAL']
            ) / 2
            
            # Final verdict
            final_verdict = 'FAKE' if fake_prob_avg > real_prob_avg else 'REAL'
            final_confidence = max(fake_prob_avg, real_prob_avg)
            
            result = {
                'verdict': final_verdict,
                'confidence': final_confidence,
                'probabilities': {
                    'FAKE': fake_prob_avg,
                    'REAL': real_prob_avg
                },
                'models_comparison': {
                    'standard': standard_result,
                    'advanced': advanced_result
                },
                'model': 'ensemble'
            }
            
            # Include timeline if video
            if is_video and 'timeline' in standard_result:
                result['timeline'] = standard_result['timeline']  # Use standard timeline
                result['stats'] = standard_result['stats']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_ensemble: {e}", exc_info=True)
            raise


# Test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🧪 Testing ModelManager...")
    print("="*60)
    
    # Create manager
    manager = ModelManager()
    
    # Test với dummy image
    from PIL import Image
    import numpy as np
    
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    print("\n1. Testing Standard Model...")
    result = manager.predict_image_standard(dummy_image)
    print(f"Result: {result}")
    
    print("\n2. Testing Advanced Model...")
    result = manager.predict_image_advanced(dummy_image)
    print(f"Result: {result}")
    
    print("\n3. Testing Ensemble...")
    result = manager.predict_ensemble(dummy_image, is_video=False)
    print(f"Result: {result}")
    
    print("\n✅ All tests passed!")
