# src/app/services/prediction_service.py
"""
PredictionService: Orchestrate prediction workflow
Kết hợp ModelManager + FileProcessor
"""

import os
import time
import logging
from typing import Dict, Optional
from PIL import Image

from .model_manager import ModelManager
from .file_processor import FileProcessor
from .explainability_service import get_explainability_service

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Main service để handle prediction requests
    
    Workflow:
        1. Process file (extract faces)
        2. Call appropriate model
        3. Format và return results
    """
    
    def __init__(self):
        """Initialize PredictionService"""
        logger.info("🚀 Initializing PredictionService...")
        
        self.model_manager = ModelManager()
        self.file_processor = FileProcessor(skip_frames=5, face_margin=20)
        
        logger.info("✅ PredictionService ready!")
    
    def predict(
        self,
        file_path: str,
        file_type: str,
        model_choice: str = 'standard',
        options: Optional[Dict] = None
    ) -> Dict:
        """
        Main prediction method
        
        Args:
            file_path: Path to uploaded file
            file_type: 'image' or 'video'
            model_choice: 'standard', 'advanced', or 'ensemble'
            options: Dict with optional settings:
                - show_timeline: bool
                - threshold: float
                - max_frames: int (for video)
        
        Returns:
            Complete result dict
        """
        options = options or {}
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Starting prediction...")
        logger.info(f"   - File: {os.path.basename(file_path)}")
        logger.info(f"   - Type: {file_type}")
        logger.info(f"   - Model: {model_choice}")
        logger.info(f"{'='*60}")
        
        try:
            if file_type == 'image':
                result = self._predict_image(file_path, model_choice, options)
            elif file_type == 'video':
                result = self._predict_video(file_path, model_choice, options)
            else:
                raise ValueError(f"Invalid file_type: {file_type}")
            
            # Add processing time
            processing_time = time.time() - start_time
            result['processing_time'] = round(processing_time, 2)
            
            # Log result
            if result.get('success', False):
                logger.info(f"\n✅ Prediction completed in {processing_time:.2f}s")
                logger.info(f"   Verdict: {result['verdict']} ({result['confidence']*100:.2f}%)")
            else:
                logger.warning(f"\n⚠️ Prediction failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            raise
    
    def _predict_image(self, image_path: str, model_choice: str, options: Dict) -> Dict:
        """
        Predict single image
        
        Returns:
            {
                'success': bool,
                'verdict': 'FAKE' or 'REAL',
                'confidence': float,
                'probabilities': {'FAKE': float, 'REAL': float},
                'model_used': str,
                'processing_time': float,
                'details': {
                    'face_detected': bool,
                    'face_size': tuple or None
                }
            }
        """
        logger.info("📸 Processing image...")
        
        # 1. Extract face
        face = self.file_processor.process_image(image_path)
        
        if face is None:
            return {
                'success': False,
                'error': 'No face detected in image',
                'details': {'face_detected': False}
            }
        
        logger.info(f"   ✅ Face detected: {face.size}")
        
        # 2. Predict
        if model_choice == 'standard':
            prediction = self.model_manager.predict_image_standard(face)
        elif model_choice == 'advanced':
            prediction = self.model_manager.predict_image_advanced(face)
        elif model_choice == 'ensemble':
            prediction = self.model_manager.predict_ensemble(face, is_video=False)
        else:
            raise ValueError(f"Invalid model_choice: {model_choice}")
        
        # 3. Format result
        result = {
            'success': True,
            'verdict': prediction['verdict'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'model_used': prediction['model'],
            'details': {
                'face_detected': True,
                'face_size': face.size
            }
        }
        
        # Add models_comparison if ensemble
        if 'models_comparison' in prediction:
            result['models_comparison'] = prediction['models_comparison']
        
        # 4. Generate Grad-CAM heatmap if requested
        if options.get('generate_heatmap', False):
            try:
                explainability_service = get_explainability_service()
                if explainability_service.enabled:
                    # Use standard model for heatmap (simpler, faster)
                    model_for_heatmap = self.model_manager.get_model('standard')
                    device = str(self.model_manager.get_device())
                    
                    explanation = explainability_service.generate_explanation(
                        image=face,
                        model=model_for_heatmap,
                        prediction_result=result,
                        device=device,
                        save_to_disk=True,
                        prediction_id=options.get('prediction_id')
                    )
                    
                    if explanation.get('success'):
                        result['heatmap'] = {
                            'image_base64': explanation['heatmap_base64'],
                            'explanation': explanation['explanation'],
                            'path': explanation.get('heatmap_path')
                        }
                        logger.info("🔥 Heatmap generated successfully")
                    else:
                        logger.warning(f"⚠️ Heatmap generation failed: {explanation.get('error')}")
                else:
                    logger.warning("⚠️ Heatmap requested but Grad-CAM not available")
            except Exception as e:
                logger.error(f"❌ Error generating heatmap: {e}", exc_info=True)
                # Don't fail the whole prediction, just skip heatmap
        
        return result
    
    def _predict_video(self, video_path: str, model_choice: str, options: Dict) -> Dict:
        """
        Predict video
        
        Returns:
            {
                'success': bool,
                'verdict': 'FAKE' or 'REAL',
                'confidence': float,
                'model_used': str,
                'processing_time': float,
                'timeline': [{'frame': int, 'verdict': str, 'confidence': float}, ...],
                'stats': {
                    'total_frames': int,
                    'frames_analyzed': int,
                    'fake_count': int,
                    'real_count': int,
                    'fake_ratio': float
                },
                'details': {
                    'fps': float,
                    'duration': float,
                    'frames_with_face': int
                }
            }
        """
        logger.info("🎬 Processing video...")
        
        # 1. Process video - extract frames
        max_frames = options.get('max_frames', None)
        video_data = self.file_processor.process_video(video_path, max_frames=max_frames)
        
        all_frames = video_data['all_frames']
        metadata = video_data['metadata']
        
        if len(all_frames) == 0:
            return {
                'success': False,
                'error': 'No faces detected in video',
                'details': metadata
            }
        
        logger.info(f"   ✅ Extracted {len(all_frames)} frames with faces")
        
        # 2. Predict
        if model_choice == 'standard':
            prediction = self.model_manager.predict_video_standard(all_frames)
        elif model_choice == 'advanced':
            prediction = self.model_manager.predict_video_advanced(all_frames)
        elif model_choice == 'ensemble':
            prediction = self.model_manager.predict_ensemble(all_frames, is_video=True)
        else:
            raise ValueError(f"Invalid model_choice: {model_choice}")
        
        # 3. Format result
        result = {
            'success': True,
            'verdict': prediction['verdict'],
            'confidence': prediction['confidence'],
            'model_used': prediction['model'],
            'timeline': prediction.get('timeline', []),
            'stats': prediction.get('stats', {}),
            'details': {
                'fps': metadata['fps'],
                'duration': metadata['duration'],
                'total_frames': metadata['total_frames'],
                'processed_frames': metadata['processed_frames'],
                'frames_with_face': metadata['frames_with_face']
            }
        }
        
        # Add probabilities if available
        if 'probabilities' in prediction:
            result['probabilities'] = prediction['probabilities']
        
        # Add models_comparison if ensemble
        if 'models_comparison' in prediction:
            result['models_comparison'] = prediction['models_comparison']
        
        # 4. Generate Key Frame Heatmap (frame with highest FAKE probability)
        if options.get('generate_heatmap', False) and len(all_frames) > 0:
            try:
                timeline = prediction.get('timeline', [])
                if timeline:
                    # Find frame with highest FAKE indication
                    # For FAKE verdict, use highest confidence; for REAL, find lowest REAL confidence
                    key_frame_idx = 0
                    max_fake_score = 0
                    
                    for i, item in enumerate(timeline):
                        if item['verdict'] == 'FAKE':
                            # FAKE with high confidence = suspicious
                            fake_score = item['confidence']
                        else:
                            # REAL with low confidence = also suspicious
                            fake_score = 1 - item['confidence']
                        
                        if fake_score > max_fake_score:
                            max_fake_score = fake_score
                            # Map timeline index to frame index
                            key_frame_idx = min(item['frame'] - 1, len(all_frames) - 1)
                    
                    key_frame = all_frames[max(0, key_frame_idx)]
                    key_frame_number = timeline[0]['frame'] if timeline else 1  # Fallback
                    
                    # Find actual frame number from timeline
                    for item in timeline:
                        if item['frame'] - 1 == key_frame_idx:
                            key_frame_number = item['frame']
                            break
                    
                    logger.info(f"🔑 Key frame selected: Frame {key_frame_number} (index {key_frame_idx}, fake_score: {max_fake_score:.2f})")
                    
                    # Generate heatmap for key frame
                    explainability_service = get_explainability_service()
                    if explainability_service.enabled:
                        model_for_heatmap = self.model_manager.get_model('standard')
                        device = str(self.model_manager.get_device())
                        
                        explanation = explainability_service.generate_explanation(
                            image=key_frame,
                            model=model_for_heatmap,
                            prediction_result={'verdict': prediction['verdict']},
                            device=device,
                            save_to_disk=True,
                            prediction_id=options.get('prediction_id')
                        )
                        
                        if explanation.get('success'):
                            result['key_frame_heatmap'] = {
                                'frame_number': key_frame_number,
                                'image_base64': explanation['heatmap_base64'],
                                'explanation': explanation['explanation'],
                                'fake_score': round(max_fake_score, 3)
                            }
                            logger.info(f"🔥 Key frame heatmap generated for frame {key_frame_number}")
                        else:
                            logger.warning(f"⚠️ Key frame heatmap failed: {explanation.get('error')}")
                    else:
                        logger.warning("⚠️ Grad-CAM not available for key frame")
            except Exception as e:
                logger.error(f"❌ Error generating key frame heatmap: {e}", exc_info=True)
        
        return result


# Test
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("🧪 Testing PredictionService...")
    print("="*60)
    
    service = PredictionService()
    
    print("\n✅ PredictionService initialized successfully!")
    print("   Ready to handle predictions!")
