# src/app/services/file_processor.py
"""
FileProcessor: Xử lý ảnh và video
- Extract faces từ images/videos
- Group frames into sequences (cho Advanced model)
"""

import cv2
import mediapipe as mp
from PIL import Image
from typing import List, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Xử lý ảnh và video:
    - Detect và crop faces
    - Extract frames từ video
    - Group frames thành sequences
    """
    
    def __init__(self, skip_frames: int = 5, face_margin: int = 20):
        """
        Args:
            skip_frames: Số frames bỏ qua giữa mỗi lần extract (default: 5)
            face_margin: Margin khi crop face (default: 20px)
        """
        self.skip_frames = skip_frames
        self.face_margin = face_margin
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range, 0 for short range
            min_detection_confidence=0.5
        )
        
        logger.info("✅ FileProcessor initialized")
        logger.info(f"   - Skip frames: {skip_frames}")
        logger.info(f"   - Face margin: {face_margin}px")
    
    def extract_face(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Extract face từ single image
        
        Args:
            image: PIL Image
        
        Returns:
            PIL Image của face được crop, hoặc None nếu không detect được
        """
        try:
            # Convert PIL → numpy array (RGB)
            image_np = np.array(image)
            if image_np.ndim == 2:  # Grayscale → RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Detect face
            results = self.face_detector.process(image_np)
            
            if not results.detections:
                logger.warning("No face detected in image")
                return None
            
            # Get first (most prominent) face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            
            ih, iw = image_np.shape[:2]
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Add margin
            x1 = max(0, x - self.face_margin)
            y1 = max(0, y - self.face_margin)
            x2 = min(iw, x + w + self.face_margin)
            y2 = min(ih, y + h + self.face_margin)
            
            # Crop face
            face_np = image_np[y1:y2, x1:x2]
            
            if face_np.size == 0:
                return None
            
            # Convert back to PIL
            face_image = Image.fromarray(face_np)
            return face_image
            
        except Exception as e:
            logger.error(f"Error extracting face: {e}", exc_info=True)
            return None
    
    def process_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Process single image file
        
        Args:
            image_path: Path to image file
        
        Returns:
            PIL Image của face, hoặc None nếu fail
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Extract face
            face = self.extract_face(image)
            
            if face is None:
                logger.warning(f"Could not extract face from {image_path}")
            
            return face
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            return None
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """
        Process video: Extract frames với faces
        
        Args:
            video_path: Path to video file
            max_frames: Maximum số frames to process (None = all)
        
        Returns:
            {
                'all_frames': List[PIL.Image],  # All frames with faces
                'sequences': List[List[PIL.Image]],  # Grouped into sequences of 5
                'metadata': {
                    'total_frames': int,
                    'processed_frames': int,
                    'frames_with_face': int,
                    'fps': float,
                    'duration': float
                }
            }
        """
        try:
            logger.info(f"📹 Processing video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"   - Total frames: {total_frames}")
            logger.info(f"   - FPS: {fps}")
            logger.info(f"   - Duration: {duration:.2f}s")
            
            # Extract frames
            all_frames = []
            frame_count = 0
            processed_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames
                if frame_count % self.skip_frames != 0:
                    continue
                
                processed_count += 1
                
                # Check max_frames limit
                if max_frames and processed_count > max_frames:
                    break
                
                # Convert BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Extract face
                face = self.extract_face(frame_pil)
                
                if face is not None:
                    all_frames.append(face)
            
            cap.release()
            
            frames_with_face = len(all_frames)
            logger.info(f"   - Processed {processed_count} frames")
            logger.info(f"   - Found faces in {frames_with_face} frames")
            
            # Group into sequences of 5
            sequence_length = 5
            sequences = []
            
            for i in range(0, len(all_frames) - sequence_length + 1, sequence_length):
                sequence = all_frames[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    sequences.append(sequence)
            
            logger.info(f"   - Created {len(sequences)} sequences of {sequence_length} frames")
            
            return {
                'all_frames': all_frames,
                'sequences': sequences,
                'metadata': {
                    'total_frames': total_frames,
                    'processed_frames': processed_count,
                    'frames_with_face': frames_with_face,
                    'fps': fps,
                    'duration': duration
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detector'):
            self.face_detector.close()


# Test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🧪 Testing FileProcessor...")
    print("="*60)
    
    processor = FileProcessor()
    
    # Test với dummy image
    print("\n1. Testing image processing...")
    from PIL import Image
    import numpy as np
    
    # Tạo dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    face = processor.extract_face(dummy_image)
    
    if face:
        print(f"✅ Face extracted: {face.size}")
    else:
        print("⚠️  No face detected (expected for random image)")
    
    print("\n✅ FileProcessor test completed!")
