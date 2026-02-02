# src/app/services/explainability_service.py
"""
ExplainabilityService: Tạo Grad-CAM heatmaps để giải thích quyết định của model
Sử dụng pytorch-grad-cam library

🔬 Grad-CAM (Gradient-weighted Class Activation Mapping):
   - Visualize vùng nào trên ảnh ảnh hưởng đến quyết định của model
   - Giúp user hiểu TẠI SAO model cho kết quả FAKE/REAL
"""

import os
import io
import base64
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("⚠️ pytorch-grad-cam not installed. Run: pip install grad-cam")

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Service để tạo visual explanations (Grad-CAM heatmaps) cho predictions
    
    Features:
        - Generate Grad-CAM heatmap từ EfficientNet model
        - Overlay heatmap lên ảnh gốc
        - Lưu heatmap và trả về URL/base64
    """
    
    def __init__(self, heatmap_dir: str = None):
        """
        Initialize ExplainabilityService
        
        Args:
            heatmap_dir: Directory để lưu heatmaps
        """
        if not GRADCAM_AVAILABLE:
            logger.warning("⚠️ Grad-CAM library not available. Heatmap feature disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Setup heatmap storage directory
        if heatmap_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            heatmap_dir = os.path.join(project_root, 'data', 'heatmaps')
        
        self.heatmap_dir = heatmap_dir
        os.makedirs(self.heatmap_dir, exist_ok=True)
        
        # Image preprocessing (same as model training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"✅ ExplainabilityService initialized")
        logger.info(f"   Heatmap dir: {self.heatmap_dir}")
    
    def get_target_layer(self, model: torch.nn.Module) -> list:
        """
        Lấy target layer cuối cùng của EfficientNet cho Grad-CAM
        
        EfficientNet structure:
            - conv_stem, bn1
            - blocks (các MBConv blocks)
            - conv_head, bn2, global_pool, classifier
        
        Chọn conv_head (layer cuối trước pooling) để có heatmap tốt nhất
        
        Args:
            model: EfficientNet model từ timm
            
        Returns:
            List chứa target layer
        """
        # Với timm EfficientNet, conv_head là layer tốt nhất cho Grad-CAM
        if hasattr(model, 'conv_head'):
            return [model.conv_head]
        elif hasattr(model, 'features'):
            # Fallback cho một số model structures khác
            return [model.features[-1]]
        else:
            # Last resort: tìm last conv layer
            last_conv = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv:
                return [last_conv]
            raise ValueError("Cannot find suitable target layer for Grad-CAM")
    
    def generate_gradcam(
        self,
        image: Image.Image,
        model: torch.nn.Module,
        target_class: int = None,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for given image and model
        
        Args:
            image: PIL Image (face đã crop)
            model: PyTorch model (EfficientNet)
            target_class: Class index (0=FAKE, 1=REAL). None = use predicted class
            device: 'cuda' or 'cpu'
            
        Returns:
            Tuple[grayscale_cam, cam_image]:
                - grayscale_cam: numpy array (224, 224) với values 0-1
                - cam_image: numpy array (224, 224, 3) - heatmap overlay on image
        """
        if not self.enabled:
            raise RuntimeError("Grad-CAM not available. Install: pip install grad-cam")
        
        model.eval()
        model.to(device)
        
        # Prepare input
        input_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Get original image as numpy (for overlay)
        rgb_image = image.resize((224, 224))
        rgb_np = np.array(rgb_image).astype(np.float32) / 255.0
        
        # Get target layer
        target_layers = self.get_target_layer(model)
        
        # Create Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Determine target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first (and only) image
        
        # Create visualization
        cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
        
        logger.info(f"✅ Generated Grad-CAM for class {target_class}")
        
        return grayscale_cam, cam_image
    
    def save_heatmap(
        self,
        cam_image: np.ndarray,
        prediction_id: int = None,
        filename: str = None
    ) -> str:
        """
        Lưu heatmap image vào disk
        
        Args:
            cam_image: numpy array (H, W, 3) từ generate_gradcam
            prediction_id: ID của prediction (để naming)
            filename: Custom filename (optional)
            
        Returns:
            Absolute path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if prediction_id:
                filename = f"heatmap_{prediction_id}_{timestamp}.png"
            else:
                filename = f"heatmap_{timestamp}.png"
        
        filepath = os.path.join(self.heatmap_dir, filename)
        
        # Convert numpy to PIL and save
        heatmap_pil = Image.fromarray(cam_image.astype(np.uint8))
        heatmap_pil.save(filepath, 'PNG')
        
        logger.info(f"✅ Saved heatmap: {filepath}")
        return filepath
    
    def heatmap_to_base64(self, cam_image: np.ndarray) -> str:
        """
        Convert heatmap to base64 string (for JSON response)
        
        Args:
            cam_image: numpy array (H, W, 3)
            
        Returns:
            Base64 encoded PNG string
        """
        heatmap_pil = Image.fromarray(cam_image.astype(np.uint8))
        
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_explanation(
        self,
        image: Image.Image,
        model: torch.nn.Module,
        prediction_result: Dict,
        device: str = 'cpu',
        save_to_disk: bool = True,
        prediction_id: int = None
    ) -> Dict:
        """
        Main method: Generate complete explanation for a prediction
        
        Args:
            image: PIL Image (face)
            model: PyTorch model
            prediction_result: Dict với 'verdict', 'probabilities', etc.
            device: 'cuda' or 'cpu'
            save_to_disk: Có lưu file không
            prediction_id: ID để naming file
            
        Returns:
            {
                'heatmap_base64': str,
                'heatmap_path': str or None,
                'target_class': int,
                'target_class_name': 'FAKE' or 'REAL',
                'explanation': str (human-readable)
            }
        """
        if not self.enabled:
            return {
                'error': 'Grad-CAM not available',
                'heatmap_base64': None,
                'heatmap_path': None
            }
        
        try:
            # Determine target class from prediction
            verdict = prediction_result.get('verdict', 'FAKE')
            target_class = 0 if verdict == 'FAKE' else 1
            
            # Generate heatmap
            grayscale_cam, cam_image = self.generate_gradcam(
                image=image,
                model=model,
                target_class=target_class,
                device=device
            )
            
            # Convert to base64
            heatmap_base64 = self.heatmap_to_base64(cam_image)
            
            # Save to disk if requested
            heatmap_path = None
            if save_to_disk:
                heatmap_path = self.save_heatmap(
                    cam_image=cam_image,
                    prediction_id=prediction_id
                )
            
            # Generate human-readable explanation
            # Analyze heatmap to find which region is most important
            explanation = self._generate_text_explanation(grayscale_cam, verdict)
            
            return {
                'heatmap_base64': heatmap_base64,
                'heatmap_path': heatmap_path,
                'target_class': target_class,
                'target_class_name': verdict,
                'explanation': explanation,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating explanation: {e}", exc_info=True)
            return {
                'error': str(e),
                'heatmap_base64': None,
                'heatmap_path': None,
                'success': False
            }
    
    def _generate_text_explanation(self, grayscale_cam: np.ndarray, verdict: str) -> str:
        """
        Generate human-readable explanation based on heatmap
        
        Chia ảnh thành 9 vùng (3x3 grid) và xác định vùng nào có activation cao nhất
        """
        h, w = grayscale_cam.shape
        h3, w3 = h // 3, w // 3
        
        # Define regions (approximate face anatomy for 224x224 face crop)
        regions = {
            'trán': grayscale_cam[:h3, w3:-w3],
            'mắt trái': grayscale_cam[h3:2*h3, :w3],
            'mắt phải': grayscale_cam[h3:2*h3, -w3:],
            'mũi': grayscale_cam[h3:2*h3, w3:-w3],
            'má trái': grayscale_cam[2*h3:, :w3],
            'miệng': grayscale_cam[2*h3:, w3:-w3],
            'má phải': grayscale_cam[2*h3:, -w3:],
        }
        
        # Find top activated regions
        region_activations = {name: np.mean(region) for name, region in regions.items()}
        sorted_regions = sorted(region_activations.items(), key=lambda x: x[1], reverse=True)
        
        top_regions = [r[0] for r in sorted_regions[:2]]
        
        if verdict == 'FAKE':
            explanation = f"AI phát hiện dấu hiệu bất thường tập trung ở vùng {' và '.join(top_regions)}. "
            explanation += "Đây có thể là do sự không khớp về ánh sáng, kết cấu da, hoặc biên khuôn mặt."
        else:
            explanation = f"AI không phát hiện dấu hiệu đáng ngờ. "
            explanation += f"Các vùng {' và '.join(top_regions)} được xem xét kỹ nhất và không có bất thường."
        
        return explanation


# Singleton instance
_explainability_service = None

def get_explainability_service() -> ExplainabilityService:
    """Get singleton ExplainabilityService instance"""
    global _explainability_service
    if _explainability_service is None:
        _explainability_service = ExplainabilityService()
    return _explainability_service


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🧪 Testing ExplainabilityService...")
    print("="*60)
    
    service = get_explainability_service()
    print(f"Enabled: {service.enabled}")
    print(f"Heatmap dir: {service.heatmap_dir}")
    
    if service.enabled:
        print("\n✅ ExplainabilityService ready!")
    else:
        print("\n⚠️ Grad-CAM not available")
