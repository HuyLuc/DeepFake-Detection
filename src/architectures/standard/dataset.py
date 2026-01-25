# src/architectures/standard/dataset.py
"""
🔵 STANDARD ARCHITECTURE - Dataset
Dataset đơn giản, load từng frame độc lập.
"""

import os
from typing import Optional, Tuple, List, Callable
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from glob import glob
import torch


class DeepfakeDataset(Dataset):
    """
    Dataset tùy chỉnh cho bài toán Deepfake - Kiến trúc Standard.
    Load từng frame độc lập, không có temporal information.
    """
    
    def __init__(self, data_dir: str, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data_dir: Đường dẫn đến thư mục dữ liệu (e.g., processed_data/train)
            transform: Các phép biến đổi cần áp dụng
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Tự động tìm các lớp (FAKE, REAL)
        self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load dữ liệu
        self.image_paths, self.labels = self._load_data()
        print(f"📂 Loaded {len(self.image_paths)} images from {data_dir}")

    def _load_data(self) -> Tuple[List[str], List[int]]:
        images = []
        labels = []
        for class_name, class_idx in self.class_to_idx.items():
            video_dirs = glob(os.path.join(self.data_dir, class_name, '*'))
            for video_dir in video_dirs:
                if os.path.isdir(video_dir):
                    frame_paths = glob(os.path.join(video_dir, '*.png'))
                    for frame_path in frame_paths:
                        images.append(frame_path)
                        labels.append(class_idx)
        return images, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Lấy một mẫu từ dataset với xử lý lỗi an toàn.
        """
        original_idx = idx
        max_attempts = len(self.image_paths)
        attempts = 0
        
        while attempts < max_attempts:
            try:
                current_idx = idx
                img_path = self.image_paths[current_idx]
                
                if os.path.getsize(img_path) == 0:
                    idx = (idx + 1) % len(self.image_paths)
                    attempts += 1
                    continue

                image = Image.open(img_path).convert("RGB")
                image.verify()
                image = Image.open(img_path).convert("RGB")
                
                label = self.labels[current_idx]

                if self.transform:
                    image = self.transform(image)
                
                return image, label

            except (OSError, IOError, UnidentifiedImageError, SyntaxError) as e:
                idx = (idx + 1) % len(self.image_paths)
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Không thể load ảnh hợp lệ sau {max_attempts} lần thử. "
                        f"Bắt đầu từ idx {original_idx}. Lỗi: {e}"
                    )
