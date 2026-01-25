# src/training/temporal_dataset.py
"""
Dataset cho Temporal Models: Load sequences of frames thay vì single frames.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Tuple, List
from glob import glob
import random
import numpy as np


class TemporalDeepfakeDataset(Dataset):
    """
    Dataset load sequences of frames cho Temporal/Ensemble models.
    
    Thay vì load từng frame độc lập, dataset này load một sequence
    của frames từ cùng một video để model có thể học temporal patterns.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        sequence_length: int = 10,
        sampling_strategy: str = 'uniform',  # 'uniform', 'random', 'consecutive'
        return_video_id: bool = False
    ):
        """
        Args:
            data_dir: Đường dẫn đến thư mục dữ liệu (e.g., processed_data/train)
            transform: Transform cho mỗi frame
            sequence_length: Số frames trong mỗi sequence
            sampling_strategy: Cách chọn frames từ video:
                - 'uniform': Lấy đều từ đầu đến cuối
                - 'random': Lấy ngẫu nhiên
                - 'consecutive': Lấy liên tiếp từ vị trí ngẫu nhiên
            return_video_id: Có trả về video_id không
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.sampling_strategy = sampling_strategy
        self.return_video_id = return_video_id
        
        # Tìm tất cả các lớp
        self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load danh sách videos (mỗi video là một thư mục chứa frames)
        self.videos = self._load_videos()
        
        print(f"📂 Loaded {len(self.videos)} videos from {data_dir}")
        print(f"   Classes: {self.classes}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Sampling strategy: {sampling_strategy}")
    
    def _load_videos(self) -> List[dict]:
        """
        Load danh sách videos với các frame paths.
        
        Returns:
            List of dicts: [{'video_id': str, 'frame_paths': List[str], 'label': int}, ...]
        """
        videos = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            
            # Mỗi video là một thư mục con
            video_dirs = [d for d in os.scandir(class_dir) if d.is_dir()]
            
            for video_dir in video_dirs:
                frame_paths = sorted(glob(os.path.join(video_dir.path, '*.png')))
                
                if len(frame_paths) > 0:
                    videos.append({
                        'video_id': video_dir.name,
                        'frame_paths': frame_paths,
                        'label': class_idx
                    })
        
        return videos
    
    def _sample_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Chọn frames từ video theo strategy.
        
        Args:
            frame_paths: Danh sách đường dẫn tất cả frames của video
        
        Returns:
            List of selected frame paths
        """
        num_frames = len(frame_paths)
        
        if num_frames <= self.sequence_length:
            # Video ngắn: Lấy tất cả + padding
            selected = frame_paths.copy()
            # Padding bằng cách lặp lại frame cuối
            while len(selected) < self.sequence_length:
                selected.append(frame_paths[-1])
            return selected
        
        if self.sampling_strategy == 'uniform':
            # Lấy đều từ đầu đến cuối
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            return [frame_paths[i] for i in indices]
        
        elif self.sampling_strategy == 'random':
            # Lấy ngẫu nhiên (không lặp)
            indices = sorted(random.sample(range(num_frames), self.sequence_length))
            return [frame_paths[i] for i in indices]
        
        elif self.sampling_strategy == 'consecutive':
            # Lấy liên tiếp từ vị trí ngẫu nhiên
            max_start = num_frames - self.sequence_length
            start = random.randint(0, max_start)
            return frame_paths[start:start + self.sequence_length]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Lấy một sequence của frames.
        
        Returns:
            Tuple:
                - frames: (sequence_length, C, H, W)
                - label: int
        """
        video_info = self.videos[idx]
        frame_paths = video_info['frame_paths']
        label = video_info['label']
        
        # Chọn frames theo strategy
        selected_paths = self._sample_frames(frame_paths)
        
        # Load và transform các frames
        frames = []
        for path in selected_paths:
            try:
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            except Exception as e:
                # Nếu lỗi, tạo tensor zeros
                print(f"Warning: Error loading {path}: {e}")
                if self.transform and len(frames) > 0:
                    frames.append(torch.zeros_like(frames[-1]))
                else:
                    # Tạo dummy tensor
                    frames.append(torch.zeros(3, 380, 380))
        
        # Stack thành tensor (sequence_length, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        
        if self.return_video_id:
            return frames_tensor, label, video_info['video_id']
        
        return frames_tensor, label


class HybridDataset(Dataset):
    """
    Dataset hỗn hợp: Có thể trả về cả single frames và sequences.
    Hữu ích khi muốn train với cả image-level và video-level loss.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        sequence_length: int = 10,
        mode: str = 'sequence'  # 'single', 'sequence', 'both'
    ):
        self.temporal_dataset = TemporalDeepfakeDataset(
            data_dir=data_dir,
            transform=transform,
            sequence_length=sequence_length
        )
        self.mode = mode
        self.classes = self.temporal_dataset.classes
        self.class_to_idx = self.temporal_dataset.class_to_idx
    
    def __len__(self) -> int:
        if self.mode == 'single':
            # Đếm tổng số frames
            total = sum(len(v['frame_paths']) for v in self.temporal_dataset.videos)
            return total
        return len(self.temporal_dataset)
    
    def __getitem__(self, idx: int):
        if self.mode == 'sequence':
            return self.temporal_dataset[idx]
        
        elif self.mode == 'single':
            # Map idx to video and frame
            current = 0
            for video in self.temporal_dataset.videos:
                num_frames = len(video['frame_paths'])
                if idx < current + num_frames:
                    frame_idx = idx - current
                    frame_path = video['frame_paths'][frame_idx]
                    label = video['label']
                    
                    image = Image.open(frame_path).convert('RGB')
                    if self.temporal_dataset.transform:
                        image = self.temporal_dataset.transform(image)
                    
                    return image, label
                current += num_frames
            
            raise IndexError(f"Index {idx} out of range")
        
        elif self.mode == 'both':
            # Trả về cả sequence và middle frame
            frames, label = self.temporal_dataset[idx]
            middle_frame = frames[len(frames) // 2]
            return frames, middle_frame, label


def create_temporal_dataloaders(
    train_dir: str,
    val_dir: str,
    train_transform: Callable,
    val_transform: Callable,
    sequence_length: int = 10,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Helper function để tạo DataLoaders cho temporal training.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TemporalDeepfakeDataset(
        data_dir=train_dir,
        transform=train_transform,
        sequence_length=sequence_length,
        sampling_strategy='uniform'
    )
    
    val_dataset = TemporalDeepfakeDataset(
        data_dir=val_dir,
        transform=val_transform,
        sequence_length=sequence_length,
        sampling_strategy='uniform'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last để batch size consistent
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
