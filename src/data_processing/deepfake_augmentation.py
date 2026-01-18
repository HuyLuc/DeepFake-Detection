# src/data_processing/deepfake_augmentation.py
"""
Data Augmentation chuyên biệt cho bài toán Deepfake Detection.
Bao gồm các phép biến đổi mô phỏng các đặc điểm thực tế của video giả.
"""

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import random
import io
import numpy as np


class JPEGCompression:
    """
    Mô phỏng compression artifacts từ việc nén JPEG/Video.
    Deepfake thường bị mờ đi và xuất hiện artifacts khi nén.
    """
    def __init__(self, quality_range=(30, 95), p=0.5):
        """
        Args:
            quality_range: Tuple (min_quality, max_quality) cho JPEG compression (0-100)
            p: Xác suất áp dụng augmentation này
        """
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image đã được compress
        """
        if random.random() > self.p:
            return img
        
        # Random quality level
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        
        # Convert PIL Image to JPEG bytes với quality thấp
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=quality)
        output_buffer.seek(0)
        
        # Load lại từ bytes để có compression artifacts
        compressed_img = Image.open(output_buffer)
        
        return compressed_img


class AdaptiveGaussianNoise:
    """
    Thêm Gaussian noise với cường độ thích ứng.
    Mô phỏng nhiễu từ camera chất lượng thấp hoặc điều kiện ánh sáng xấu.
    """
    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        """
        Args:
            std_range: Tuple (min_std, max_std) cho độ lệch chuẩn của noise
            p: Xác suất áp dụng augmentation này
        """
        self.std_range = std_range
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image đã thêm noise
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Random noise level
        std = random.uniform(self.std_range[0], self.std_range[1])
        
        # Thêm Gaussian noise
        noise = np.random.normal(0, std, img_array.shape).astype(np.float32)
        noisy_img = img_array + noise
        
        # Clip về range [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # Convert lại sang PIL Image
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class AdaptiveGaussianBlur:
    """
    Blur thích ứng để mô phỏng sự mất nét trong video Deepfake.
    """
    def __init__(self, kernel_size_range=(3, 9), sigma_range=(0.1, 2.0), p=0.2):
        """
        Args:
            kernel_size_range: Range cho kernel size (phải là số lẻ)
            sigma_range: Range cho sigma của Gaussian blur
            p: Xác suất áp dụng augmentation này
        """
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image đã blur
        """
        if random.random() > self.p:
            return img
        
        # Random kernel size (phải là số lẻ)
        kernel_size = random.randrange(
            self.kernel_size_range[0], 
            self.kernel_size_range[1] + 1, 
            2  # step = 2 để đảm bảo số lẻ
        )
        
        # Random sigma
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # Áp dụng Gaussian blur
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return blurred_img


class FaceCutout:
    """
    Random cutout (xóa) một phần nhỏ trên khuôn mặt.
    Giúp model học cách tập trung vào nhiều đặc điểm khác nhau
    thay vì chỉ nhìn vào một vùng cụ thể (mắt, miệng).
    """
    def __init__(self, num_holes=1, max_h_size=0.15, max_w_size=0.15, p=0.3):
        """
        Args:
            num_holes: Số lượng cutout regions
            max_h_size: Tỷ lệ chiều cao tối đa của cutout so với ảnh (0-1)
            max_w_size: Tỷ lệ chiều rộng tối đa của cutout so với ảnh (0-1)
            p: Xác suất áp dụng augmentation này
        """
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image với cutout regions
        """
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        h, w, c = img_array.shape
        
        for _ in range(self.num_holes):
            # Random kích thước cutout
            cutout_h = int(h * random.uniform(0.05, self.max_h_size))
            cutout_w = int(w * random.uniform(0.05, self.max_w_size))
            
            # Random vị trí cutout
            y = random.randint(0, h - cutout_h)
            x = random.randint(0, w - cutout_w)
            
            # Fill cutout region với màu xám (giá trị trung bình)
            # Thay vì fill bằng 0 (đen) để tự nhiên hơn
            mean_color = img_array[y:y+cutout_h, x:x+cutout_w].mean(axis=(0, 1))
            img_array[y:y+cutout_h, x:x+cutout_w] = mean_color
        
        return Image.fromarray(img_array)


class MixedDeepfakeAugmentation:
    """
    Kết hợp nhiều phép augmentation chuyên biệt cho Deepfake.
    Đảm bảo ít nhất một phép augmentation được áp dụng.
    """
    def __init__(self, enable_compression=True, enable_noise=True, 
                 enable_blur=True, enable_cutout=True):
        """
        Args:
            enable_compression: Bật/tắt JPEG compression
            enable_noise: Bật/tắt Gaussian noise
            enable_blur: Bật/tắt Gaussian blur
            enable_cutout: Bật/tắt Face cutout
        """
        self.augmentations = []
        
        if enable_compression:
            self.augmentations.append(JPEGCompression(quality_range=(30, 95), p=0.5))
        if enable_noise:
            self.augmentations.append(AdaptiveGaussianNoise(std_range=(0.01, 0.05), p=0.3))
        if enable_blur:
            self.augmentations.append(AdaptiveGaussianBlur(kernel_size_range=(3, 7), sigma_range=(0.1, 1.5), p=0.2))
        if enable_cutout:
            self.augmentations.append(FaceCutout(num_holes=1, max_h_size=0.15, max_w_size=0.15, p=0.3))
    
    def __call__(self, img):
        """
        Áp dụng các augmentations theo xác suất của từng phép.
        
        Args:
            img: PIL Image
        Returns:
            PIL Image đã augment
        """
        for aug in self.augmentations:
            img = aug(img)
        
        return img


def get_deepfake_train_transforms(image_size=(380, 380), use_deepfake_aug=True):
    """
    Tạo transform pipeline cho training với augmentation chuyên biệt.
    
    Args:
        image_size: Tuple (height, width) cho kích thước ảnh đầu ra
        use_deepfake_aug: Bật/tắt Deepfake-specific augmentations
    
    Returns:
        torchvision.transforms.Compose object
    """
    from torchvision import transforms
    
    transform_list = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ]
    
    # THÊM: Deepfake-specific augmentations (áp dụng TRƯỚC khi chuyển sang Tensor)
    if use_deepfake_aug:
        transform_list.append(MixedDeepfakeAugmentation(
            enable_compression=True,
            enable_noise=True,
            enable_blur=True,
            enable_cutout=True
        ))
    
    # Standard augmentations
    transform_list.extend([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.1),
        transforms.ToTensor(),  # Chuyển PIL -> Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.1)),
    ])
    
    return transforms.Compose(transform_list)


def get_deepfake_val_transforms(image_size=(380, 380)):
    """
    Tạo transform pipeline cho validation (không có augmentation).
    
    Args:
        image_size: Tuple (height, width) cho kích thước ảnh đầu ra
    
    Returns:
        torchvision.transforms.Compose object
    """
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
