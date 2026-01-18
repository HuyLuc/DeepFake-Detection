# src/training/balanced_dataset.py
"""
Dataset wrapper với Oversampling để cân bằng dữ liệu.
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from collections import Counter


class OversampledDataset(Dataset):
    """
    Wrapper dataset thực hiện oversampling cho lớp thiểu số.
    Thay vì chỉ dùng class_weights, ta tăng số lượng mẫu của lớp REAL
    để model nhìn thấy lớp REAL nhiều hơn, giảm False Positive.
    """
    def __init__(self, base_dataset, oversample_ratio=1.5):
        """
        Args:
            base_dataset: Dataset gốc (DeepfakeDataset)
            oversample_ratio: Tỷ lệ oversample cho lớp thiểu số
                             - 1.0: không oversample (cân bằng hoàn toàn)
                             - 1.5: lớp thiểu số sẽ có 1.5x số mẫu của lớp đa số
                             - 2.0: lớp thiểu số sẽ có 2x số mẫu của lớp đa số
        """
        self.base_dataset = base_dataset
        self.oversample_ratio = oversample_ratio
        
        # Phân tích phân bố lớp
        labels = np.array(base_dataset.labels)
        self.label_counts = Counter(labels)
        
        print(f"\n📊 Phân bố dữ liệu gốc:")
        for label_idx, count in sorted(self.label_counts.items()):
            class_name = base_dataset.classes[label_idx]
            print(f"   {class_name}: {count} mẫu")
        
        # Tính toán indices cho oversampling
        self.oversampled_indices = self._create_oversampled_indices()
        
        print(f"\n📊 Phân bố dữ liệu sau oversampling (ratio={oversample_ratio}):")
        oversampled_labels = [labels[i] for i in self.oversampled_indices]
        oversampled_counts = Counter(oversampled_labels)
        for label_idx, count in sorted(oversampled_counts.items()):
            class_name = base_dataset.classes[label_idx]
            print(f"   {class_name}: {count} mẫu")
        
        print(f"\n✅ Tổng số mẫu sau oversampling: {len(self.oversampled_indices)}")
    
    def _create_oversampled_indices(self):
        """
        Tạo danh sách indices với oversampling cho lớp thiểu số.
        
        Returns:
            List of indices
        """
        labels = np.array(self.base_dataset.labels)
        
        # Tìm lớp đa số và lớp thiểu số
        max_count = max(self.label_counts.values())
        min_count = min(self.label_counts.values())
        
        majority_label = max(self.label_counts, key=self.label_counts.get)
        minority_label = min(self.label_counts, key=self.label_counts.get)
        
        print(f"\n🔍 Phân tích:")
        print(f"   Lớp đa số: {self.base_dataset.classes[majority_label]} ({max_count} mẫu)")
        print(f"   Lớp thiểu số: {self.base_dataset.classes[minority_label]} ({min_count} mẫu)")
        print(f"   Tỷ lệ mất cân bằng: {max_count/min_count:.2f}:1")
        
        # Tạo oversampled indices
        indices = []
        
        for label_idx in range(len(self.base_dataset.classes)):
            # Lấy tất cả indices của lớp này
            class_indices = np.where(labels == label_idx)[0].tolist()
            
            if label_idx == minority_label:
                # Oversample lớp thiểu số
                target_count = int(max_count * self.oversample_ratio)
                
                # Lặp lại các mẫu để đạt target_count
                num_repeats = target_count // len(class_indices)
                remainder = target_count % len(class_indices)
                
                # Thêm các mẫu đã lặp
                indices.extend(class_indices * num_repeats)
                
                # Thêm phần dư (random sampling)
                if remainder > 0:
                    indices.extend(np.random.choice(class_indices, remainder, replace=False).tolist())
            else:
                # Giữ nguyên lớp đa số
                indices.extend(class_indices)
        
        # Shuffle indices để tránh bias
        np.random.shuffle(indices)
        
        return indices
    
    def __len__(self):
        return len(self.oversampled_indices)
    
    def __getitem__(self, idx):
        """
        Lấy mẫu từ base_dataset theo oversampled_indices.
        """
        real_idx = self.oversampled_indices[idx]
        return self.base_dataset[real_idx]


def create_weighted_sampler(dataset, oversample_ratio=1.5):
    """
    Tạo WeightedRandomSampler để oversample lớp thiểu số.
    Đây là cách alternative cho OversampledDataset, dùng sampler thay vì wrapper.
    
    Args:
        dataset: Dataset gốc (DeepfakeDataset)
        oversample_ratio: Tỷ lệ oversample cho lớp thiểu số
    
    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    labels = np.array(dataset.labels)
    label_counts = Counter(labels)
    
    print(f"\n📊 Phân bố dữ liệu gốc:")
    for label_idx, count in sorted(label_counts.items()):
        class_name = dataset.classes[label_idx]
        print(f"   {class_name}: {count} mẫu")
    
    # Tính toán weights cho mỗi mẫu
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    
    majority_label = max(label_counts, key=label_counts.get)
    minority_label = min(label_counts, key=label_counts.get)
    
    print(f"\n🔍 Phân tích:")
    print(f"   Lớp đa số: {dataset.classes[majority_label]} ({max_count} mẫu)")
    print(f"   Lớp thiểu số: {dataset.classes[minority_label]} ({min_count} mẫu)")
    print(f"   Tỷ lệ mất cân bằng: {max_count/min_count:.2f}:1")
    
    # Tính weight cho mỗi lớp
    class_weights = {}
    for label_idx, count in label_counts.items():
        if label_idx == minority_label:
            # Weight cao hơn cho lớp thiểu số
            class_weights[label_idx] = (max_count * oversample_ratio) / count
        else:
            # Weight bình thường cho lớp đa số
            class_weights[label_idx] = 1.0
    
    print(f"\n⚖️ Class weights:")
    for label_idx, weight in sorted(class_weights.items()):
        class_name = dataset.classes[label_idx]
        print(f"   {class_name}: {weight:.2f}")
    
    # Tạo weight cho mỗi mẫu
    sample_weights = [class_weights[label] for label in labels]
    
    # Tạo sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Cho phép lặp lại mẫu
    )
    
    print(f"\n✅ Đã tạo WeightedRandomSampler với oversample_ratio={oversample_ratio}")
    
    return sampler


def get_balanced_dataloader(dataset, batch_size, num_workers=0, pin_memory=False,
                            method='oversampling', oversample_ratio=1.5):
    """
    Tạo DataLoader với cân bằng dữ liệu.
    
    Args:
        dataset: Dataset gốc (DeepfakeDataset)
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
        pin_memory: Pin memory cho CUDA
        method: Phương pháp cân bằng ('oversampling' hoặc 'weighted_sampler')
        oversample_ratio: Tỷ lệ oversample cho lớp thiểu số
    
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    
    if method == 'oversampling':
        # Sử dụng OversampledDataset wrapper
        balanced_dataset = OversampledDataset(dataset, oversample_ratio=oversample_ratio)
        loader = DataLoader(
            balanced_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle vì không dùng sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False
        )
    elif method == 'weighted_sampler':
        # Sử dụng WeightedRandomSampler
        sampler = create_weighted_sampler(dataset, oversample_ratio=oversample_ratio)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # Dùng sampler thay vì shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False
        )
    else:
        raise ValueError(f"Unknown balancing method: {method}. Use 'oversampling' or 'weighted_sampler'.")
    
    print(f"\n✅ Đã tạo balanced DataLoader với method='{method}'")
    return loader
