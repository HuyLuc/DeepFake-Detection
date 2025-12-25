# src/training/dataset.py (Phiên bản chống lỗi ảnh hỏng)

import os
from typing import Optional, Tuple, List, Callable
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError # --- THÊM MỚI: Import UnidentifiedImageError
from glob import glob
import torch

class DeepfakeDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh cho bài toán Deepfake.
    Nó sẽ tự động tìm tất cả các ảnh trong các thư mục con (train/REAL, train/FAKE,...)
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None) -> None:
        """
        :param data_dir: Đường dẫn đến thư mục chứa dữ liệu (ví dụ: .../processed_data/train)
        :param transform: Các phép biến đổi (augmentation) cần áp dụng
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Tự động tìm các lớp (FAKE, REAL)
        self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Lấy danh sách tất cả các đường dẫn ảnh và nhãn tương ứng
        self.image_paths, self.labels = self._load_data()
        print(f"Loaded {len(self.image_paths)} images from {data_dir}.")

    def _load_data(self) -> Tuple[List[str], List[int]]:
        images = []
        labels = []
        for class_name, class_idx in self.class_to_idx.items():
            # Mỗi video là một thư mục con chứa các frame
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
        Nếu ảnh bị hỏng, sẽ thử các ảnh tiếp theo (tối đa len(self.image_paths) lần).
        Đảm bảo label luôn khớp với ảnh được load.
        """
        original_idx = idx
        max_attempts = len(self.image_paths)
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Lưu idx hiện tại để đảm bảo label khớp với ảnh
                current_idx = idx
                img_path = self.image_paths[current_idx]
                
                # Kiểm tra kích thước file trước khi mở
                if os.path.getsize(img_path) == 0:
                    # Nếu file rỗng, thử lấy ảnh khác
                    idx = (idx + 1) % len(self.image_paths)
                    attempts += 1
                    continue

                image = Image.open(img_path).convert("RGB")
                # Kiểm tra ảnh có hợp lệ không
                image.verify()  # Verify image integrity
                image = Image.open(img_path).convert("RGB")  # Reopen after verify
                
                # QUAN TRỌNG: Lấy label tương ứng với ảnh hiện tại (current_idx), không phải idx đã bị thay đổi
                label = self.labels[current_idx]

                if self.transform:
                    image = self.transform(image)
                
                return image, label  # Trả về dữ liệu hợp lệ với label đúng

            except (OSError, IOError, UnidentifiedImageError, SyntaxError) as e:
                # Catch các exception liên quan đến lỗi file/ảnh:
                # - OSError, IOError: Lỗi đọc file
                # - UnidentifiedImageError: Không nhận diện được định dạng ảnh
                # - SyntaxError: Image.verify() có thể raise SyntaxError khi dữ liệu ảnh không hợp lệ
                # Không catch Exception để tránh che giấu lỗi lập trình (NameError, TypeError, etc.)
                # Nếu gặp lỗi khi mở ảnh (ảnh hỏng), thử ảnh tiếp theo
                idx = (idx + 1) % len(self.image_paths)
                attempts += 1
                if attempts >= max_attempts:
                    # Nếu đã thử hết tất cả ảnh mà vẫn lỗi, raise exception
                    raise RuntimeError(
                        f"Không thể load ảnh hợp lệ sau {max_attempts} lần thử. "
                        f"Bắt đầu từ idx {original_idx}. Lỗi cuối cùng: {e}"
                    )
