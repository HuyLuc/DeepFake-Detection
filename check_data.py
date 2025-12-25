#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiem tra chat luong du lieu da xu ly
"""

import os
import sys
import glob
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from configs import config
from src.training.dataset import DeepfakeDataset

def check_processed_data():
    """Kiem tra du lieu da xu ly"""
    print("=" * 60)
    print("KIEM TRA DU LIEU DA XU LY")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    classes = ['FAKE', 'REAL']
    
    total_videos = 0
    total_frames = 0
    videos_with_issues = []
    all_frame_counts = []
    
    # Kiểm tra từng split
    for split in splits:
        print(f"\n--- {split.upper()} ---")
        split_videos = 0
        split_frames = 0
        
        for cls in classes:
            dir_path = os.path.join(config.PROCESSED_DATA_DIR, split, cls)
            if not os.path.exists(dir_path):
                print(f"  {cls}: Thư mục không tồn tại")
                continue
            
            video_dirs = [d for d in os.listdir(dir_path) 
                         if os.path.isdir(os.path.join(dir_path, d))]
            frame_counts = []
            
            for vdir in video_dirs:
                frames = glob.glob(os.path.join(dir_path, vdir, '*.png'))
                count = len(frames)
                frame_counts.append(count)
                all_frame_counts.append(count)
                
                # Kiểm tra video có vấn đề
                if count < config.NUM_FRAMES_PER_VIDEO:
                    videos_with_issues.append((split, cls, vdir, count))
                elif count > config.NUM_FRAMES_PER_VIDEO:
                    videos_with_issues.append((split, cls, vdir, count))
            
            split_videos += len(video_dirs)
            split_frames += sum(frame_counts)
            
            if frame_counts:
                print(f"  {cls}:")
                print(f"    - Số video: {len(video_dirs)}")
                print(f"    - Frames/video: min={min(frame_counts)}, max={max(frame_counts)}, "
                      f"avg={sum(frame_counts)/len(frame_counts):.1f}")
                print(f"    - Videos đủ {config.NUM_FRAMES_PER_VIDEO} frames: "
                      f"{sum(1 for c in frame_counts if c == config.NUM_FRAMES_PER_VIDEO)}")
        
        total_videos += split_videos
        total_frames += split_frames
        print(f"  Tổng {split}: {split_videos} videos, {split_frames} frames")
    
    # Thống kê tổng quan
    print("\n" + "=" * 60)
    print("THONG KE TONG QUAN")
    print("=" * 60)
    print(f"Tổng số video: {total_videos}")
    print(f"Tổng số frames: {total_frames}")
    
    if all_frame_counts:
        print(f"\nSố frames/video:")
        print(f"  - Min: {min(all_frame_counts)}")
        print(f"  - Max: {max(all_frame_counts)}")
        print(f"  - Trung bình: {np.mean(all_frame_counts):.1f}")
        print(f"  - Median: {np.median(all_frame_counts):.1f}")
        
        perfect_count = sum(1 for c in all_frame_counts if c == config.NUM_FRAMES_PER_VIDEO)
        less_count = sum(1 for c in all_frame_counts if c < config.NUM_FRAMES_PER_VIDEO)
        more_count = sum(1 for c in all_frame_counts if c > config.NUM_FRAMES_PER_VIDEO)
        
        print(f"\nChat luong du lieu:")
        print(f"  OK Videos du {config.NUM_FRAMES_PER_VIDEO} frames: {perfect_count} "
              f"({100*perfect_count/len(all_frame_counts):.1f}%)")
        print(f"  WARNING Videos thieu frames (<{config.NUM_FRAMES_PER_VIDEO}): {less_count} "
              f"({100*less_count/len(all_frame_counts):.1f}%)")
        print(f"  WARNING Videos thua frames (>{config.NUM_FRAMES_PER_VIDEO}): {more_count} "
              f"({100*more_count/len(all_frame_counts):.1f}%)")
    
    # Liet ke videos co van de
    if videos_with_issues:
        print("\n" + "=" * 60)
        print(f"WARNING: PHAT HIEN {len(videos_with_issues)} VIDEO CO VAN DE")
        print("=" * 60)
        for split, cls, vdir, count in videos_with_issues[:20]:  # Hien thi 20 dau tien
            status = "THIEU" if count < config.NUM_FRAMES_PER_VIDEO else "THUA"
            print(f"  {status}: {split}/{cls}/{vdir}: {count} frames")
        if len(videos_with_issues) > 20:
            print(f"  ... va {len(videos_with_issues) - 20} video khac")
    else:
        print("\n" + "=" * 60)
        print("OK: TAT CA VIDEO DEU DU SO FRAMES!")
        print("=" * 60)
    
    # Kiểm tra Dataset loader
    print("\n" + "=" * 60)
    print("KIỂM TRA DATASET LOADER")
    print("=" * 60)
    for split in splits:
        dir_path = os.path.join(config.PROCESSED_DATA_DIR, split)
        if os.path.exists(dir_path):
            try:
                ds = DeepfakeDataset(dir_path)
                print(f"{split.upper()}: ✅ {len(ds)} images, classes: {ds.classes}")
            except Exception as e:
                print(f"{split.upper()}: ❌ Lỗi - {e}")
    
    print("\n" + "=" * 60)
    print("KET LUAN")
    print("=" * 60)
    if not videos_with_issues:
        print("OK: Du lieu hoan toan OK! Tat ca video deu co du so frames.")
        print("OK: Co the tiep tuc training/evaluation.")
    else:
        print(f"WARNING: Co {len(videos_with_issues)} video co van de.")
        print("KHUYEN NGHI: Chay lai preprocess de sua cac video thieu/thua frames.")
        print("   (Code moi da co Uniform Sampling + Temporal Padding)")

if __name__ == "__main__":
    check_processed_data()

