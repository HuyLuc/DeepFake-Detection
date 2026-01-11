# src/data_processing/preprocess.py (phiên bản TỐI ƯU HÓA TỐC ĐỘ)

import cv2
import mediapipe as mp
import os
import glob
from tqdm import tqdm
import multiprocessing
import random
import numpy as np

# Import các cấu hình và tiện ích
from configs import config
from src.utils.utils import verify_data_structure

def process_single_video(video_info):
    """
    Hàm xử lý cho MỘT video, được TỐI ƯU để đọc tuần tự.
    Khởi tạo MediaPipe trong hàm để tránh memory leak với multiprocessing.
    Tự động bỏ qua video đã được xử lý (đã có đủ số frame).
    """
    source_video_path, output_dir = video_info
    
    # KIỂM TRA: Nếu video đã được xử lý rồi thì bỏ qua
    if os.path.exists(output_dir):
        # Đếm số frame đã có trong thư mục
        existing_frames = glob.glob(os.path.join(output_dir, 'frame_*.png'))
        if len(existing_frames) >= config.NUM_FRAMES_PER_VIDEO:
            # Video đã được xử lý đầy đủ, bỏ qua
            return f"Skipped (already processed): {os.path.basename(source_video_path)} ({len(existing_frames)} frames)"
    
    # Nếu chưa có hoặc thiếu frame, xóa folder cũ và tạo mới
    import shutil
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except (OSError, PermissionError, FileNotFoundError) as e:
            # Nếu không xóa được, log lỗi nhưng tiếp tục
            print(f"Warning: Không thể xóa thư mục {output_dir}: {e}")
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo MediaPipe Face Detector trong process để tránh memory leak
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=getattr(config, 'FACE_DETECTION_MODEL', 1),
        min_detection_confidence=getattr(config, 'FACE_DETECTION_CONFIDENCE', 0.5)
    )

    try:
        cap = cv2.VideoCapture(source_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- UNIFORM SAMPLING + TEMPORAL PADDING ---
        if total_frames <= 0:
            return f"Error: Video {source_video_path} has no frames"
        
        if total_frames >= config.NUM_FRAMES_PER_VIDEO:
            # Video dài: Uniform Sampling với linspace
            # Tạo các chỉ số rải đều trên toàn bộ video
            indices = np.linspace(0, total_frames - 1, config.NUM_FRAMES_PER_VIDEO, dtype=int)
        else:
            # Video ngắn: Lấy tất cả frames + Temporal Padding
            # Lấy tất cả frames có sẵn
            indices = np.arange(total_frames)
            # Padding: Lặp lại frame cuối cùng cho đến khi đủ NUM_FRAMES_PER_VIDEO
            last_frame_idx = total_frames - 1
            padding_needed = config.NUM_FRAMES_PER_VIDEO - total_frames
            if padding_needed > 0:
                padding_indices = np.full(padding_needed, last_frame_idx, dtype=int)
                indices = np.concatenate([indices, padding_indices])
        # --------------------------------------------
            
        saved_frame_count = 0
        frame_counter = 0
        next_frame_idx_to_save = 0

        while cap.isOpened() and next_frame_idx_to_save < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chỉ xử lý những frame có trong danh sách đã chọn
            if frame_counter == indices[next_frame_idx_to_save]:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(image_rgb)
                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    margin = config.FACE_MARGIN
                    face = frame[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
                    
                    if face.size != 0:
                        output_filename = os.path.join(output_dir, f'frame_{frame_counter}.png')
                        cv2.imwrite(output_filename, face)
                        saved_frame_count += 1
                
                next_frame_idx_to_save += 1 # Chuyển sang frame tiếp theo cần lưu

            frame_counter += 1
            
        cap.release()
        # Cleanup MediaPipe detector (sử dụng context manager hoặc del)
        del face_detector
        return f"Processed: {os.path.basename(source_video_path)} ({saved_frame_count} frames)"

    except Exception as e:
        # Đảm bảo cleanup ngay cả khi có lỗi
        try:
            del face_detector
        except:
            pass
        return f"Error processing video {source_video_path}: {e}"

def run_preprocessing():
    """Hàm chính điều phối tiền xử lý (phiên bản tối ưu memory)."""
    verify_data_structure()
    print("\n--- Bước 1: Thu thập và phân chia ID video ---")
    all_original_paths = []
    for dir_path in config.ORIGINAL_DIRS.values():
        if os.path.exists(dir_path):
            all_original_paths.extend(glob.glob(os.path.join(dir_path, '*.mp4')))
        else:
            print(f"Cảnh báo: Không tìm thấy thư mục {dir_path}")
    if not all_original_paths:
        print("Lỗi nghiêm trọng: Không tìm thấy bất kỳ video gốc nào. Dừng quá trình.")
        return
    original_ids = sorted(list(set([os.path.basename(p).split('.')[0] for p in all_original_paths])))
    print(f"Tìm thấy {len(original_ids)} ID video gốc duy nhất.")
    random.seed(config.RANDOM_SEED)
    random.shuffle(original_ids)
    train_count = int(len(original_ids) * config.TRAIN_SPLIT)
    val_count = int(len(original_ids) * config.VAL_SPLIT)
    train_ids = original_ids[:train_count]
    val_ids = original_ids[train_count : train_count + val_count]
    test_ids = original_ids[train_count + val_count:]
    print(f"Chia dữ liệu: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test.")
    print("\n--- Bước 2: Chuẩn bị danh sách các tác vụ xử lý ---")
    tasks = []
    id_splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    for split_name, ids in id_splits.items():
        for video_id in tqdm(ids, desc=f"Chuẩn bị tác vụ cho bộ {split_name}"):
            found_original = False
            for dir_path in config.ORIGINAL_DIRS.values():
                original_path = os.path.join(dir_path, f'{video_id}.mp4')
                if os.path.exists(original_path):
                    output_path_real = os.path.join(config.PROCESSED_DATA_DIR, split_name, 'REAL', video_id)
                    tasks.append((original_path, output_path_real))
                    found_original = True
                    break
            for method, dir_path in config.MANIPULATION_DIRS.items():
                fake_paths = glob.glob(os.path.join(dir_path, f'{video_id}*.mp4'))
                for fake_path in fake_paths:
                    fake_name = os.path.basename(fake_path).split('.')[0]
                    output_path_fake = os.path.join(config.PROCESSED_DATA_DIR, split_name, 'FAKE', fake_name)
                    tasks.append((fake_path, output_path_fake))
    print(f"\nTổng số tác vụ xử lý (cả gốc và giả): {len(tasks)}")
    print("\n--- Bước 3: Bắt đầu xử lý đa luồng (tối ưu cho máy yếu) ---")
    if not tasks:
        print("Không có tác vụ nào để thực hiện. Dừng lại.")
        return
    
    # Giảm số processes cho máy yếu để tránh overload
    # Sử dụng NUM_WORKERS từ config nếu có, nếu không thì dùng giá trị mặc định
    num_workers = getattr(config, 'NUM_WORKERS', 2)
    num_processes = min(2, max(1, num_workers // 2))  # Giảm xuống 1/2
    print(f"Sử dụng {num_processes} luồng để xử lý (tối ưu cho máy yếu)...")
    
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # Sử dụng imap_unordered để bắt đầu nhận kết quả ngay lập tức
        results = list(tqdm(pool.imap_unordered(process_single_video, tasks), total=len(tasks), desc="Đang xử lý video"))
        pool.close()
    except KeyboardInterrupt:
        print("\n⚠️ Nhận tín hiệu dừng (Ctrl+C). Đang dừng các tiến trình con...")
        pool.terminate()
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình xử lý: {e}")
        pool.terminate()
    finally:
        pool.join()
        
    print("\n--- Hoàn tất quá trình tiền xử lý! ---")