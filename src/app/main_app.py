# src/app/main_app.py

from flask import Flask, request, jsonify, render_template
from typing import Tuple, Optional
import torch
import timm
from torchvision import transforms
from PIL import Image
import io
import cv2
import os
import tempfile
import mediapipe as mp
import logging

# Import từ các file khác trong dự án
from configs import config
from src.utils.utils import load_checkpoint

# --- 1. KHỞI TẠO ỨNG DỤNG VÀ CÁC THÀNH PHẦN CỐ ĐỊNH ---
# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("--- 🚀 Khởi tạo ứng dụng và tải mô hình ---")
app = Flask(__name__, template_folder='templates')

# Tải mô hình MỘT LẦN DUY NHẤT khi ứng dụng khởi động
# Sử dụng config.DEVICE thay vì hardcode CPU
device = torch.device(config.DEVICE)
num_classes = len(config.CLASS_NAMES)
model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=num_classes)
best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'model_best.pth.tar')

try:
    model, _, _, _ = load_checkpoint(best_model_path, model)
    model = model.to(device)
    model.eval()
    logger.info(f"✅ Mô hình đã được tải và sẵn sàng trên {device}!")
except Exception as e:
    logger.error(f"❌ Lỗi khi tải mô hình: {e}", exc_info=True)
    raise

# Các phép biến đổi cho ảnh đầu vào
inference_transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Khởi tạo MediaPipe Face Detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=getattr(config, 'FACE_DETECTION_MODEL', 1),
    min_detection_confidence=getattr(config, 'FACE_DETECTION_CONFIDENCE', 0.5)
)

# Sử dụng CLASS_NAMES từ config thay vì hardcode
CLASS_NAMES = config.CLASS_NAMES

# --- 2. CÁC HÀM XỬ LÝ LOGIC ---

def predict_single_face(face_image: Image.Image) -> Tuple[str, float]:
    """Dự đoán trên MỘT ảnh khuôn mặt đã được cắt.
    
    Args:
        face_image: PIL Image của khuôn mặt đã được cắt
        
    Returns:
        Tuple (predicted_class, confidence_score)
    """
    image_tensor = inference_transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
    
    # Lấy tên lớp và điểm số tin cậy
    predicted_class = CLASS_NAMES[preds.item()]
    confidence_score = confidence.item()
    return predicted_class, confidence_score

# --- 3. ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN (ROUTES) ---

@app.route('/', methods=['GET'])
def index():
    """Render trang chủ."""
    return render_template('index.html')

def validate_video_file(file) -> Tuple[bool, Optional[str]]:
    """
    Kiểm tra tính hợp lệ của file video.
    
    Args:
        file: File object từ Flask request
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Kiểm tra tên file
    if not file.filename:
        return False, "Chưa chọn file nào"
    
    # Kiểm tra extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Định dạng file không được hỗ trợ. Chỉ chấp nhận: {', '.join(config.ALLOWED_VIDEO_EXTENSIONS)}"
    
    # Kiểm tra kích thước file
    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)  # Reset file pointer
    
    if file_size_mb > config.MAX_VIDEO_SIZE_MB:
        return False, f"File quá lớn. Kích thước tối đa: {config.MAX_VIDEO_SIZE_MB}MB"
    
    return True, None

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được gửi lên'}), 400
    
    file = request.files['file']
    
    # Validation input
    is_valid, error_msg = validate_video_file(file)
    if not is_valid:
        logger.warning(f"Invalid file upload: {error_msg}")
        return jsonify({'error': error_msg}), 400

    # Lưu file video tạm thời để xử lý
    temp_video_path = None
    try:
        # Bug fix: Ghi trực tiếp vào file handle đã mở thay vì dùng file.save()
        # để tránh file locking error trên Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            # Đọc toàn bộ nội dung file và ghi vào temporary file
            file.seek(0)  # Đảm bảo file pointer ở đầu
            tfile.write(file.read())
            temp_video_path = tfile.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            # Bug fix: Release VideoCapture trước khi return để tránh file lock error trên Windows
            cap.release()
            return jsonify({'error': 'Không thể mở file video. Vui lòng kiểm tra định dạng video.'}), 500

        # Kiểm tra video có hợp lệ không
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return jsonify({'error': 'Video không hợp lệ hoặc không có frame nào'}), 400

        fake_evidence_count = 0
        real_evidence_count = 0
        frames_with_face = 0
        total_processed_frames = 0
        final_verdict = "REAL"  # Mặc định là REAL
        strongest_fake_confidence = 0.0
        fake_confidences = []
        
        # Tối ưu: Chỉ xử lý mỗi N frame thay vì tất cả để tiết kiệm tài nguyên
        frame_count = 0
        skip_frames = config.SKIP_FRAMES

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:  # Bỏ qua frame
                continue
            
            total_processed_frames += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(image_rgb)

            if results.detections:
                frames_with_face += 1  # Chỉ đếm khi thực sự có face
                detection = results.detections[0]  # Chỉ lấy mặt rõ nhất
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                margin = config.FACE_MARGIN
                face_img = frame[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
                
                if face_img.size != 0:
                    pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    pred_class, confidence = predict_single_face(pil_face)

                    if pred_class == 'FAKE' and confidence >= config.EVIDENCE_THRESHOLD:
                        fake_evidence_count += 1
                        fake_confidences.append(confidence)
                        if confidence > strongest_fake_confidence:
                            strongest_fake_confidence = confidence
                    elif pred_class == 'REAL':
                        real_evidence_count += 1

        cap.release()

        # Logic quyết định cuối cùng - cải thiện
        if frames_with_face == 0:
            return jsonify({
                'error': 'Không phát hiện được khuôn mặt nào trong video. Vui lòng thử video khác.'
            }), 400
        
        # Tính tỷ lệ fake evidence
        fake_ratio = fake_evidence_count / frames_with_face if frames_with_face > 0 else 0
        
        # Quyết định dựa trên tỷ lệ và số lượng evidence
        # Cần ít nhất 30% frames có fake evidence hoặc có ít nhất 3 frames fake với confidence cao
        if fake_ratio >= 0.3 or (fake_evidence_count >= 3 and strongest_fake_confidence >= 0.85):
            final_verdict = "FAKE"
            avg_confidence = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0
            reason = (
                f"Phát hiện {fake_evidence_count}/{frames_with_face} khung hình có dấu hiệu giả mạo "
                f"(tỷ lệ: {fake_ratio*100:.1f}%, độ tin cậy trung bình: {avg_confidence*100:.2f}%)."
            )
        else:
            reason = (
                f"Không tìm thấy bằng chứng giả mạo rõ ràng. "
                f"Đã phân tích {frames_with_face} khung hình có khuôn mặt "
                f"({real_evidence_count} REAL, {fake_evidence_count} FAKE)."
            )
        
        logger.info(f"Prediction completed: {final_verdict}, processed {frames_with_face} frames with faces")
        
        return jsonify({
            'verdict': final_verdict,
            'reason': reason,
            'processed_frames': frames_with_face,
            'total_frames': total_processed_frames,
            'fake_evidence_count': fake_evidence_count,
            'real_evidence_count': real_evidence_count
        })

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        return jsonify({'error': f'Lỗi khi xử lý video: {str(e)}'}), 500
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)  # Xóa file tạm

def run_app():
    # Chạy app ở chế độ debug=False khi triển khai thực tế
    # Lưu ý: debug=True chỉ dùng cho development, không dùng trong production
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)