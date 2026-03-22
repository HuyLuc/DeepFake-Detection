# src/app/services/deepfake_generator_service.py
"""
Deepfake Generator Service - Giả lập 4 phương pháp FaceForensics++
Sử dụng OpenCV + MediaPipe để tạo ảnh giả mạo cho mục đích demo & kiểm thử.

4 phương pháp được mô phỏng:
1. DeepFakes     — Smooth/blur vùng mặt + blend biên mặt (autoencoder artifacts)
2. Face2Face     — Warp biểu cảm + noise vùng miệng/mắt (reenactment artifacts)
3. FaceSwap      — Lật & hoán đổi vùng mặt + blend viền cứng (geometry artifacts)
4. NeuralTextures — Thêm texture pattern + color jitter lên da (neural rendering artifacts)
"""

import cv2
import numpy as np
import base64
import logging
import os

logger = logging.getLogger(__name__)

# Cố gắng import MediaPipe để phát hiện khuôn mặt chính xác hơn
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    logger.info("✅ MediaPipe available for face detection")
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("⚠️ MediaPipe not installed, using OpenCV Haar Cascade fallback")


class DeepfakeGeneratorService:
    """
    Service tạo ảnh giả mạo theo 4 phương pháp FaceForensics++.
    Hoạt động offline hoàn toàn bằng OpenCV + MediaPipe (nếu có).
    """
    
    # Mô tả từng phương pháp (để hiển thị trên UI)
    METHODS = {
        'deepfakes': {
            'name': 'DeepFakes',
            'description': 'Mô phỏng lỗi Autoencoder: Làm mờ vùng mặt, mất chi tiết da, blend viền không tự nhiên.',
            'icon': '🤖'
        },
        'face2face': {
            'name': 'Face2Face',
            'description': 'Mô phỏng tái tạo biểu cảm: Biến dạng nhẹ vùng miệng/mắt, texture không khớp.',
            'icon': '🎭'
        },
        'faceswap': {
            'name': 'FaceSwap',
            'description': 'Mô phỏng hoán đổi khuôn mặt: Viền mặt cứng, ánh sáng/màu da không đồng nhất.',
            'icon': '🔄'
        },
        'neuraltextures': {
            'name': 'NeuralTextures',
            'description': 'Mô phỏng Neural Rendering: Kết cấu da bất thường, pattern lạ trên bề mặt.',
            'icon': '🧠'
        }
    }
    
    def __init__(self):
        """Khởi tạo service, chuẩn bị face detector."""
        
        # =====================================================================
        # KHỞI TẠO BỘ PHÁT HIỆN KHUÔN MẶT
        # MediaPipe chính xác hơn (trả về 468 facial landmarks chi tiết).
        # OpenCV Haar Cascade là fallback khi không cài được MediaPipe.
        # =====================================================================
        if HAS_MEDIAPIPE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            logger.info("🟢 DeepfakeGeneratorService initialized with MediaPipe")
        else:
            # Fallback: dùng Haar Cascade có sẵn trong OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("🟡 DeepfakeGeneratorService initialized with OpenCV Cascade")
    
    # =========================================================================
    # HÀM CHÍNH: TẠO 4 ẢNH GIẢ TỪ 1 ẢNH GỐC
    # =========================================================================
    def generate_all(self, image_bytes):
        """
        Nhận ảnh gốc dạng bytes, trả về dict gồm 4 ảnh fake (base64).
        
        Args:
            image_bytes: bytes của ảnh gốc (từ upload)
        
        Returns:
            dict: {
                'success': True/False,
                'original': base64 ảnh gốc,
                'results': [
                    {'method': 'deepfakes', 'name': ..., 'image': base64, ...},
                    ...
                ]
            }
        """
        try:
            # Đọc ảnh từ bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {'success': False, 'error': 'Không thể đọc được ảnh. Vui lòng thử lại với file khác.'}
            
            # Resize nếu ảnh quá lớn (tối đa 1024px chiều dài nhất)
            h, w = img.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # Phát hiện vùng khuôn mặt
            face_rect, landmarks = self._detect_face(img)
            
            if face_rect is None:
                return {'success': False, 'error': 'Không tìm thấy khuôn mặt trong ảnh. Vui lòng tải ảnh chân dung rõ mặt.'}
            
            # Encode ảnh gốc sang base64
            original_b64 = self._encode_base64(img)
            
            # Tạo 4 ảnh fake bằng 4 phương pháp
            results = []
            methods = [
                ('deepfakes', self._apply_deepfakes),
                ('face2face', self._apply_face2face),
                ('faceswap', self._apply_faceswap),
                ('neuraltextures', self._apply_neuraltextures),
            ]
            
            for method_id, method_func in methods:
                try:
                    fake_img = method_func(img.copy(), face_rect, landmarks)
                    fake_b64 = self._encode_base64(fake_img)
                    
                    info = self.METHODS[method_id]
                    results.append({
                        'method': method_id,
                        'name': info['name'],
                        'description': info['description'],
                        'icon': info['icon'],
                        'image': fake_b64
                    })
                except Exception as e:
                    logger.error(f"❌ Error in method {method_id}: {e}")
                    results.append({
                        'method': method_id,
                        'name': self.METHODS[method_id]['name'],
                        'description': f'Lỗi khi tạo: {str(e)}',
                        'icon': '❌',
                        'image': original_b64  # Fallback về ảnh gốc
                    })
            
            return {
                'success': True,
                'original': original_b64,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ DeepfakeGenerator error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # PHÁT HIỆN KHUÔN MẶT VÀ LANDMARKS
    # =========================================================================
    def _detect_face(self, img):
        """
        Phát hiện khuôn mặt và facial landmarks trong ảnh.
        Trả về (face_rect, landmarks) hoặc (None, None) nếu không tìm thấy.
        
        face_rect: (x, y, w, h) bounding box khuôn mặt
        landmarks: list các điểm (x, y) trên mặt (nếu có MediaPipe)
        """
        h, w = img.shape[:2]
        
        if HAS_MEDIAPIPE:
            # Dùng MediaPipe Face Detection + Face Mesh cho landmarks
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Bước 1: Detect face bounding box
            with self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0=close range, 1=far range
                min_detection_confidence=0.5
            ) as face_detection:
                results = face_detection.process(rgb)
                
                if not results.detections:
                    return None, None
                
                # Lấy khuôn mặt lớn nhất
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                fw = min(w - x, int(bbox.width * w))
                fh = min(h - y, int(bbox.height * h))
                face_rect = (x, y, fw, fh)
            
            # Bước 2: Lấy landmarks chi tiết (468 điểm)
            landmarks = []
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks[0].landmark:
                        landmarks.append((int(lm.x * w), int(lm.y * h)))
            
            return face_rect, landmarks if landmarks else None
            
        else:
            # Fallback: OpenCV Haar Cascade
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            if len(faces) == 0:
                return None, None
            
            # Lấy khuôn mặt lớn nhất
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face_rect = tuple(faces[0])
            
            return face_rect, None  # Cascade không có landmarks chi tiết
    
    # =========================================================================
    # PHƯƠNG PHÁP 1: DEEPFAKES (Autoencoder Artifacts)
    # =========================================================================
    def _apply_deepfakes(self, img, face_rect, landmarks):
        """
        Mô phỏng hiệu ứng DeepFakes:
        - Smooth/blur mạnh vùng mặt (giống lỗi autoencoder mất chi tiết)
        - Thay đổi tone màu da (color shift mạnh) 
        - Blend viền mặt không tự nhiên (boundary artifact)
        - Compression artifacts (JPEG noise kép)
        """
        x, y, w, h = face_rect
        result = img.copy()
        
        # Trích xuất vùng mặt
        face_roi = result[y:y+h, x:x+w].copy()
        
        # BƯỚC 1: Blur mạnh vùng mặt (mô phỏng autoencoder mất chi tiết lỗ chân lông)
        # Dùng bilateral filter mạnh + Gaussian blur lớn để mô phỏng decoder output
        face_blurred = cv2.bilateralFilter(face_roi, 21, 120, 120)
        face_blurred = cv2.GaussianBlur(face_blurred, (11, 11), 5)
        
        # BƯỚC 2: Color shift mạnh (mô phỏng khác biệt ánh sáng khi decode)
        hsv = cv2.cvtColor(face_blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 15) % 180  # Dịch Hue rõ rệt
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.30, 0, 255)  # Tăng Saturation mạnh
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.90, 0, 255)  # Giảm Brightness
        face_colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # BƯỚC 3: Tạo mask elip cho vùng mặt (blend không hoàn hảo = artifact)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.42), int(h * 0.48))  # Elip hơi nhỏ hơn vùng mặt
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Blur mask nhưng KHÔNG quá mịn => tạo viền blend cứng đặc trưng DeepFakes
        mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
        mask_3ch = cv2.merge([mask_blurred, mask_blurred, mask_blurred]) / 255.0
        
        # BƯỚC 4: Blend vùng mặt fake lên ảnh gốc
        blended_face = (face_colored * mask_3ch + face_roi * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended_face
        
        # BƯỚC 5: Compression artifacts kép (JPEG noise mạnh)
        # Lần 1: nén quality thấp
        result = self._apply_jpeg_compression(result, quality=40)
        # Lần 2: nén lại để tạo double-compression artifact đặc trưng deepfake
        result = self._apply_jpeg_compression(result, quality=55)
        
        return result
    
    # =========================================================================
    # PHƯƠNG PHÁP 2: FACE2FACE (Reenactment Artifacts)
    # =========================================================================
    def _apply_face2face(self, img, face_rect, landmarks):
        """
        Mô phỏng hiệu ứng Face2Face:
        - Warp biến dạng mạnh vùng miệng (mô phỏng reenactment biểu cảm)
        - Thêm noise texture mạnh vào vùng mắt/miệng
        - Thay đổi contrast cục bộ rõ rệt (lighting inconsistency)
        """
        x, y, w, h = face_rect
        result = img.copy()
        
        face_roi = result[y:y+h, x:x+w].copy()
        
        # BƯỚC 0: Bilateral blur mạnh (mô phỏng autoencoder smoothing đặc trưng deepfake)
        face_roi = cv2.bilateralFilter(face_roi, 19, 100, 100)
        face_roi = cv2.GaussianBlur(face_roi, (9, 9), 4)
        
        # BƯỚC 1: Biến dạng vùng miệng bằng phép biến đổi affine
        # Chia vùng mặt thành phần trên và dưới. Phần dưới (miệng) bị warp nhiều hơn.
        mouth_y_start = int(h * 0.55)  # Vùng miệng bắt đầu từ 55% chiều cao mặt
        mouth_region = face_roi[mouth_y_start:, :].copy()
        
        if mouth_region.shape[0] > 10 and mouth_region.shape[1] > 10:
            mh, mw = mouth_region.shape[:2]
            
            # Tạo displacement map để warp (mô phỏng expression transfer)
            map_x = np.zeros((mh, mw), dtype=np.float32)
            map_y = np.zeros((mh, mw), dtype=np.float32)
            
            for iy in range(mh):
                for ix in range(mw):
                    # Displacement mạnh hơn: miệng bị kéo xuống + sang ngang rõ rệt
                    offset_x = 6 * np.sin(2 * np.pi * iy / mh)
                    offset_y = 4 * np.sin(2 * np.pi * ix / mw)
                    map_x[iy, ix] = ix + offset_x
                    map_y[iy, ix] = iy + offset_y
            
            warped_mouth = cv2.remap(mouth_region, map_x, map_y, cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_REFLECT)
            face_roi[mouth_y_start:, :] = warped_mouth
        
        # BƯỚC 2: Thêm noise texture mạnh vào vùng mắt (rendering imperfection rõ rệt)
        eye_y_end = int(h * 0.45)
        eye_region = face_roi[:eye_y_end, :].copy()
        noise = np.random.normal(0, 20, eye_region.shape).astype(np.float32)
        eye_noisy = np.clip(eye_region.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        face_roi[:eye_y_end, :] = eye_noisy
        
        # BƯỚC 3: Thay đổi contrast cục bộ mạnh (lighting không khớp rõ rệt)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.15 + 10, 0, 255)  # Tăng L (luminance) mạnh
        face_roi = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # BƯỚC 4: Blend mặt đã biến dạng quay lại ảnh gốc
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.44), int(h*0.46)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        original_face = result[y:y+h, x:x+w]
        blended = (face_roi * mask_3ch + original_face * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended
        
        # BƯỚC 5: Compression artifacts kép (giống DeepFakes)
        result = self._apply_jpeg_compression(result, quality=40)
        result = self._apply_jpeg_compression(result, quality=55)
        
        return result
    
    # =========================================================================
    # PHƯƠNG PHÁP 3: FACESWAP (Geometry-based Artifacts)
    # =========================================================================
    def _apply_faceswap(self, img, face_rect, landmarks):
        """
        Mô phỏng hiệu ứng FaceSwap:
        - Lật ngang vùng mặt (simulate đổi mặt từ người khác)
        - Thay đổi kích thước/scale (geometric mismatch)
        - Blend viền cứng rõ ràng (hard edge boundary)
        - Color mismatch mạnh giữa vùng mặt và cổ/tóc
        """
        x, y, w, h = face_rect
        result = img.copy()
        
        face_roi = result[y:y+h, x:x+w].copy()
        
        # BƯỚC 0: Bilateral blur mạnh (mô phỏng autoencoder smoothing)
        face_roi_blurred = cv2.bilateralFilter(face_roi, 19, 100, 100)
        face_roi_blurred = cv2.GaussianBlur(face_roi_blurred, (9, 9), 4)
        
        # BƯỚC 1: Lật ngang khuôn mặt (mô phỏng việc lấy mặt người khác dán vào)
        face_flipped = cv2.flip(face_roi_blurred, 1)
        
        # BƯỚC 2: Scale nhẹ (geometric mismatch - mặt bị to/nhỏ so với khung đầu)
        scale_factor = 0.93
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        face_scaled = cv2.resize(face_flipped, (new_w, new_h))
        
        # Đặt lại vào center
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        face_padded = face_roi.copy()
        face_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = face_scaled
        
        # BƯỚC 3: Color mismatch mạnh (mặt swap khác tone rõ rệt so với da cổ/tai)
        hsv = cv2.cvtColor(face_padded, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] - 12) % 180   # Dịch hue mạnh
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.70, 0, 255)  # Giảm saturation mạnh
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)  # Tăng brightness rõ
        face_colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # BƯỚC 4: Tạo mask CỨNG (hard edge) - đặc trưng FaceSwap
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.40), int(h*0.45)), 0, 0, 360, 255, -1)
        # Chỉ blur rất nhẹ (viền cứng hơn so với DeepFakes)
        mask = cv2.GaussianBlur(mask, (9, 9), 4)
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        # BƯỚC 5: Blend lên ảnh gốc
        original_face = result[y:y+h, x:x+w]
        blended = (face_colored * mask_3ch + original_face * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended
        
        # BƯỚC 6: Compression artifacts kép (giống DeepFakes)
        result = self._apply_jpeg_compression(result, quality=40)
        result = self._apply_jpeg_compression(result, quality=55)
        
        return result
    
    # =========================================================================
    # PHƯƠNG PHÁP 4: NEURALTEXTURES (Neural Rendering Artifacts)
    # =========================================================================
    def _apply_neuraltextures(self, img, face_rect, landmarks):
        """
        Mô phỏng hiệu ứng NeuralTextures:
        - Thêm texture pattern mạnh trên bề mặt da (neural rendering imperfections)
        - Color jitter mạnh vùng mặt (rendering color inconsistency)
        - Frequency noise mạnh (high-frequency artifacts từ neural network output)
        - Giữ nguyên hình dạng (không warp) nhưng thay đổi kết cấu bề mặt rõ rệt
        """
        x, y, w, h = face_rect
        result = img.copy()
        
        face_roi = result[y:y+h, x:x+w].copy()
        
        # BƯỚC 0: Bilateral blur MẠNH (mô phỏng neural rendering smoothing)
        # Blur mạnh trước, rồi thêm texture lên trên — giữ được signature smoothing
        face_roi = cv2.bilateralFilter(face_roi, 21, 120, 120)
        face_roi = cv2.GaussianBlur(face_roi, (11, 11), 5)
        
        # BƯỚC 1: Tạo texture pattern (mô phỏng neural rendering output)
        # Giảm noise amplitude để không counteract blur quá mạnh
        noise_fine = np.random.normal(0, 15, face_roi.shape).astype(np.float32)
        noise_coarse = cv2.GaussianBlur(
            np.random.normal(0, 25, face_roi.shape).astype(np.float32), 
            (15, 15), 5
        )
        texture_noise = noise_fine * 0.5 + noise_coarse * 0.5
        
        face_textured = np.clip(face_roi.astype(np.float32) + texture_noise, 0, 255).astype(np.uint8)
        
        # BƯỚC 2: Color jitter mạnh (neural renderer không duy trì được chính xác tone màu)
        # Dịch Hue tương tự DeepFakes để model nhận ra
        hsv = cv2.cvtColor(face_textured, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 12) % 180  # Dịch Hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.20, 0, 255)  # Tăng Saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0, 255)  # Giảm Brightness
        face_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # BƯỚC 3: Giữ lại chút sharpening nhẹ (neural net ringing effect)
        kernel_sharpen = np.array([[-1, -1, -1],
                                    [-1,  10.0, -1],
                                    [-1, -1, -1]]) / 1.5
        face_sharpened = cv2.filter2D(face_jittered, -1, kernel_sharpen)
        face_sharpened = np.clip(face_sharpened, 0, 255).astype(np.uint8)
        
        # Blend: giữ tỷ lệ nhiều phần blurred hơn
        face_final = cv2.addWeighted(face_jittered, 0.65, face_sharpened, 0.35, 0)
        
        # BƯỚC 3.5: Smoothing lại sau khi blend (giữ signature blur cho model nhận diện)
        face_final = cv2.GaussianBlur(face_final, (5, 5), 2)
        
        # BƯỚC 4: Tạo mask hình elip cho blend mịn
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.43), int(h*0.47)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 12)
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        # BƯỚC 5: Blend lên ảnh gốc
        original_face = result[y:y+h, x:x+w]
        blended = (face_final * mask_3ch + original_face * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended
        
        # BƯỚC 6: Compression artifacts kép (giống DeepFakes)
        result = self._apply_jpeg_compression(result, quality=38)
        result = self._apply_jpeg_compression(result, quality=55)
        
        return result
    
    # =========================================================================
    # TIỆN ÍCH
    # =========================================================================
    def _apply_jpeg_compression(self, img, quality=50):
        """
        Áp dụng JPEG compression để tạo compression artifacts.
        Giúp mô phỏng hiệu ứng nén đặc trưng trong ảnh deepfake.
        
        Args:
            img: numpy array (BGR)
            quality: JPEG quality (0-100, thấp = nhiều artifact hơn)
        
        Returns:
            numpy array đã qua compression
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def _encode_base64(self, img):
        """Chuyển đổi ảnh OpenCV (numpy array) sang chuỗi base64 (PNG)."""
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_methods(self):
        """Trả về danh sách 4 phương pháp với mô tả."""
        return list(self.METHODS.values())
