# src/app/api/deepfake_gen_routes.py
"""
API Routes cho tính năng Tạo Ảnh Giả Mạo (Deepfake Generator Demo).
Blueprint: /api/deepfake-gen
"""

import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# Blueprint
deepfake_gen_bp = Blueprint('deepfake_gen', __name__, url_prefix='/api/deepfake-gen')

# Lazy load service
_generator_service = None

def get_generator_service():
    """Lazy load DeepfakeGeneratorService (tránh tải nặng khi khởi động app)."""
    global _generator_service
    if _generator_service is None:
        from ..services.deepfake_generator_service import DeepfakeGeneratorService
        _generator_service = DeepfakeGeneratorService()
    return _generator_service


# =============================================================================
# API ENDPOINTS
# =============================================================================

@deepfake_gen_bp.route('/generate', methods=['POST'])
def generate_deepfakes():
    """
    Tạo 4 ảnh giả mạo từ 1 ảnh gốc.
    ---
    tags:
      - Deepfake Generator
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Ảnh chân dung để tạo ảnh giả mạo
    responses:
      200:
        description: Trả về 4 ảnh fake dạng base64
      400:
        description: Lỗi đầu vào (thiếu file, sai định dạng)
      500:
        description: Lỗi server
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Vui lòng tải lên một tệp ảnh.'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Chưa chọn tệp nào.'
            }), 400
        
        # Kiểm tra định dạng ảnh
        allowed_exts = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in allowed_exts:
            return jsonify({
                'success': False, 
                'error': f'Định dạng không hỗ trợ. Cho phép: {", ".join(allowed_exts)}'
            }), 400
        
        # Đọc bytes ảnh
        image_bytes = file.read()
        
        if len(image_bytes) == 0:
            return jsonify({
                'success': False,
                'error': 'Tệp ảnh trống.'
            }), 400
        
        # Gọi service tạo ảnh fake
        service = get_generator_service()
        result = service.generate_all(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Deepfake generation error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500


@deepfake_gen_bp.route('/methods', methods=['GET'])
def get_methods():
    """
    Lấy danh sách 4 phương pháp tạo ảnh giả mạo và mô tả.
    ---
    tags:
      - Deepfake Generator
    responses:
      200:
        description: Danh sách phương pháp
    """
    try:
        service = get_generator_service()
        methods = service.get_methods()
        return jsonify({
            'success': True,
            'methods': methods
        })
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
