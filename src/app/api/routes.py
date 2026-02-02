# src/app/api/routes.py
"""
API Routes for DeepFake Detection Web App V2.0
RESTful endpoints for prediction, history, and export
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

from ..services.prediction_service import PredictionService
from ..services.history_service import HistoryService

logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# Lazy initialization of services
_prediction_service = None
_history_service = None


def get_prediction_service():
    """Lazy load PredictionService"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def get_history_service():
    """Lazy load HistoryService"""
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service


# =============================================================================
# CONFIGURATION
# =============================================================================
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def allowed_file(filename, file_type='any'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    else:
        return ext in ALLOWED_IMAGE_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS


def get_file_type(filename):
    """Determine file type from extension"""
    if '.' not in filename:
        return None
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return 'image'
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return 'video'
    return None


# =============================================================================
# PREDICTION ENDPOINTS
# =============================================================================

@api.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request:
        - file: File upload (image or video)
        - model: 'standard', 'advanced', or 'ensemble' (default: 'standard')
        - save_history: 'true' or 'false' (default: 'true')
    
    Response:
        {
            "success": bool,
            "verdict": "FAKE" or "REAL",
            "confidence": float,
            "probabilities": {"FAKE": float, "REAL": float},
            "model_used": str,
            "processing_time": float,
            "file_info": {...},
            "details": {...},
            "history_id": int (if save_history)
        }
    """
    try:
        # Validate file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: images ({", ".join(ALLOWED_IMAGE_EXTENSIONS)}) and videos ({", ".join(ALLOWED_VIDEO_EXTENSIONS)})'
            }), 400
        
        # Determine file type
        file_type = get_file_type(file.filename)
        if not file_type:
            return jsonify({
                'success': False,
                'error': 'Could not determine file type'
            }), 400
        
        # Get parameters
        model_choice = request.form.get('model', 'standard')
        if model_choice not in ['standard', 'advanced', 'ensemble']:
            return jsonify({
                'success': False,
                'error': 'Invalid model choice. Use: standard, advanced, or ensemble'
            }), 400
        
        save_history = request.form.get('save_history', 'true').lower() == 'true'
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = current_app.config.get('UPLOAD_FOLDER', tempfile.gettempdir())
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        
        logger.info(f"📁 File saved: {temp_path} ({file_size} bytes)")
        
        try:
            # Run prediction
            prediction_service = get_prediction_service()
            result = prediction_service.predict(
                file_path=temp_path,
                file_type=file_type,
                model_choice=model_choice
            )
            
            # Add file info
            result['file_info'] = {
                'file_name': filename,
                'file_type': file_type,
                'file_size': file_size
            }
            
            # Save to history if requested
            history_id = None
            if save_history and result.get('success', False):
                try:
                    history_service = get_history_service()
                    history = history_service.save_prediction(
                        prediction_result=result,
                        file_info=result['file_info']
                    )
                    history_id = history.id
                    result['history_id'] = history_id
                except Exception as e:
                    logger.warning(f"Could not save to history: {e}")
            
            return jsonify(result)
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"🗑️ Temp file removed: {temp_path}")
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/predict/image', methods=['POST'])
def predict_image():
    """
    Predict endpoint specifically for images
    Same as /predict but validates image file type
    """
    # Add validation for image only
    if 'file' in request.files:
        file = request.files['file']
        if file.filename and not allowed_file(file.filename, 'image'):
            return jsonify({
                'success': False,
                'error': f'Invalid image type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
            }), 400
    
    # Delegate to main predict
    return predict()


@api.route('/predict/video', methods=['POST'])
def predict_video():
    """
    Predict endpoint specifically for videos
    Same as /predict but validates video file type
    """
    # Add validation for video only
    if 'file' in request.files:
        file = request.files['file']
        if file.filename and not allowed_file(file.filename, 'video'):
            return jsonify({
                'success': False,
                'error': f'Invalid video type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
            }), 400
    
    # Delegate to main predict
    return predict()


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================

@api.route('/history', methods=['GET'])
def get_history():
    """
    Get prediction history with pagination and filters
    
    Query params:
        - page: int (default: 1)
        - per_page: int (default: 10, max: 100)
        - file_type: 'image' or 'video' (optional)
        - verdict: 'FAKE' or 'REAL' (optional)
        - model: 'standard', 'advanced', or 'ensemble' (optional)
        - sort_by: column name (default: 'created_at')
        - sort_order: 'asc' or 'desc' (default: 'desc')
    
    Response:
        {
            "success": true,
            "items": [...],
            "total": int,
            "page": int,
            "per_page": int,
            "pages": int
        }
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        file_type = request.args.get('file_type')
        verdict = request.args.get('verdict')
        model_used = request.args.get('model')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        history_service = get_history_service()
        result = history_service.get_history(
            page=page,
            per_page=per_page,
            file_type=file_type,
            verdict=verdict,
            model_used=model_used,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"❌ History error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/history/<int:prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """
    Get single prediction by ID
    """
    try:
        history_service = get_history_service()
        prediction = history_service.get_by_id(prediction_id)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': f'Prediction {prediction_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'prediction': prediction.to_dict()
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/history/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """
    Delete single prediction by ID
    """
    try:
        history_service = get_history_service()
        success = history_service.delete_prediction(prediction_id)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'Prediction {prediction_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Prediction {prediction_id} deleted'
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/history', methods=['DELETE'])
def clear_history():
    """
    Clear all prediction history
    
    ⚠️ WARNING: This is destructive!
    
    Request body:
        {"confirm": true}
    """
    try:
        data = request.get_json() or {}
        
        if not data.get('confirm', False):
            return jsonify({
                'success': False,
                'error': 'Please confirm by sending {"confirm": true}'
            }), 400
        
        history_service = get_history_service()
        count = history_service.clear_history()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {count} predictions from history'
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get overall statistics
    
    Response:
        {
            "success": true,
            "statistics": {
                "total_predictions": int,
                "fake_count": int,
                "real_count": int,
                "fake_ratio": float,
                "by_model": {...},
                "by_file_type": {...},
                "avg_confidence": float,
                "avg_processing_time": float
            }
        }
    """
    try:
        history_service = get_history_service()
        stats = history_service.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
    })


@api.route('/models', methods=['GET'])
def get_available_models():
    """Get available models info"""
    return jsonify({
        'success': True,
        'models': [
            {
                'id': 'standard',
                'name': 'Standard Model',
                'description': 'EfficientNet-B4 - Fast single frame analysis',
                'recommended_for': 'Quick checks, single images'
            },
            {
                'id': 'advanced',
                'name': 'Advanced Model',
                'description': 'EfficientNet-B4 + LSTM - Temporal analysis',
                'recommended_for': 'Videos, high accuracy requirements'
            },
            {
                'id': 'ensemble',
                'name': 'Ensemble Model',
                'description': 'Combination of Standard and Advanced',
                'recommended_for': 'Maximum accuracy, critical analysis'
            }
        ]
    })


@api.route('/supported-formats', methods=['GET'])
def get_supported_formats():
    """Get supported file formats"""
    return jsonify({
        'success': True,
        'formats': {
            'images': list(ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(ALLOWED_VIDEO_EXTENSIONS),
            'max_file_size': MAX_FILE_SIZE,
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024)
        }
    })
