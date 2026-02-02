# src/app/api/export_routes.py
"""
Export API Routes
Endpoints for exporting predictions as PDF/JSON
"""

import os
import logging
from flask import Blueprint, request, jsonify, send_file

from ..services.export_service import ExportService
from ..services.history_service import HistoryService

logger = logging.getLogger(__name__)

# Create Blueprint
export_api = Blueprint('export_api', __name__, url_prefix='/api/export')

# Lazy initialization
_export_service = None
_history_service = None


def get_export_service():
    """Lazy load ExportService"""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service


def get_history_service():
    """Lazy load HistoryService"""
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================

@export_api.route('/json/<int:prediction_id>', methods=['GET'])
def export_json(prediction_id):
    """
    Export prediction as JSON file
    
    Args:
        prediction_id: ID of prediction in history
    
    Query params:
        download: 'true' to download file, 'false' to get file path (default: true)
    
    Returns:
        JSON file download or file path
    """
    try:
        # Get prediction from history
        history_service = get_history_service()
        prediction = history_service.get_by_id(prediction_id)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': f'Prediction {prediction_id} not found'
            }), 404
        
        # Export
        export_service = get_export_service()
        prediction_data = prediction.to_dict()
        
        filepath = export_service.export_json(
            prediction_data=prediction_data,
            filename=f"prediction_{prediction_id}"
        )
        
        # Return download or path
        download = request.args.get('download', 'true').lower() == 'true'
        
        if download:
            return send_file(
                filepath,
                mimetype='application/json',
                as_attachment=True,
                download_name=os.path.basename(filepath)
            )
        else:
            return jsonify({
                'success': True,
                'file_path': filepath,
                'file_name': os.path.basename(filepath)
            })
        
    except Exception as e:
        logger.error(f"❌ Export JSON error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@export_api.route('/pdf/<int:prediction_id>', methods=['GET'])
def export_pdf(prediction_id):
    """
    Export prediction as PDF report
    
    Args:
        prediction_id: ID of prediction in history
    
    Query params:
        download: 'true' to download file, 'false' to get file path (default: true)
    
    Returns:
        PDF file download or file path
    """
    try:
        # Get prediction from history
        history_service = get_history_service()
        prediction = history_service.get_by_id(prediction_id)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': f'Prediction {prediction_id} not found'
            }), 404
        
        # Export
        export_service = get_export_service()
        prediction_data = prediction.to_dict()
        
        filepath = export_service.export_pdf(
            prediction_data=prediction_data,
            filename=f"report_{prediction_id}"
        )
        
        # Return download or path
        download = request.args.get('download', 'true').lower() == 'true'
        
        if download:
            return send_file(
                filepath,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=os.path.basename(filepath)
            )
        else:
            return jsonify({
                'success': True,
                'file_path': filepath,
                'file_name': os.path.basename(filepath)
            })
        
    except ImportError as e:
        return jsonify({
            'success': False,
            'error': 'PDF export requires reportlab. Run: pip install reportlab'
        }), 500
    except Exception as e:
        logger.error(f"❌ Export PDF error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@export_api.route('/direct', methods=['POST'])
def export_direct():
    """
    Export prediction data directly (without saving to history first)
    
    Request body:
        {
            "prediction_data": {...},  // Prediction result
            "format": "json" or "pdf"
        }
    
    Returns:
        File download
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        prediction_data = data.get('prediction_data')
        export_format = data.get('format', 'json')
        
        if not prediction_data:
            return jsonify({
                'success': False,
                'error': 'prediction_data is required'
            }), 400
        
        if export_format not in ['json', 'pdf']:
            return jsonify({
                'success': False,
                'error': 'format must be "json" or "pdf"'
            }), 400
        
        # Export
        export_service = get_export_service()
        filepath = export_service.export(prediction_data, format=export_format)
        
        # Return file
        mimetype = 'application/json' if export_format == 'json' else 'application/pdf'
        return send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=True,
            download_name=os.path.basename(filepath)
        )
        
    except ImportError as e:
        return jsonify({
            'success': False,
            'error': 'PDF export requires reportlab. Run: pip install reportlab'
        }), 500
    except Exception as e:
        logger.error(f"❌ Export error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@export_api.route('/formats', methods=['GET'])
def get_export_formats():
    """Get available export formats"""
    return jsonify({
        'success': True,
        'formats': [
            {
                'id': 'json',
                'name': 'JSON',
                'description': 'Complete data export in JSON format',
                'mime_type': 'application/json'
            },
            {
                'id': 'pdf',
                'name': 'PDF Report',
                'description': 'Formatted report with tables and styling',
                'mime_type': 'application/pdf',
                'requires': 'reportlab'
            }
        ]
    })
