# src/app/services/history_service.py
"""
HistoryService: Manage prediction history
CRUD operations for PredictionHistory model
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..models.database import db, PredictionHistory

logger = logging.getLogger(__name__)


class HistoryService:
    """
    Service để quản lý prediction history
    
    Methods:
        - save_prediction: Lưu kết quả prediction mới
        - get_history: Lấy list predictions (với pagination)
        - get_by_id: Lấy 1 prediction theo ID
        - delete_prediction: Xóa 1 prediction
        - clear_history: Xóa toàn bộ history
        - get_statistics: Thống kê tổng quan
    """
    
    def __init__(self):
        logger.info("✅ HistoryService initialized")
    
    def save_prediction(self, prediction_result: Dict, file_info: Dict) -> PredictionHistory:
        """
        Lưu kết quả prediction vào database
        
        Args:
            prediction_result: Dict từ PredictionService.predict()
            file_info: Dict với file_name, file_size, file_type
        
        Returns:
            PredictionHistory instance
        """
        try:
            # Extract probabilities
            probabilities = prediction_result.get('probabilities', {})
            
            # Create record
            history = PredictionHistory(
                file_name=file_info.get('file_name', 'unknown'),
                file_type=file_info.get('file_type', 'unknown'),
                file_size=file_info.get('file_size'),
                model_used=prediction_result.get('model_used', 'unknown'),
                verdict=prediction_result.get('verdict', 'UNKNOWN'),
                confidence=prediction_result.get('confidence', 0.0),
                fake_probability=probabilities.get('FAKE'),
                real_probability=probabilities.get('REAL'),
                processing_time=prediction_result.get('processing_time'),
                details_json=json.dumps(prediction_result.get('details', {})),
                frames_analyzed=prediction_result.get('stats', {}).get('total_frames'),
                fake_ratio=prediction_result.get('stats', {}).get('fake_ratio')
            )
            
            db.session.add(history)
            db.session.commit()
            
            logger.info(f"✅ Saved prediction history: ID={history.id}")
            return history
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error saving prediction: {e}", exc_info=True)
            raise
    
    def get_history(
        self,
        page: int = 1,
        per_page: int = 10,
        file_type: Optional[str] = None,
        verdict: Optional[str] = None,
        model_used: Optional[str] = None,
        sort_by: str = 'created_at',
        sort_order: str = 'desc'
    ) -> Dict:
        """
        Lấy danh sách predictions với pagination và filters
        
        Args:
            page: Số trang (1-indexed)
            per_page: Số items mỗi trang
            file_type: Filter theo 'image' or 'video'
            verdict: Filter theo 'FAKE' or 'REAL'
            model_used: Filter theo model
            sort_by: Column để sort
            sort_order: 'asc' or 'desc'
        
        Returns:
            {
                'items': [...],
                'total': int,
                'page': int,
                'per_page': int,
                'pages': int
            }
        """
        try:
            query = PredictionHistory.query
            
            # Apply filters
            if file_type:
                query = query.filter(PredictionHistory.file_type == file_type)
            if verdict:
                query = query.filter(PredictionHistory.verdict == verdict)
            if model_used:
                query = query.filter(PredictionHistory.model_used == model_used)
            
            # Apply sorting
            sort_column = getattr(PredictionHistory, sort_by, PredictionHistory.created_at)
            if sort_order == 'desc':
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            
            # Paginate
            pagination = query.paginate(page=page, per_page=per_page, error_out=False)
            
            return {
                'items': [item.to_dict() for item in pagination.items],
                'total': pagination.total,
                'page': pagination.page,
                'per_page': pagination.per_page,
                'pages': pagination.pages
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting history: {e}", exc_info=True)
            raise
    
    def get_by_id(self, prediction_id: int) -> Optional[PredictionHistory]:
        """
        Lấy 1 prediction theo ID
        
        Args:
            prediction_id: ID của prediction
        
        Returns:
            PredictionHistory or None
        """
        try:
            return PredictionHistory.query.get(prediction_id)
        except Exception as e:
            logger.error(f"❌ Error getting prediction {prediction_id}: {e}")
            return None
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Xóa 1 prediction
        
        Args:
            prediction_id: ID của prediction
        
        Returns:
            True if success, False otherwise
        """
        try:
            prediction = PredictionHistory.query.get(prediction_id)
            if prediction:
                db.session.delete(prediction)
                db.session.commit()
                logger.info(f"✅ Deleted prediction: ID={prediction_id}")
                return True
            return False
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error deleting prediction {prediction_id}: {e}")
            return False
    
    def clear_history(self) -> int:
        """
        Xóa toàn bộ history
        
        Returns:
            Số records đã xóa
        """
        try:
            count = PredictionHistory.query.delete()
            db.session.commit()
            logger.info(f"✅ Cleared {count} predictions from history")
            return count
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error clearing history: {e}")
            return 0
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê tổng quan
        
        Returns:
            {
                'total_predictions': int,
                'fake_count': int,
                'real_count': int,
                'fake_ratio': float,
                'by_model': {'standard': int, 'advanced': int, 'ensemble': int},
                'by_file_type': {'image': int, 'video': int},
                'avg_confidence': float,
                'avg_processing_time': float
            }
        """
        try:
            from sqlalchemy import func
            
            total = PredictionHistory.query.count()
            
            if total == 0:
                return {
                    'total_predictions': 0,
                    'fake_count': 0,
                    'real_count': 0,
                    'fake_ratio': 0.0,
                    'by_model': {'standard': 0, 'advanced': 0, 'ensemble': 0},
                    'by_file_type': {'image': 0, 'video': 0},
                    'avg_confidence': 0.0,
                    'avg_processing_time': 0.0
                }
            
            fake_count = PredictionHistory.query.filter_by(verdict='FAKE').count()
            real_count = PredictionHistory.query.filter_by(verdict='REAL').count()
            
            # By model
            by_model = {
                'standard': PredictionHistory.query.filter_by(model_used='standard').count(),
                'advanced': PredictionHistory.query.filter_by(model_used='advanced').count(),
                'ensemble': PredictionHistory.query.filter_by(model_used='ensemble').count()
            }
            
            # By file type
            by_file_type = {
                'image': PredictionHistory.query.filter_by(file_type='image').count(),
                'video': PredictionHistory.query.filter_by(file_type='video').count()
            }
            
            # Averages
            avg_confidence = db.session.query(func.avg(PredictionHistory.confidence)).scalar() or 0.0
            avg_processing_time = db.session.query(func.avg(PredictionHistory.processing_time)).scalar() or 0.0
            
            return {
                'total_predictions': total,
                'fake_count': fake_count,
                'real_count': real_count,
                'fake_ratio': fake_count / total if total > 0 else 0.0,
                'by_model': by_model,
                'by_file_type': by_file_type,
                'avg_confidence': round(avg_confidence, 4),
                'avg_processing_time': round(avg_processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}", exc_info=True)
            return {}
