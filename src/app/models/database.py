# src/app/models/database.py
"""
Database setup with SQLAlchemy
SQLite database for history tracking
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from configs import config  # Import config để lấy BASE_DIR

# Initialize SQLAlchemy (will be attached to Flask app later)
db = SQLAlchemy()


class PredictionHistory(db.Model):
    """
    Model để lưu lịch sử predictions
    
    Attributes:
        id: Primary key
        created_at: Timestamp
        file_name: Tên file gốc
        file_type: 'image' or 'video'
        model_used: 'standard', 'advanced', or 'ensemble'
        verdict: 'FAKE' or 'REAL'
        confidence: Float 0-1
        fake_probability: Float 0-1
        real_probability: Float 0-1
        processing_time: Float (seconds)
        details_json: JSON string với additional details
        thumbnail_path: Path to saved thumbnail (optional)
    """
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # File info
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    file_size = db.Column(db.Integer, nullable=True)  # bytes
    
    # Prediction info
    model_used = db.Column(db.String(20), nullable=False)  # 'standard', 'advanced', 'ensemble'
    verdict = db.Column(db.String(10), nullable=False)  # 'FAKE' or 'REAL'
    confidence = db.Column(db.Float, nullable=False)
    fake_probability = db.Column(db.Float, nullable=True)
    real_probability = db.Column(db.Float, nullable=True)
    
    # Performance
    processing_time = db.Column(db.Float, nullable=True)  # seconds
    
    # Additional details (JSON string)
    details_json = db.Column(db.Text, nullable=True)
    
    # Optional: saved thumbnail for quick preview
    thumbnail_path = db.Column(db.String(500), nullable=True)
    
    # Video specific
    frames_analyzed = db.Column(db.Integer, nullable=True)
    fake_ratio = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f"<PredictionHistory {self.id}: {self.file_name} - {self.verdict}>"
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'model_used': self.model_used,
            'verdict': self.verdict,
            'confidence': self.confidence,
            'fake_probability': self.fake_probability,
            'real_probability': self.real_probability,
            'processing_time': self.processing_time,
            'details_json': self.details_json,
            'thumbnail_path': self.thumbnail_path,
            'frames_analyzed': self.frames_analyzed,
            'fake_ratio': self.fake_ratio
        }


def init_db(app):
    """
    Initialize database with Flask app
    
    Args:
        app: Flask application instance
    """
    # Configure SQLite database using BASE_DIR from config
    # Sử dụng thư mục data/ ở root dự án
    db_path = os.path.join(config.BASE_DIR, 'data', 'history.db')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
        # In thông báo để debug
        # print(f"✅ Database initialized at: {db_path}")
    
    return db