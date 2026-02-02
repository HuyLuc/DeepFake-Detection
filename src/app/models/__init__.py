# src/app/models/__init__.py
"""
Database models for history tracking
"""

from .database import db, PredictionHistory, init_db

__all__ = ['db', 'PredictionHistory', 'init_db']
