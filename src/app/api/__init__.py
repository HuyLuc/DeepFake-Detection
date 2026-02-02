# src/app/api/__init__.py
"""
API endpoints for DeepFake Detection Web App V2.0
"""

from .routes import api
from .export_routes import export_api

__all__ = ['api', 'export_api']
