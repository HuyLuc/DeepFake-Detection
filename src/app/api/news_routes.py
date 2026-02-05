
"""
News API Routes
"""

from flask import Blueprint, jsonify
from ..services.news_service import NewsService

news_bp = Blueprint('news', __name__, url_prefix='/api/news')
news_service = NewsService()

@news_bp.route('/', methods=['GET'])
def get_news():
    """
    Get latest deepfake news
    ---
    tags:
      - News
    responses:
      200:
        description: List of news items
        schema:
          type: object
          properties:
            success:
              type: boolean
            items:
              type: array
              items:
                type: object
                properties:
                    title:
                        type: string
                    link:
                        type: string
                    source:
                        type: string
                    published:
                        type: string
    """
    try:
        items = news_service.get_news()
        return jsonify({
            'success': True,
            'items': items
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
