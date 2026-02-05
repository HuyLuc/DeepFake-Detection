
"""
Chat API Routes
"""
from flask import Blueprint, jsonify, request
from ..services.chat_service import ChatService

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
chat_service = ChatService()

@chat_bp.route('/message', methods=['POST'])
def send_message():
    """
    Send a message to the chatbot
    ---
    tags:
      - Chat
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            message:
              type: string
    responses:
      200:
        description: Bot response
        schema:
          type: object
          properties:
            success:
              type: boolean
            response:
              type: string
            related_items:
              type: array
    """
    try:
        data = request.json
        user_msg = data.get('message', '')
        
        if not user_msg:
            return jsonify({'success': False, 'error': 'Message required'}), 400
            
        response_text, related_items = chat_service.get_response(user_msg)
        
        return jsonify({
            'success': True,
            'response': response_text,
            'related_items': related_items
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500
