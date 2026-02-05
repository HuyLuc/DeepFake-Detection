
"""
Chat Service
Handles chatbot logic, fuzzy pattern matching, and stats integration.
"""
import random
import difflib
from .news_service import NewsService
from .history_service import HistoryService
from .llm_service import LLMService
from ..models.database import PredictionHistory

class ChatService:
    def __init__(self):
        self.news_service = NewsService()
        self.history_service = HistoryService()
        self.llm_service = LLMService()
        
        # Knowledge Base with Keywords mapping
        self.knowledge_base = {
            # ... (Existing keywords kept for hybrid fallback) ...
             "hello": {
                "keywords": ['xin chào', 'hi', 'hello', 'chào', 'bạn là ai'],
                "responses": [
                    "Xin chào! Tôi là AI Assistant. Tôi có thể giúp gì cho bạn về Deepfake?",
                    "Chào bạn! Tôi trực sẵn sàng để hỗ trợ bạn.",
                    "Hello! Bạn cần kiểm tra video hay tìm hiểu thông tin?"
                ]
            },
        }
        
        self.default_responses = [
            "Xin lỗi, tôi chưa hiểu rõ ý bạn. Bạn thử hỏi: 'Deepfake là gì' hoặc 'Tin mới' xem?",
        ]

    def _get_system_context(self):
        """Retrieve dynamic context for RAG"""
        context = []
        
        # 1. Stats Context
        try:
            stats = PredictionHistory.query.with_entities(PredictionHistory.verdict).all()
            total = len(stats)
            fake_count = sum(1 for s in stats if s.verdict == 'FAKE')
            context.append(f"System Stats: Total {total} files scanned. {fake_count} FAKES detected.")
        except:
            pass
            
        # 2. News Context
        try:
            news_items = self.news_service.get_news(limit=3)
            news_titles = [f"- {item['title']}" for item in news_items]
            context.append("Latest News:\n" + "\n".join(news_titles))
        except:
            pass
            
        return "\n".join(context)

    def get_response(self, message):
        message = message.lower().strip()
        
        # 1. Try LLM first (Smart RAG)
        context = self._get_system_context()
        llm_response = self.llm_service.generate_response(message, context)
        if llm_response:
             # Check if news should be attached
            related = []
            if 'tin tức' in message or 'news' in message:
                 related = self.news_service.get_news(limit=3)
            return llm_response, related

        # 2. Hybrid Fallback (Original Logic)
        # Only reached if LLM fails (e.g. no API key) or errors
        
        # ... (Existing Logic below) ...
        return self._fallback_logic(message)

    def _fallback_logic(self, message):
        # Fallback to simple keyword matching if AI fails
        for intent, data in self.knowledge_base.items():
            for kw in data['keywords']:
                if kw in message:
                    return random.choice(data['responses']), []
        
        return "Xin lỗi, kết nối AI đang chập chờn (Error 404/500). Vui lòng kiểm tra lại API Key hoặc mạng internet.", []

