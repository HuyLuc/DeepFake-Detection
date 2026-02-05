
"""
LLM Service
Handles interactions with Google Gemini API for RAG
"""
import os
import google.generativeai as genai
from configs import config

class LLMService:
    def __init__(self):
        self.api_key = os.getenv('AI_API_KEY')
        self.provider = os.getenv('AI_PROVIDER', 'gemini')
        self.model = None
        self._setup()

    def _setup(self):
        if not self.api_key or 'YOUR_GEMINI_API_KEY' in self.api_key:
            print("⚠️ LLMService: No valid API Key found.")
            return

        if self.provider == 'gemini':
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_response(self, user_query, context_text):
        """
        Generate response using RAG context
        """
        if not self.model:
            return None

        prompt = f"""
        Role: You are a friendly Deepfake Security Expert Assistant.
        
        Context Information:
        {context_text}
        
        User Query: {user_query}
        
        Instructions:
        1. Answer the query based on the Context Information provided above.
        2. If the answer is not in the context, use your general knowledge about Deepfakes and Security.
        3. Be concise, helpful, and polite.
        4. If mentioning statistics, cite the numbers from context.
        5. Answer in Vietnamese.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None
