
import os
import google.generativeai as genai
from configs import config
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('AI_API_KEY')
print(f"Key loaded: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

if not api_key:
    print("❌ No API Key found in .env")
    exit()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, are you working?")
    print(f"✅ Response from Gemini: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
