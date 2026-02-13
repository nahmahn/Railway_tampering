"""Quick test to verify Gemini API key is valid."""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

print(f"API Key: {api_key[:10]}...{api_key[-4:]}" if api_key else "❌ No API key found!")
print(f"Model:   {model_name}")
print("-" * 40)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Say 'hello' in one word.")
    print(f"✅ SUCCESS! Response: {response.text.strip()}")
except Exception as e:
    print(f"❌ FAILED: {e}")
