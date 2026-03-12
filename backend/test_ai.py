from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Try different models
models_to_try = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemma-3-4b-it",
]

for model_name in models_to_try:
    print(f"\n🧪 Testing model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents='Say hello in Marathi. Return ONLY JSON: {"intent": "UNKNOWN", "reply": "hello in marathi"}'
        )
        print(f"✅ Works! Response: {response.text[:100]}")
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")