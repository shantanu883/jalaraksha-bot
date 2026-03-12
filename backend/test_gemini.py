import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:20] if api_key else 'NOT FOUND'}")

genai.configure(api_key=api_key)

# List available models
print("\nAvailable models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content('Say hello in Marathi as JSON: {"intent": "UNKNOWN", "reply": "hello"}')
print("\nTest response:")
print(response.text)