import httpx
import os
import tempfile
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("WHATSAPP_TOKEN")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("✅ Groq Whisper loaded!")

async def download_audio(media_id: str) -> str:
    print(f"Downloading audio: {media_id}")

    url = f"https://graph.facebook.com/v22.0/{media_id}"
    headers = {"Authorization": f"Bearer {TOKEN}"}

    async with httpx.AsyncClient() as client:
        # Get media URL
        response = await client.get(url, headers=headers)
        media_data = response.json()

        if "url" not in media_data:
            print(f"❌ No URL in response: {media_data}")
            return ""

        media_url = media_data["url"]
        print(f"Media URL obtained")

        # Download audio
        audio_response = await client.get(
            media_url,
            headers=headers
        )
        print(f"Audio size: {len(audio_response.content)} bytes")

        # Save to temp file
        audio_path = f"audio_{media_id}.ogg"
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)

        print(f"Audio saved: {audio_path}")
        return audio_path

async def transcribe_audio(media_id: str) -> str:
    try:
        # Download audio
        audio_path = await download_audio(media_id)

        if not audio_path:
            return ""

        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found")
            return ""

        print(f"Transcribing with Groq Whisper...")

        # Transcribe using Groq API
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )

        # Delete audio file
        os.remove(audio_path)
        print("Audio file deleted")

        text = transcription.strip() if isinstance(transcription, str) else transcription
        print(f"✅ Transcribed: {text}")
        return text

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(f"audio_{media_id}.ogg"):
            os.remove(f"audio_{media_id}.ogg")
        return ""