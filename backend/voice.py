import whisper
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded!")

TOKEN = os.getenv("WHATSAPP_TOKEN")

async def download_audio(media_id: str) -> str:
    print(f"Downloading audio: {media_id}")
    print(f"Token starts with: {TOKEN[:20] if TOKEN else 'NONE'}")

    url = f"https://graph.facebook.com/v22.0/{media_id}"
    headers = {"Authorization": f"Bearer {TOKEN}"}

    async with httpx.AsyncClient() as client:
        # Step 1 - Get media URL
        print("Getting media URL...")
        response = await client.get(url, headers=headers)
        print(f"Media URL response: {response.status_code}")
        print(f"Media URL data: {response.text}")

        media_data = response.json()

        if "url" not in media_data:
            print(f"ERROR: No URL in response: {media_data}")
            return ""

        media_url = media_data["url"]
        print(f"Media URL obtained: {media_url[:50]}")

        # Step 2 - Download audio
        print("Downloading audio file...")
        audio_response = await client.get(
            media_url,
            headers=headers
        )
        print(f"Audio download status: {audio_response.status_code}")
        print(f"Audio size: {len(audio_response.content)} bytes")

        # Step 3 - Save locally
        audio_path = f"audio_{media_id}.ogg"
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)

        print(f"Audio saved to: {audio_path}")

    return audio_path

async def transcribe_audio(media_id: str) -> str:
    try:
        # Download audio
        audio_path = await download_audio(media_id)

        if not audio_path:
            print("ERROR: Audio download failed")
            return ""

        # Check file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found: {audio_path}")
            return ""

        print(f"Transcribing: {audio_path}")

        # Transcribe
        result = model.transcribe(
            audio_path,
            task="transcribe"
        )

        text = result["text"].strip()
        detected_lang = result["language"]

        print(f"Transcribed text: {text}")
        print(f"Detected language: {detected_lang}")

        # Delete audio file
        os.remove(audio_path)
        print("Audio file deleted")

        return text

    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""