from ai import process_with_ai
from fastapi import FastAPI, Request
from whatsapp import send_message
from intent import detect_intent
from reply import generate_reply
from voice import transcribe_audio
from sessions import (
    get_user_language, set_user_language,
    is_new_user, reset_user,
    set_user_district, set_user_village,
    get_user_location
)
from menu import (
    send_language_menu, send_more_languages,
    send_division_menu, send_division_menu_2,
    send_konkan_districts, send_konkan_districts_2,
    send_pune_districts, send_pune_districts_2,
    send_nashik_districts, send_nashik_districts_2,
    send_aurangabad_districts, send_aurangabad_districts_2,
    send_aurangabad_districts_3,
    send_amravati_districts, send_amravati_districts_2,
    send_nagpur_districts, send_nagpur_districts_2,
    send_nagpur_districts_3,
    parse_language_choice, parse_district_choice,
    get_confirmation_message
)
from firebase_db import save_report
from alerts import check_and_alert
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Jalaraksha Bot is Running!"}

@app.get("/reset/{phone}")
async def reset_session(phone: str):
    reset_user(phone)
    return {"status": f"Session reset for {phone}"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    verify_token = os.getenv("VERIFY_TOKEN")
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == verify_token:
        print("✅ Webhook verified!")
        return int(challenge)
    print("❌ Verification failed!")
    return {"error": "Invalid token"}

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    print("Incoming data:", data)

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        message = value["messages"][0]
        phone = message["from"]
        msg_type = message["type"]

        print(f"Message type: {msg_type} from {phone}")

        # ── STEP 1: Extract text ──
        if msg_type == "text":
            text = message["text"]["body"]
            print(f"Text: {text}")

        elif msg_type == "interactive":
            interactive = message["interactive"]
            itype = interactive["type"]
            if itype == "button_reply":
                text = interactive["button_reply"]["id"]
                print(f"Button: {text}")
            elif itype == "list_reply":
                text = interactive["list_reply"]["id"]
                print(f"List: {text}")
            else:
                text = ""

        elif msg_type == "audio":
            media_id = message["audio"]["id"]
            text = await transcribe_audio(media_id)
            if not text:
                await send_message(
                    phone,
                    "❌ Could not understand voice.\n"
                    "Please type your message 🙏"
                )
                return {"status": "ok"}
            print(f"Transcribed: {text}")

        else:
            await send_message(phone, "Please send text or voice message 🙏")
            return {"status": "ok"}

        # ── STEP 2: New user → show division menu ──
        if is_new_user(phone):
            await send_message(
                phone,
                "🙏 *Welcome to JALARAKSHA!*\n\n"
                "Early Warning System for Waterborne Diseases\n\n"
                "📍 Select your Division in Maharashtra 👇"
            )
            await send_division_menu(phone)
            set_user_language(phone, "selecting_division")
            return {"status": "ok"}

        current = get_user_language(phone)
        print(f"Current status: {current}")

        # ── STEP 3: Division selection ──
        if current == "selecting_division":
            if text == "div_konkan":
                await send_konkan_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_pune":
                await send_pune_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_nashik":
                await send_nashik_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_aurangabad":
                await send_aurangabad_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_amravati":
                await send_amravati_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_nagpur":
                await send_nagpur_districts(phone)
                set_user_language(phone, "selecting_district")
            elif text == "div_more2":
                await send_division_menu_2(phone)
            else:
                await send_division_menu(phone)
            return {"status": "ok"}

        # ── STEP 4: District more pages ──
        if current == "selecting_district":
            # More buttons
            if text == "dist_konkan_more":
                await send_konkan_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_pune_more":
                await send_pune_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_nashik_more":
                await send_nashik_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_aurang_more":
                await send_aurangabad_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_aurang_more2":
                await send_aurangabad_districts_3(phone)
                return {"status": "ok"}
            if text == "dist_amrav_more":
                await send_amravati_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_nagpur_more":
                await send_nagpur_districts_2(phone)
                return {"status": "ok"}
            if text == "dist_nagpur_more2":
                await send_nagpur_districts_3(phone)
                return {"status": "ok"}

            # Actual district selected
            district = parse_district_choice(text)
            if district:
                set_user_district(phone, district)
                await send_message(
                    phone,
                    f"✅ District: *{district}*\n\n"
                    f"🏘️ Now type your *Village / Taluka* name:"
                )
                set_user_language(phone, "selecting_village")
            else:
                await send_message(
                    phone,
                    "❌ Please select your district 👆"
                )
            return {"status": "ok"}

        # ── STEP 5: Village input ──
        if current == "selecting_village":
            set_user_village(phone, text)
            location = get_user_location(phone)
            await send_message(
                phone,
                f"✅ Location saved!\n"
                f"📍 District: *{location.get('district', '')}*\n"
                f"🏘️ Village: *{text}*\n\n"
                f"Now select your language 👇"
            )
            await send_language_menu(phone)
            set_user_language(phone, "selecting")
            return {"status": "ok"}

        # ── STEP 6: Language selection ──
        if current == "selecting":
            if text == "lang_more":
                await send_more_languages(phone)
                return {"status": "ok"}

            language = parse_language_choice(text)
            if language:
                set_user_language(phone, language)
                await send_message(
                    phone,
                    get_confirmation_message(language)
                )
                print(f"Language set: {language}")
            else:
                await send_message(
                    phone,
                    "❌ Please select a language 👆"
                )
                await send_language_menu(phone)
            return {"status": "ok"}

        # ── STEP 7: Change triggers ──
        if text.strip().lower() in [
            "change language", "language",
            "bhasha badlo", "भाषा बदलो"
        ]:
            set_user_language(phone, "selecting")
            await send_language_menu(phone)
            return {"status": "ok"}

        if text.strip().lower() in [
            "change location", "location",
            "change village", "गाव बदला"
        ]:
            await send_message(
                phone,
                "📍 Let's update your location.\n"
                "Select your Division 👇"
            )
            await send_division_menu(phone)
            set_user_language(phone, "selecting_division")
            return {"status": "ok"}

        # ── STEP 8: Normal report processing with AI ──
        language = get_user_language(phone)
        print(f"Processing in language: {language}")

        # Use Gemini AI to detect intent and generate reply
        # Use Gemini AI to detect intent and generate reply
        from ai import process_with_ai
        location = get_user_location(phone)
        result = await process_with_ai(
            text,
            language,
            district=location.get("district"),
            village=location.get("village")
        )

        intent = result["intent"]
        reply = result["reply"]

        print(f"AI Intent: {intent}")
        print(f"AI Reply: {reply}")

        # Save to Firebase
        location = get_user_location(phone)
        save_report(
            phone, text, intent, language,
            district=location.get("district"),
            village=location.get("village")
        )

        # Check if alert needed
        await check_and_alert(phone, intent)

        # Send AI generated reply
        await send_message(phone, reply)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return {"status": "ok"}
