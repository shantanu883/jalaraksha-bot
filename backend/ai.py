from google import genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini API Key loaded: {api_key[:20] if api_key else 'NOT FOUND'}")

client = genai.Client(api_key=api_key)
print("✅ Gemini AI connected!")

SYSTEM_PROMPT = """
You are JALARAKSHA, a caring AI health assistant for rural Maharashtra, India.
You help poor villagers who report waterborne diseases via WhatsApp.

RULES:
1. Detect intent from message:
   - SYMPTOM: fever, diarrhea, vomiting, stomach pain, cholera, typhoid, jaundice, loose motions, weakness, ताप, जुलाब, उलटी, पोटदुखी, बुखार, दस्त, उल्टी, ताप आहे, अंग दुखणे
   - WATER: dirty water, contaminated water, bad smell, गंदा पाणी, घाण पाणी, पाण्याला वास, पाणी खराब, नळाचे पाणी घाण
   - ANIMAL: cow died, buffalo died, गाय मेली, म्हैस मेली, जनावर मेले, गाय मर गई, जनावर मरण
   - SCHOOL: children absent, school closed, मुले अनुपस्थित, शाळा बंद, बच्चे अनुपस्थित, मुले शाळेत नाहीत
   - UNKNOWN: greetings, hi, hello, random messages, thanks, ok, test

2. Based on intent give specific advice:

   For SYMPTOM:
   - Express concern warmly
   - Tell them to drink ORS immediately
   - Tell them to drink only boiled water
   - Tell them to rest and eat light food like rice and dal
   - If symptoms are severe go to nearest hospital immediately
   - Give emergency number 108

   For WATER:
   - Express this is very serious
   - Tell them to stop using that water immediately
   - Tell them to boil all drinking water for 10 minutes
   - Tell them to inform their neighbors also
   - Tell them water testing team will visit soon
   - Give emergency number 108

   For ANIMAL:
   - Express this is a serious warning sign for disease outbreak
   - Tell them to keep children away from the dead animal
   - Tell them to not eat meat from that animal
   - Tell them to wash hands with soap thoroughly
   - Tell them health team will investigate soon
   - Give emergency number 108

   For SCHOOL:
   - Thank them for reporting this important information
   - Tell them to give ORS to sick children
   - Tell them health team will visit school soon
   - Tell them to keep sick children at home for now
   - Give emergency number 108

   For UNKNOWN:
   - Greet them warmly
   - Explain what JALARAKSHA does
   - Tell them they can report:
     * Symptoms like fever and loose motions
     * Dirty or contaminated water
     * Animal deaths like cow or buffalo
     * School absenteeism due to illness
   - Encourage them to report any health concern

3. Every response MUST have:
   - Warm acknowledgment of their message
   - 3 to 4 specific advice points with emojis
   - Confirmation that report is saved and team notified
   - Emergency number 108 at the end

4. Keep response maximum 8 lines
5. Use emojis to make it friendly and easy to read
6. Use simple language that uneducated villagers can understand
7. Never use medical jargon or complicated words

RESPOND ONLY IN THIS JSON FORMAT — NO MARKDOWN NO BACKTICKS:
{"intent": "SYMPTOM", "reply": "your reply here"}
"""

async def process_with_ai(message: str, language: str = "en",
                           district: str = None,
                           village: str = None) -> dict:
    try:
        print(f"\n🤖 Sending to Gemini: '{message}'")
        print(f"Language: {language} | District: {district} | Village: {village}")

        location_context = ""
        if district and village:
            location_context = f"User is from {village}, {district}, Maharashtra, India."
        elif district:
            location_context = f"User is from {district}, Maharashtra, India."

        language_instruction = {
            "en": "You MUST reply ONLY in English. Do not use any Hindi or Marathi words at all.",
            "hi": "You MUST reply ONLY in Hindi. Do not use any English or Marathi words at all.",
            "mr": "You MUST reply ONLY in Marathi. Do not use any Hindi or English words at all.",
            "ta": "You MUST reply ONLY in Tamil. Do not use any other language.",
            "te": "You MUST reply ONLY in Telugu. Do not use any other language.",
            "bn": "You MUST reply ONLY in Bengali. Do not use any other language.",
            "gu": "You MUST reply ONLY in Gujarati. Do not use any other language.",
            "kn": "You MUST reply ONLY in Kannada. Do not use any other language.",
            "ml": "You MUST reply ONLY in Malayalam. Do not use any other language.",
            "pa": "You MUST reply ONLY in Punjabi. Do not use any other language.",
        }.get(language, "You MUST reply ONLY in English.")

        prompt = f"""{SYSTEM_PROMPT}

{location_context}

User message: "{message}"

MOST IMPORTANT LANGUAGE RULE: {language_instruction}
This language rule overrides everything else. Always follow it strictly.

Return ONLY a raw JSON object. No markdown. No backticks. No extra text before or after.
Correct format: {{"intent": "SYMPTOM", "reply": "your advice here"}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        raw_text = response.text
        print(f"Raw Gemini response: {raw_text}")

        # Clean response — remove markdown if present
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Extract JSON object safely
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]

        print(f"Cleaned JSON: {text}")

        result = json.loads(text)
        intent = result.get("intent", "UNKNOWN")
        reply = result.get("reply", "")

        if not reply:
            raise ValueError("Empty reply from Gemini")

        print(f"✅ AI Intent: {intent}")
        print(f"✅ AI Reply: {reply[:100]}")

        return {"intent": intent, "reply": reply}

    except Exception as e:
        print(f"❌ Gemini error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "intent": "UNKNOWN",
            "reply": get_fallback_reply(language)
        }

def get_fallback_reply(language: str) -> str:
    fallbacks = {
        "en": (
            "🙏 Your message has been received!\n\n"
            "💊 Drink only boiled water\n"
            "💊 Take ORS solution if you have diarrhea\n"
            "💊 Rest and eat light food\n"
            "💊 If severe go to hospital immediately\n\n"
            "📍 Your report is saved. Health team notified.\n\n"
            "🚨 Emergency: 108"
        ),
        "hi": (
            "🙏 आपका संदेश मिल गया!\n\n"
            "💊 केवल उबला हुआ पानी पिएं\n"
            "💊 ORS घोल लें\n"
            "💊 आराम करें और हल्का खाना खाएं\n"
            "💊 गंभीर हो तो तुरंत अस्पताल जाएं\n\n"
            "📍 आपकी रिपोर्ट दर्ज हो गई. टीम को सूचित किया गया.\n\n"
            "🚨 आपातकाल: 108"
        ),
        "mr": (
            "🙏 तुमचा संदेश मिळाला!\n\n"
            "💊 फक्त उकळलेले पाणी प्या\n"
            "💊 ORS द्रावण घ्या\n"
            "💊 आराम करा आणि हलके जेवण करा\n"
            "💊 जास्त त्रास असेल तर ताबडतोब दवाखान्यात जा\n\n"
            "📍 तुमची तक्रार नोंदवली गेली. टीमला कळवले आहे.\n\n"
            "🚨 आपत्कालीन: 108"
        ),
        "ta": (
            "🙏 உங்கள் செய்தி கிடைத்தது!\n\n"
            "💊 கொதித்த தண்ணீர் மட்டும் குடிக்கவும்\n"
            "💊 ORS கரைசல் எடுக்கவும்\n"
            "💊 ஓய்வு எடுக்கவும்\n\n"
            "📍 உங்கள் புகார் பதிவு செய்யப்பட்டது.\n\n"
            "🚨 அவசரநிலை: 108"
        ),
        "te": (
            "🙏 మీ సందేశం అందింది!\n\n"
            "💊 మరిగించిన నీరు మాత్రమే తాగండి\n"
            "💊 ORS ద్రావణం తీసుకోండి\n"
            "💊 విశ్రాంతి తీసుకోండి\n\n"
            "📍 మీ నివేదిక సేవ్ చేయబడింది.\n\n"
            "🚨 అత్యవసర: 108"
        ),
    }
    return fallbacks.get(language, fallbacks["en"])