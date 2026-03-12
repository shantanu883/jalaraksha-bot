from whatsapp import send_buttons

# ── Language Menu ──
async def send_language_menu(to: str):
    await send_buttons(
        to=to,
        body=(
            "🙏 *Select your Language:*\n\n"
            "भाषा निवडा / Choose language"
        ),
        buttons=[
            {"id": "lang_mr", "title": "मराठी"},
            {"id": "lang_hi", "title": "हिंदी"},
            {"id": "lang_en", "title": "English"},
        ]
    )

async def send_more_languages(to: str):
    await send_buttons(
        to=to,
        body="🌐 *More Languages:*",
        buttons=[
            {"id": "lang_ta", "title": "தமிழ்"},
            {"id": "lang_te", "title": "తెలుగు"},
            {"id": "lang_other", "title": "Other →"},
        ]
    )

# ── Division Menu (Page 1) ──
async def send_division_menu(to: str):
    await send_buttons(
        to=to,
        body=(
            "📍 *Select your Division in Maharashtra:*\n\n"
            "Page 1 of 3"
        ),
        buttons=[
            {"id": "div_konkan", "title": "Konkan"},
            {"id": "div_pune", "title": "Pune"},
            {"id": "div_nashik", "title": "Nashik →"},
        ]
    )

async def send_division_menu_2(to: str):
    await send_buttons(
        to=to,
        body=(
            "📍 *Select your Division in Maharashtra:*\n\n"
            "Page 2 of 3"
        ),
        buttons=[
            {"id": "div_aurangabad", "title": "Aurangabad"},
            {"id": "div_amravati", "title": "Amravati"},
            {"id": "div_nagpur", "title": "Nagpur →"},
        ]
    )

async def send_division_menu_3(to: str):
    await send_buttons(
        to=to,
        body=(
            "📍 *Select your Division in Maharashtra:*\n\n"
            "Page 3 of 3"
        ),
        buttons=[
            {"id": "div_nagpur", "title": "Nagpur"},
            {"id": "div_more1", "title": "⬅ Back Pg 1"},
            {"id": "div_more2", "title": "⬅ Back Pg 2"},
        ]
    )

# ── District Menus per Division ──
async def send_konkan_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Konkan Division — Select District:*",
        buttons=[
            {"id": "dist_mumbai", "title": "Mumbai"},
            {"id": "dist_thane", "title": "Thane"},
            {"id": "dist_konkan_more", "title": "More →"},
        ]
    )

async def send_konkan_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Konkan Division — More Districts:*",
        buttons=[
            {"id": "dist_raigad", "title": "Raigad"},
            {"id": "dist_ratnagiri", "title": "Ratnagiri"},
            {"id": "dist_sindhudurg", "title": "Sindhudurg"},
        ]
    )

async def send_pune_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Pune Division — Select District:*",
        buttons=[
            {"id": "dist_pune", "title": "Pune"},
            {"id": "dist_satara", "title": "Satara"},
            {"id": "dist_pune_more", "title": "More →"},
        ]
    )

async def send_pune_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Pune Division — More Districts:*",
        buttons=[
            {"id": "dist_sangli", "title": "Sangli"},
            {"id": "dist_solapur", "title": "Solapur"},
            {"id": "dist_kolhapur", "title": "Kolhapur"},
        ]
    )

async def send_nashik_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Nashik Division — Select District:*",
        buttons=[
            {"id": "dist_nashik", "title": "Nashik"},
            {"id": "dist_dhule", "title": "Dhule"},
            {"id": "dist_nashik_more", "title": "More →"},
        ]
    )

async def send_nashik_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Nashik Division — More Districts:*",
        buttons=[
            {"id": "dist_nandurbar", "title": "Nandurbar"},
            {"id": "dist_jalgaon", "title": "Jalgaon"},
            {"id": "dist_ahmednagar", "title": "Ahmednagar"},
        ]
    )

async def send_aurangabad_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Aurangabad Division — Select District:*",
        buttons=[
            {"id": "dist_aurangabad", "title": "Aurangabad"},
            {"id": "dist_jalna", "title": "Jalna"},
            {"id": "dist_aurang_more", "title": "More →"},
        ]
    )

async def send_aurangabad_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Aurangabad Division — More Districts:*",
        buttons=[
            {"id": "dist_beed", "title": "Beed"},
            {"id": "dist_latur", "title": "Latur"},
            {"id": "dist_aurang_more2", "title": "More →"},
        ]
    )

async def send_aurangabad_districts_3(to: str):
    await send_buttons(
        to=to,
        body="📍 *Aurangabad Division — More Districts:*",
        buttons=[
            {"id": "dist_osmanabad", "title": "Osmanabad"},
            {"id": "dist_nanded", "title": "Nanded"},
            {"id": "dist_parbhani", "title": "Parbhani"},
        ]
    )

async def send_amravati_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Amravati Division — Select District:*",
        buttons=[
            {"id": "dist_amravati", "title": "Amravati"},
            {"id": "dist_akola", "title": "Akola"},
            {"id": "dist_amrav_more", "title": "More →"},
        ]
    )

async def send_amravati_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Amravati Division — More Districts:*",
        buttons=[
            {"id": "dist_yavatmal", "title": "Yavatmal"},
            {"id": "dist_washim", "title": "Washim"},
            {"id": "dist_buldhana", "title": "Buldhana"},
        ]
    )

async def send_nagpur_districts(to: str):
    await send_buttons(
        to=to,
        body="📍 *Nagpur Division — Select District:*",
        buttons=[
            {"id": "dist_nagpur", "title": "Nagpur"},
            {"id": "dist_wardha", "title": "Wardha"},
            {"id": "dist_nagpur_more", "title": "More →"},
        ]
    )

async def send_nagpur_districts_2(to: str):
    await send_buttons(
        to=to,
        body="📍 *Nagpur Division — More Districts:*",
        buttons=[
            {"id": "dist_chandrapur", "title": "Chandrapur"},
            {"id": "dist_gadchiroli", "title": "Gadchiroli"},
            {"id": "dist_nagpur_more2", "title": "More →"},
        ]
    )

async def send_nagpur_districts_3(to: str):
    await send_buttons(
        to=to,
        body="📍 *Nagpur Division — More Districts:*",
        buttons=[
            {"id": "dist_gondia", "title": "Gondia"},
            {"id": "dist_bhandara", "title": "Bhandara"},
            {"id": "dist_hingoli", "title": "Hingoli"},
        ]
    )

# ── Parse Functions ──
def parse_language_choice(text: str) -> str:
    text = text.strip().lower()
    lang_map = {
        "lang_mr": "mr", "lang_hi": "hi", "lang_en": "en",
        "lang_ta": "ta", "lang_te": "te",
        "1": "en", "2": "hi", "3": "mr",
        "english": "en", "hindi": "hi", "marathi": "mr",
        "tamil": "ta", "telugu": "te",
    }
    return lang_map.get(text, None)

def parse_district_choice(text: str) -> str:
    text = text.strip().lower()
    district_map = {
        "dist_mumbai": "Mumbai",
        "dist_mumbai_suburban": "Mumbai Suburban",
        "dist_thane": "Thane",
        "dist_raigad": "Raigad",
        "dist_ratnagiri": "Ratnagiri",
        "dist_sindhudurg": "Sindhudurg",
        "dist_pune": "Pune",
        "dist_satara": "Satara",
        "dist_sangli": "Sangli",
        "dist_solapur": "Solapur",
        "dist_kolhapur": "Kolhapur",
        "dist_nashik": "Nashik",
        "dist_dhule": "Dhule",
        "dist_nandurbar": "Nandurbar",
        "dist_jalgaon": "Jalgaon",
        "dist_ahmednagar": "Ahmednagar",
        "dist_aurangabad": "Aurangabad",
        "dist_jalna": "Jalna",
        "dist_beed": "Beed",
        "dist_latur": "Latur",
        "dist_osmanabad": "Osmanabad",
        "dist_nanded": "Nanded",
        "dist_parbhani": "Parbhani",
        "dist_hingoli": "Hingoli",
        "dist_amravati": "Amravati",
        "dist_akola": "Akola",
        "dist_yavatmal": "Yavatmal",
        "dist_washim": "Washim",
        "dist_buldhana": "Buldhana",
        "dist_nagpur": "Nagpur",
        "dist_wardha": "Wardha",
        "dist_chandrapur": "Chandrapur",
        "dist_gadchiroli": "Gadchiroli",
        "dist_gondia": "Gondia",
        "dist_bhandara": "Bhandara",
    }
    return district_map.get(text, None)

def get_confirmation_message(language: str) -> str:
    messages = {
        "en": (
            "✅ Language set to *English*\n\n"
            "You can report:\n"
            "1️⃣ Symptoms — fever, diarrhea\n"
            "2️⃣ Water issue — dirty water\n"
            "3️⃣ Animal death — cow, buffalo\n"
            "4️⃣ School absence — kids absent\n\n"
            "Type or send voice message 🙏\n"
            "_Type 'change language' anytime_"
        ),
        "hi": (
            "✅ भाषा *हिंदी* सेट हो गई\n\n"
            "आप रिपोर्ट कर सकते हैं:\n"
            "1️⃣ लक्षण — बुखार, दस्त\n"
            "2️⃣ पानी — गंदा पानी\n"
            "3️⃣ पशु मृत्यु — गाय, भैंस\n"
            "4️⃣ स्कूल — बच्चे अनुपस्थित\n\n"
            "टाइप करें या वॉइस मैसेज भेजें 🙏\n"
            "_'change language' टाइप करें_"
        ),
        "mr": (
            "✅ भाषा *मराठी* सेट झाली\n\n"
            "आपण रिपोर्ट करू शकता:\n"
            "1️⃣ लक्षणे — ताप, जुलाब\n"
            "2️⃣ पाणी समस्या — घाण पाणी\n"
            "3️⃣ प्राणी मृत्यू — गाय, म्हैस\n"
            "4️⃣ शाळा अनुपस्थिती\n\n"
            "टाइप करा किंवा व्हॉइस मेसेज पाठवा 🙏\n"
            "_'change language' टाइप करा_"
        ),
        "ta": (
            "✅ மொழி *தமிழ்* அமைக்கப்பட்டது\n\n"
            "நீங்கள் புகாரளிக்கலாம்:\n"
            "1️⃣ அறிகுறிகள் — காய்ச்சல்\n"
            "2️⃣ தண்ணீர் பிரச்சனை\n"
            "3️⃣ விலங்கு இறப்பு\n"
            "4️⃣ பள்ளி வருகையின்மை\n\n"
            "தட்டச்சு செய்யுங்கள் 🙏"
        ),
        "te": (
            "✅ భాష *తెలుగు* సెట్ చేయబడింది\n\n"
            "మీరు నివేదించవచ్చు:\n"
            "1️⃣ లక్షణాలు — జ్వరం\n"
            "2️⃣ నీటి సమస్య\n"
            "3️⃣ జంతువు మరణం\n"
            "4️⃣ పాఠశాల గైర్హాజరు\n\n"
            "టైప్ చేయండి 🙏"
        ),
    }
    return messages.get(language, messages["en"])