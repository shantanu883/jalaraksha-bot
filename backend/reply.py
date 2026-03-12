def generate_reply(intent: str, language: str = "en") -> str:

    replies = {
        "SYMPTOM": {
            "en": (
                "🚨 *Symptom Report Received!*\n\n"
                "✅ Report saved. ASHA worker notified.\n\n"
                "⚠️ Please:\n"
                "- Give ORS to affected persons\n"
                "- Boil all drinking water\n"
                "- Visit nearest health center\n\n"
                "Help is coming 🙏"
            ),
            "hi": (
                "🚨 *लक्षण रिपोर्ट प्राप्त हुई!*\n\n"
                "✅ रिपोर्ट सेव। ASHA कार्यकर्ता सूचित।\n\n"
                "⚠️ कृपया:\n"
                "- प्रभावित को ORS दें\n"
                "- पानी उबालकर पिएं\n"
                "- नजदीकी स्वास्थ्य केंद्र जाएं\n\n"
                "मदद आ रही है 🙏"
            ),
            "as": (
                "🚨 *ৰোগৰ বিৱৰণ পোৱা গৈছে!*\n\n"
                "✅ ৰিপোৰ্ট সংৰক্ষিত। ASHA কৰ্মী জাননী পাইছে।\n\n"
                "⚠️ অনুগ্ৰহ কৰি:\n"
                "- ORS দিয়ক\n"
                "- পানী উতলাই খাওক\n"
                "- স্বাস্থ্য কেন্দ্ৰলৈ যাওক\n\n"
                "সহায় আহিছে 🙏"
            )
        },
        "WATER": {
            "en": (
                "💧 *Water Issue Report Received!*\n\n"
                "✅ Report saved. Testing team notified.\n\n"
                "⚠️ Please:\n"
                "- Stop using this water immediately\n"
                "- Use only boiled water\n"
                "- Inform your neighbours\n\n"
                "Team visits within 24 hours 🙏"
            ),
            "hi": (
                "💧 *पानी की समस्या रिपोर्ट मिली!*\n\n"
                "✅ रिपोर्ट सेव। टीम सूचित।\n\n"
                "⚠️ कृपया:\n"
                "- यह पानी तुरंत बंद करें\n"
                "- केवल उबला पानी पिएं\n"
                "- पड़ोसियों को बताएं\n\n"
                "24 घंटे में टीम आएगी 🙏"
            ),
            "as": (
                "💧 *পানীৰ সমস্যাৰ ৰিপোৰ্ট পোৱা গৈছে!*\n\n"
                "✅ ৰিপোৰ্ট সংৰক্ষিত। দল জাননী পাইছে।\n\n"
                "⚠️ অনুগ্ৰহ কৰি:\n"
                "- এই পানী বন্ধ কৰক\n"
                "- উতলোৱা পানী খাওক\n\n"
                "24 ঘণ্টাত দল আহিব 🙏"
            )
        },
        "ANIMAL": {
            "en": (
                "🐄 *Animal Death Report Received!*\n\n"
                "✅ Report saved. Veterinary team notified.\n\n"
                "⚠️ Please:\n"
                "- Do not touch the animal\n"
                "- Keep children away\n"
                "- Wash hands thoroughly\n\n"
                "Help is coming 🙏"
            ),
            "hi": (
                "🐄 *पशु मृत्यु रिपोर्ट मिली!*\n\n"
                "✅ रिपोर्ट सेव। पशु चिकित्सा टीम सूचित।\n\n"
                "⚠️ कृपया:\n"
                "- जानवर को मत छुएं\n"
                "- बच्चों को दूर रखें\n"
                "- हाथ धोएं\n\n"
                "मदद आ रही है 🙏"
            ),
            "as": (
                "🐄 *পশু মৃত্যুৰ ৰিপোৰ্ট পোৱা গৈছে!*\n\n"
                "✅ ৰিপোৰ্ট সংৰক্ষিত। পশু চিকিৎসা দল জাননী পাইছে।\n\n"
                "⚠️ অনুগ্ৰহ কৰি:\n"
                "- জন্তু স্পৰ্শ নকৰিব\n"
                "- ল'ৰা-ছোৱালী দূৰত ৰাখক\n\n"
                "সহায় আহিছে 🙏"
            )
        },
        "SCHOOL": {
            "en": (
                "🏫 *School Absence Report Received!*\n\n"
                "✅ Report saved. BMO notified.\n\n"
                "⚠️ Please:\n"
                "- Monitor students for symptoms\n"
                "- Ensure clean water at school\n\n"
                "Thank you 🙏"
            ),
            "hi": (
                "🏫 *स्कूल अनुपस्थिति रिपोर्ट मिली!*\n\n"
                "✅ रिपोर्ट सेव। BMO सूचित।\n\n"
                "⚠️ कृपया:\n"
                "- बच्चों में लक्षण देखें\n"
                "- साफ पानी सुनिश्चित करें\n\n"
                "धन्यवाद 🙏"
            ),
            "as": (
                "🏫 *বিদ্যালয়ত অনুপস্থিতিৰ ৰিপোৰ্ট পোৱা গৈছে!*\n\n"
                "✅ ৰিপোৰ্ট সংৰক্ষিত। BMO জাননী পাইছে।\n\n"
                "⚠️ অনুগ্ৰহ কৰি:\n"
                "- শিক্ষাৰ্থী পৰীক্ষা কৰক\n\n"
                "ধন্যবাদ 🙏"
            )
        },
        "UNKNOWN": {
            "en": (
                "👋 *Welcome to Jalaraksha Health Bot!*\n\n"
                "You can report:\n"
                "1️⃣ Symptoms — fever, diarrhea\n"
                "2️⃣ Water issue — dirty water\n"
                "3️⃣ Animal death — cow, buffalo\n"
                "4️⃣ School absence — kids absent\n\n"
                "Just type what is happening 🙏"
            ),
            "hi": (
                "👋 *जलरक्षा हेल्थ बॉट में स्वागत है!*\n\n"
                "आप रिपोर्ट कर सकते हैं:\n"
                "1️⃣ लक्षण — बुखार, दस्त\n"
                "2️⃣ पानी — गंदा पानी\n"
                "3️⃣ पशु मृत्यु — गाय, भैंस\n"
                "4️⃣ स्कूल — बच्चे अनुपस्थित\n\n"
                "बस लिखें क्या हो रहा है 🙏"
            ),
            "as": (
                "👋 *জলৰক্ষা বটলৈ স্বাগতম!*\n\n"
                "আপুনি ৰিপোৰ্ট কৰিব পাৰে:\n"
                "1️⃣ ৰোগ — জ্বৰ, বমি\n"
                "2️⃣ পানী — লেতেৰা পানী\n"
                "3️⃣ পশু মৃত্যু\n"
                "4️⃣ বিদ্যালয় — অনুপস্থিতি\n\n"
                "কি হৈছে লিখক 🙏"
            )
        }
    }

    intent_replies = replies.get(intent, replies["UNKNOWN"])
    return intent_replies.get(language, intent_replies["en"])