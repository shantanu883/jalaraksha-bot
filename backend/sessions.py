user_sessions = {}
user_locations = {}

def get_user_language(phone: str) -> str:
    return user_sessions.get(phone, None)

def set_user_language(phone: str, language: str):
    user_sessions[phone] = language
    print(f"Language set for {phone}: {language}")

def is_new_user(phone: str) -> bool:
    return phone not in user_sessions

def reset_user(phone: str):
    if phone in user_sessions:
        del user_sessions[phone]
    if phone in user_locations:
        del user_locations[phone]
    print(f"Session reset for {phone}")

def set_user_district(phone: str, district: str):
    if phone not in user_locations:
        user_locations[phone] = {}
    user_locations[phone]["district"] = district
    print(f"District saved for {phone}: {district}")

def set_user_village(phone: str, village: str):
    if phone not in user_locations:
        user_locations[phone] = {}
    user_locations[phone]["village"] = village
    print(f"Village saved for {phone}: {village}")

def get_user_location(phone: str) -> dict:
    return user_locations.get(phone, {})