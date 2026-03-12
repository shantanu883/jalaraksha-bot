from langdetect import detect

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        print(f"Language detected: {lang}")
        return lang
    except:
        return "en"