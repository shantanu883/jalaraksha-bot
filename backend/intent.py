# Intent detection and processing
def detect_intent(text: str) -> str:
    text = text.lower()

    symptom_keywords = [
        "fever", "diarrhea", "vomiting", "stomach",
        "sick", "ill", "pain", "nausea", "cholera",
        "typhoid", "bukhar", "dast", "ulti", "bimar",
        "pet dard", "loose motion"
    ]

    water_keywords = [
        "water", "smell", "dirty", "contaminated",
        "colour", "color", "yellow", "brown", "muddy",
        "paani", "ganda", "badhboo", "turbid"
    ]

    animal_keywords = [
        "cow", "died", "animal", "buffalo", "dead",
        "dog", "goat", "fish", "bird",
        "gaay", "mara", "janwar", "bail", "murgi"
    ]

    school_keywords = [
        "absent", "school", "students", "children",
        "kids", "class", "teacher",
        "bacche", "absentee", "vidyalaya"
    ]

    for word in symptom_keywords:
        if word in text:
            return "SYMPTOM"

    for word in water_keywords:
        if word in text:
            return "WATER"

    for word in animal_keywords:
        if word in text:
            return "ANIMAL"

    for word in school_keywords:
        if word in text:
            return "SCHOOL"

    return "UNKNOWN"