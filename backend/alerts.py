from firebase_db import count_recent_reports, save_alert
from whatsapp import send_message
import os
from dotenv import load_dotenv

load_dotenv()

ASHA_WORKER = os.getenv("ASHA_WORKER_NUMBER")
BMO_NUMBER = os.getenv("BMO_NUMBER")

async def check_and_alert(phone: str, intent: str):
    if intent == "UNKNOWN":
        return

    count = count_recent_reports(intent)
    print(f"Recent {intent} reports in 24hrs: {count}")

    if count >= 3:
        save_alert(phone, intent, count)

        alert_message = (
            f"🚨 *JALARAKSHA ALERT* 🚨\n\n"
            f"⚠️ *{count} {intent} reports* in last 24 hours!\n\n"
            f"🔴 Immediate action required!\n"
            f"Please visit the reported area.\n\n"
            f"— Jalaraksha Early Warning System"
        )

        if ASHA_WORKER:
            await send_message(ASHA_WORKER, alert_message)
            print("✅ Alert sent to ASHA worker!")

        if count >= 5 and BMO_NUMBER:
            await send_message(BMO_NUMBER, alert_message)
            print("✅ Alert sent to BMO!")