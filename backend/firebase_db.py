import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta, timezone
import os
import json

# Load Firebase from environment variable
firebase_json = os.getenv("FIREBASE_KEY_JSON")

if firebase_json:
    # Production — Railway
    firebase_dict = json.loads(firebase_json)
    cred = credentials.Certificate(firebase_dict)
else:
    # Local development
    cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Firebase connected!")

def save_report(phone: str, message: str, intent: str, language: str,
                district: str = None, village: str = None):
    try:
        data = {
            "phone": phone,
            "message": message,
            "intent": intent,
            "language": language,
            "timestamp": datetime.now(timezone.utc),
            "alerted": False,
            "state": "Maharashtra"
        }
        if district:
            data["district"] = district
        if village:
            data["village"] = village

        db.collection("reports").add(data)
        print(f"✅ Report saved: {intent} | {district} | {village}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")

def count_recent_reports(intent: str) -> int:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        docs = db.collection("reports")\
            .where("intent", "==", intent)\
            .stream()
        count = 0
        for doc in docs:
            ts = doc.to_dict().get("timestamp")
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    count += 1
        print(f"Recent {intent} reports in 24hrs: {count}")
        return count
    except Exception as e:
        print(f"❌ Error counting reports: {e}")
        return 0

def save_alert(phone: str, intent: str, count: int):
    try:
        db.collection("alerts").add({
            "phone": phone,
            "intent": intent,
            "report_count": count,
            "triggered_at": datetime.now(timezone.utc),
            "resolved": False
        })
        print(f"✅ Alert saved: {intent}")
    except Exception as e:
        print(f"❌ Error saving alert: {e}")

def get_all_reports():
    try:
        docs = db.collection("reports")\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(100)\
            .stream()
        reports = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            if "timestamp" in data and data["timestamp"]:
                data["timestamp"] = str(data["timestamp"])
            reports.append(data)
        return reports
    except Exception as e:
        print(f"❌ Error getting reports: {e}")
        return []

def get_all_alerts():
    try:
        docs = db.collection("alerts")\
            .order_by("triggered_at", direction=firestore.Query.DESCENDING)\
            .stream()
        alerts = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            if "triggered_at" in data and data["triggered_at"]:
                data["triggered_at"] = str(data["triggered_at"])
            alerts.append(data)
        return alerts
    except Exception as e:
        print(f"❌ Error getting alerts: {e}")
        return []

def get_dashboard_stats():
    try:
        all_docs = list(db.collection("reports").stream())
        today_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        week_cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        today_reports = 0
        week_reports = 0
        intent_count = {}

        for doc in all_docs:
            data = doc.to_dict()
            ts = data.get("timestamp")
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= today_cutoff:
                    today_reports += 1
                if ts >= week_cutoff:
                    week_reports += 1
            intent = data.get("intent", "UNKNOWN")
            intent_count[intent] = intent_count.get(intent, 0) + 1

        active_alerts = sum(
            1 for doc in db.collection("alerts").stream()
            if not doc.to_dict().get("resolved", False)
        )

        return {
            "today_reports": today_reports,
            "active_alerts": active_alerts,
            "week_reports": week_reports,
            "total_reports": len(all_docs),
            "intent_breakdown": [
                {"intent": k, "count": v}
                for k, v in intent_count.items()
            ]
        }
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {
            "today_reports": 0,
            "active_alerts": 0,
            "week_reports": 0,
            "total_reports": 0,
            "intent_breakdown": []
        }