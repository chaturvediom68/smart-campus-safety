import time
import requests

BACKEND_URL = "https://collection-wyoming-delight-outsourcing.trycloudflare.com"
COOLDOWN_SEC = 30
_last_sent = {}

def send_alert(module, camera, event_type, severity, plate="", evidence_url=""):
    key = (module, camera, event_type, plate)
    now = time.time()
    if now - _last_sent.get(key, 0) < COOLDOWN_SEC:
        return  # abhi hi bheja tha, skip
    _last_sent[key] = now
    try:
        requests.post(f"{BACKEND_URL}/events", json={
            "module": module, "camera": camera, "event_type": event_type,
            "severity": severity, "plate": plate, "evidence_url": evidence_url,
        }, timeout=5)
    except Exception as e:
        print("Alert nahi gaya:", e)