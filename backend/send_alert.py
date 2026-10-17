   import requests

   BACKEND_URL = "https://collection-wyoming-delight-outsourcing.trycloudflare.com"

   def send_alert(module, camera, event_type, severity, plate="", evidence_url=""):
       try:
           requests.post(f"{BACKEND_URL}/events", json={
               "module": module, "camera": camera, "event_type": event_type,
               "severity": severity, "plate": plate, "evidence_url": evidence_url,
           }, timeout=5)
       except Exception as e:
           print("Alert nahi gaya:", e)