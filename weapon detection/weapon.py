!pip install -q ultralytics
import cv2, os, glob, yaml, requests, numpy as np
from collections import defaultdict, deque
from google.colab import drive, files
from ultralytics import YOLO

# ---------- SYSTEM SECURITY CONFIGURATION ----------
WEAPON_CONF, FAST_TH, CLOSE_TH, ALERT_TH, WARN_TH = 0.4, 0.15, 0.6, 0.6, 0.3
WEIGHT = {"pistol": 1.0, "knife": 0.9}
COLORS = {"ALERT": (0,0,255), "WARNING": (0,165,255), "WATCH": (0,255,255), "NONE": (0,200,0)}

# TELEGRAM (asli token GitHub par kabhi mat daalna)
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
# --------------------------------------------------s

def dispatch_security_telemetry(log_string, frame_matrix=None):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        return
    endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"[CRITICAL SURVEILLANCE ALARM]\n{log_string}"}
    try:
        requests.post(endpoint, json=payload, timeout=5)
        if frame_matrix is not None:
            photo_endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            _, img_encoded = cv2.imencode('.jpg', frame_matrix)
            files_payload = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
            requests.post(photo_endpoint, data={'chat_id': TELEGRAM_CHAT_ID}, files=files_payload, timeout=5)
    except Exception as e:
        print(f"[Telemetry Error] Failed to dispatch alarm: {e}")

try:
    drive.mount('/content/drive')
except Exception as e:
    print("Google Drive mount skipped or already active.")

# ---------- 1) Weapon model dhundo (na mile to upload karwao) ----------
candidates = [
    "/content/drive/MyDrive/weapon_best.pt",
    "/content/runs/weapon/weights/best.pt",
    "/content/best.pt",
] + glob.glob("/content/runs/**/weights/best.pt", recursive=True)
weapon_model_path = next((p for p in candidates if os.path.exists(p)), None)
if weapon_model_path is None:
    print("best.pt kahin nahi mila. Apni weapon model file (best.pt) upload karo:")
    up = files.upload()
    if not up:
        raise FileNotFoundError("Weapon model file upload nahi hui.")
    weapon_model_path = list(up.keys())[0]
print("Weapon model:", weapon_model_path)

pose = YOLO("yolov8s-pose.pt")
wmodel = YOLO(weapon_model_path)

# ---------- 2) Precision / Recall khud nikalo (dataset ke test/val split se) ----------
data_yamls = (glob.glob("/content/**/data.yaml", recursive=True)
              + glob.glob("/content/drive/MyDrive/data.yaml")
              + glob.glob("/content/drive/MyDrive/*/data.yaml")
              + glob.glob("/content/drive/MyDrive/*/*/data.yaml"))
data_yamls = [p for p in data_yamls if "site-packages" not in p]

metrics_text = "Metrics nahi nikale (data.yaml / dataset nahi mila)"
if data_yamls:
    DATA_YAML = data_yamls[0]
    try:
        cfg = yaml.safe_load(open(DATA_YAML))
        split = "test" if cfg.get("test") else "val"
        m = wmodel.val(data=DATA_YAML, split=split, verbose=False)
        lines = [f"Evaluated on '{split}' split ({DATA_YAML})",
                 f"mAP50: {m.box.map50:.3f} | mAP50-95: {m.box.map:.3f} | "
                 f"Precision: {m.box.mp:.3f} | Recall: {m.box.mr:.3f}"]
        for idx, ci in enumerate(m.box.ap_class_index):
            lines.append(f"  {m.names[ci]:8}: Precision {m.box.p[idx]:.3f} | Recall {m.box.r[idx]:.3f}")
        metrics_text = "\n".join(lines)
    except Exception as e:
        metrics_text = f"Metrics nahi nikal paye ({DATA_YAML}): {e}"
if metrics_text.startswith("Metrics nahi"):
    try:
        import torch
        ck = torch.load(weapon_model_path, map_location="cpu", weights_only=False)
        tm = ck.get("train_metrics") or {}
        if tm:
            g = lambda k: tm.get(k, float("nan"))
            metrics_text = (f"Precision: {g('metrics/precision(B)'):.3f} | Recall: {g('metrics/recall(B)'):.3f} | "
                            f"mAP50: {g('metrics/mAP50(B)'):.3f} | mAP50-95: {g('metrics/mAP50-95(B)'):.3f}")
    except Exception as e:
        metrics_text += f" | best.pt se bhi nahi padh paya: {e}"
print("\n" + metrics_text + "\n")

# ---------- 3) Video upload ----------
print("Please upload/select your test video now:")
uploaded = files.upload()
if not uploaded:
    raise FileNotFoundError("No video file was uploaded. Please run the cell again and choose a video.")

VIDEO_IN = list(uploaded.keys())[0]
VIDEO_OUT = "output_telemetry_" + VIDEO_IN.rsplit(".", 1)[0] + ".mp4"

cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), f"Could not open video file: {VIDEO_IN}"
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

prev, last, alert_log, frame_no = {}, {}, [], 0
wstats, levels = defaultdict(list), defaultdict(int)
hist = defaultdict(lambda: deque(maxlen=int(fps * 1.5)))

print("Processing security tracking telemetry...")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_no += 1

    pr = pose.track(frame, persist=True, verbose=False)[0]
    wr = wmodel.predict(frame, conf=WEAPON_CONF, iou=0.45, verbose=False)[0]

    frame = wr.plot(img=frame)
    detected_threats = [wmodel.names[k] for k in wr.boxes.cls.int().tolist()]
    for name, confidence in zip(detected_threats, wr.boxes.conf.tolist()):
        wstats[name].append(confidence)

    if detected_threats:
        cv2.putText(frame, "DETECTED THREAT: " + ", ".join(sorted(set(detected_threats))).upper(), (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    people = []
    if pr.boxes.id is not None:
        for pid, b, kp, kc in zip(pr.boxes.id.int().tolist(), pr.boxes.xyxy.cpu().numpy(),
                                  pr.keypoints.xy.cpu().numpy(), pr.keypoints.conf.cpu().numpy()):
            people.append(dict(id=pid, box=b, h=max(b[3]-b[1], 1), c=np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]),
                               wr=[(kp[j], kc[j]) for j in (9, 10)]))

    held = {}
    for b, n in zip(wr.boxes.xyxy.cpu().numpy(), detected_threats):
        wc = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
        d, p = min(((np.linalg.norm(wc-pt)/p["h"], p) for p in people for pt, cf in p["wr"] if cf > 0.3),
                   key=lambda x: x[0], default=(9, None))
        if d < 0.5 and WEIGHT.get(n, 0.5) > held.get(p["id"], (0, ""))[0]:
            held[p["id"]] = (WEIGHT.get(n, 0.5), n)

    for p in people:
        others = [o for o in people if o["id"] != p["id"]]
        fast = toward = close = 0
        for j, (pt, cf) in enumerate(p["wr"]):
            if cf <= 0.3: continue
            if (p["id"], j) in prev and others:
                mv = pt - prev[(p["id"], j)]
                to = min(others, key=lambda o: np.linalg.norm(o["c"]-pt))["c"] - pt
                fast   = max(fast,   np.linalg.norm(mv)/p["h"]*(fps/30) > FAST_TH)
                toward = max(toward, mv @ to / (np.linalg.norm(mv)*np.linalg.norm(to) + 1e-6) > 0.7)
                close  = max(close,  np.linalg.norm(to)/p["h"] < CLOSE_TH)
            prev[(p["id"], j)] = pt

        hist[p["id"]].append(0.4*fast + 0.3*toward + 0.3*close)
        wt, wn = held.get(p["id"], (0, "-"))
        risk = wt * (np.mean(hist[p["id"]]) if len(hist[p["id"]]) >= 8 else 0)
        level = "ALERT" if risk > ALERT_TH else "WARNING" if risk > WARN_TH else "WATCH" if wt >= 0.9 else "NONE"

        if level != "NONE":
            levels[level] += 1
            if last.get(p["id"]) != level:
                log_msg = f"{frame_no/fps:6.1f}s | ID{p['id']} | {wn:6} | {level:7} | risk_score={risk:.2f}"
                alert_log.append(log_msg)
                if level in ["WARNING", "ALERT"]:
                    dispatch_security_telemetry(log_msg, frame)
        last[p["id"]] = level

        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[level], 2)
        cv2.putText(frame, f"ID{p['id']} {level} r={risk:.2f}", (x1, max(y1-8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[level], 2)
    out.write(frame)

cap.release()
out.release()

print("\n========== SECURITY TELEMETRY REPORT ==========")
print(f"Processed File: {VIDEO_IN} | Frame Count: {frame_no} | FPS Output: {fps:.0f}")
print("\nDETECTION SUMMARY (this video):")
for n, c in wstats.items():
    print(f"  {n.upper():7}: Occurrences: {len(c)} | Mean Confidence: {np.mean(c):.2f} | Peak Confidence: {max(c):.2f}")
if not wstats:
    print("  Zero validated threat markers observed.")
print("\nTHREAT STATE CLASSIFICATIONS:", dict(levels) or "None")
print("\nTHREAT EVENT HISTORICAL LOG:")
print("\n".join(alert_log) or "  No alert transitions logged.")
print("\nMODEL METRICS (val):")
print(metrics_text)
print("===============================================")

files.download(VIDEO_OUT)