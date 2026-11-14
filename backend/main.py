
import sqlite3
import os
import cv2
import numpy as np
import base64

from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anpr.inference import detect

app = FastAPI(title="Smart Campus Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campus.db")


def read(sql, params=()):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def write(sql, params=()):
    conn = sqlite3.connect(DB)
    cur = conn.execute(sql, params)
    conn.commit()
    last = cur.lastrowid
    conn.close()
    return last


class NewEvent(BaseModel):
    module: str
    camera: str
    event_type: str
    severity: str
    plate: str = ""
    evidence_url: str = ""


class StatusUpdate(BaseModel):
    status: str


@app.get("/events")
def get_events(limit: int = 500):
    return read("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


@app.post("/events")
def add_event(e: NewEvent):
    ts = datetime.now().isoformat(timespec="seconds")
    new_id = write(
        "INSERT INTO events (timestamp,module,camera,event_type,severity,plate,evidence_url) VALUES (?,?,?,?,?,?,?)",
        (ts, e.module, e.camera, e.event_type, e.severity, e.plate, e.evidence_url))
    return {"id": new_id, "timestamp": ts}


@app.patch("/events/{event_id}")
def update_event(event_id: int, s: StatusUpdate):
    write("UPDATE events SET status=? WHERE id=?", (s.status, event_id))
    return {"id": event_id, "status": s.status}


@app.get("/cameras")
def get_cameras():
    return read("SELECT name, status, stream_url FROM cameras")
@app.post("/vehicle/detect")
async def vehicle_detection(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")

    image_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = detect(frame)

    output = frame.copy()

    for vehicle in results["vehicles"]:
        x1, y1, x2, y2 = vehicle["box"]

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            output, vehicle["type"],
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    for plate in results["plates"]:
        x1, y1, x2, y2 = plate["box"]

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            output, "Number Plate",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    success, encoded = cv2.imencode(".jpg", output)

    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return {
        "success": True,
        "vehicles": results["vehicles"],
        "plates": results["plates"],
        "vehicle_count": len(results["vehicles"]),
        "plate_count": len(results["plates"]),
        "output_image": base64.b64encode(encoded).decode("utf-8")
    }
