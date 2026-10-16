
import sqlite3
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Smart Campus Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DB = "campus.db"


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
