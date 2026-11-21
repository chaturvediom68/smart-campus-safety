"""SQLite event storage and persistent geofence configuration."""
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "backend_v2" / "campus_safety.db")))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH

DEFAULT_GEOFENCE = {
    "enabled": False,
    "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.85], [0.25, 0.85]],
    "restricted_start": "18:00",
    "restricted_end": "23:59",
    "timezone": "Asia/Kolkata",
}


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT NOT NULL,
                camera TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                plate TEXT DEFAULT '',
                evidence_url TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'new',
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                details TEXT DEFAULT '{}'
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                location TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'offline'
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
        """)
        db.execute("INSERT OR IGNORE INTO cameras(name, location, status) VALUES ('Camera 1', 'Main Gate', 'offline')")
        db.execute("INSERT OR IGNORE INTO cameras(name, location, status) VALUES ('Camera 2', 'Campus Area', 'offline')")
        db.execute(
            "INSERT OR IGNORE INTO app_settings(setting_key, setting_value) VALUES ('geofence', ?)",
            (json.dumps(DEFAULT_GEOFENCE),),
        )


def get_geofence_config():
    with connect() as db:
        row = db.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key='geofence'"
        ).fetchone()
    return json.loads(row["setting_value"]) if row else dict(DEFAULT_GEOFENCE)


def save_geofence_config(config: dict):
    value = json.dumps(config)
    with connect() as db:
        db.execute(
            """INSERT INTO app_settings(setting_key, setting_value) VALUES ('geofence', ?)
               ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value""",
            (value,),
        )
    return config


def list_events(limit: int = 200):
    with connect() as db:
        rows = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def create_event(event: dict[str, Any]):
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO events(module,camera,event_type,severity,plate,evidence_url,status,details)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                event.get("module", "System"), event.get("camera", "Camera 1"),
                event.get("event_type", "unknown"), event.get("severity", "medium"),
                event.get("plate", ""), event.get("evidence_url", ""),
                event.get("status", "new"), event.get("details", "{}"),
            ),
        )
        row = db.execute("SELECT * FROM events WHERE id=?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def update_event(event_id: int, status: str):
    with connect() as db:
        cursor = db.execute("UPDATE events SET status=? WHERE id=?", (status, event_id))
        if cursor.rowcount == 0:
            return None
        row = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    return dict(row)


def list_cameras():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM cameras ORDER BY id").fetchall()]
