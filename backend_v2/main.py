"""Central Smart Campus Safety API (v2)."""
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .event_store import (
    create_event, get_geofence_config, init_db, list_cameras,
    list_events, save_geofence_config, update_event,
)
from .model_registry import registry
from .pipeline import decode_image, process_frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    registry.load_all()
    yield


app = FastAPI(
    title="Smart Campus Safety Central Backend",
    version="2.1.0",
    description="Central API for campus safety AI models, rule-based geofencing and event management.",
)
app.router.lifespan_context = lifespan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewEvent(BaseModel):
    module: str
    camera: str
    event_type: str
    severity: str = "medium"
    plate: str = ""
    evidence_url: str = ""
    details: dict = Field(default_factory=dict)


class StatusUpdate(BaseModel):
    status: str


class GeofenceConfig(BaseModel):
    enabled: bool = False
    polygon: list[list[float]] = Field(default_factory=lambda: [
        [0.25, 0.25], [0.75, 0.25], [0.75, 0.85], [0.25, 0.85]
    ])
    restricted_start: str = "18:00"
    restricted_end: str = "23:59"
    timezone: str = "Asia/Kolkata"

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if len(value) < 3:
            raise ValueError("Polygon must contain at least three [x, y] points.")
        for point in value:
            if len(point) != 2 or not all(0.0 <= coordinate <= 1.0 for coordinate in point):
                raise ValueError("Every polygon point must contain normalized x,y values between 0 and 1.")
        return value

    @field_validator("restricted_start", "restricted_end")
    @classmethod
    def validate_time(cls, value):
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError as exc:
            raise ValueError("Time must use 24-hour HH:MM format.") from exc
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Unknown timezone. Example: Asia/Kolkata") from exc
        return value


@app.get("/")
def root():
    return {"service": "Smart Campus Safety Central Backend", "version": "2.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "models": registry.status}


@app.get("/models")
def models():
    return registry.status


@app.get("/events")
def events(limit: int = 200):
    return list_events(max(1, min(limit, 1000)))


@app.post("/events")
def add_event(payload: NewEvent):
    data = payload.model_dump()
    import json
    data["details"] = json.dumps(data["details"])
    return create_event(data)


@app.patch("/events/{event_id}")
def patch_event(event_id: int, payload: StatusUpdate):
    if payload.status not in {"new", "acknowledged", "resolved", "dismissed"}:
        raise HTTPException(status_code=400, detail="Unsupported event status.")
    updated = update_event(event_id, payload.status)
    if updated is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    return updated


@app.get("/cameras")
def cameras():
    return list_cameras()


@app.get("/geofence/config")
def read_geofence_config():
    return get_geofence_config()


@app.put("/geofence/config")
def update_geofence_config(payload: GeofenceConfig):
    config = payload.model_dump()
    return save_geofence_config(config)


@app.post("/process/image")
async def process_image(file: UploadFile = File(...), camera: str = Form("Camera 1")):
    contents = await file.read()
    frame = decode_image(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")
    return process_frame(frame, camera)


@app.post("/process/frame")
async def process_frame_endpoint(file: UploadFile = File(...), camera: str = Form("Camera 1")):
    """Alias retained for simple camera/frame integrations."""
    return await process_image(file, camera)


@app.post("/process/video")
async def process_video(
    file: UploadFile = File(...),
    camera: str = Form("Camera 1"),
    sample_every: int = Form(15),
):
    """Process a video upload by sampling every Nth frame."""
    import cv2
    import os
    import tempfile

    suffix = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    cap = cv2.VideoCapture(temp_path)
    sampled = 0
    created = []
    frame_index = 0
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open uploaded video.")
        step = max(1, min(sample_every, 300))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                result = process_frame(frame, camera)
                created.extend(result["events_created"])
                sampled += 1
            frame_index += 1
    finally:
        cap.release()
        os.unlink(temp_path)
    return {"camera": camera, "frames_sampled": sampled, "events_created": created, "models": registry.status}


@app.post("/models/reload")
def reload_models():
    registry.models.clear()
    registry.status.clear()
    registry.load_all()
    return registry.status
