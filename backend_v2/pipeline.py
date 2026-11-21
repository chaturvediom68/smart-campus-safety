"""One frame enters here; all available models and geofence rules contribute events."""
import json
import os
import time

import cv2

from .event_store import create_event
from .geofence import detect_and_alert
from .model_registry import registry

COOLDOWN = int(os.getenv("EVENT_COOLDOWN_SECONDS", "30"))
_last_seen: dict[str, float] = {}


def _emit(module: str, camera: str, event_type: str, severity: str, details: dict, plate: str = ""):
    key = f"{camera}:{module}:{event_type}:{plate}:{details.get('label', '')}"
    now = time.time()
    if now - _last_seen.get(key, 0) < COOLDOWN:
        return None
    _last_seen[key] = now
    return create_event({
        "module": module,
        "camera": camera,
        "event_type": event_type,
        "severity": severity,
        "plate": plate,
        "details": json.dumps(details),
    })


def _anpr_detect(frame):
    """Use the shared registry. Do not load ANPR weights a second time."""
    return {
        "vehicles": registry.predict("anpr_vehicle", frame),
        "plates": registry.predict("anpr_plate", frame),
    }


def process_frame(frame, camera: str = "Camera 1"):
    if frame is None:
        raise ValueError("Could not decode the uploaded image/frame.")

    generated = []

    # Weapon detection
    for detection in registry.predict("weapon", frame):
        event = _emit("Weapon Detection", camera, "weapon_detected", "high", detection)
        if event:
            generated.append(event)

    # Helmet detection
    for detection in registry.predict("helmet", frame):
        label = detection["label"]
        if any(token in label for token in ("no_helmet", "without_helmet", "nohelmet")):
            event = _emit("Helmet", camera, "no_helmet", "high", detection)
            if event:
                generated.append(event)

    # Triple-riding detection
    for detection in registry.predict("triple_riding", frame):
        label = detection["label"]
        if "triple" in label or "three" in label:
            event = _emit("Helmet + Triple Riding", camera, "triple_riding", "high", detection)
            if event:
                generated.append(event)

    # Seatbelt detection
    for detection in registry.predict("seatbelt", frame):
        label = detection["label"]
        if any(token in label for token in ("no_seatbelt", "without_seatbelt", "no_seat_belt")):
            event = _emit("Seatbelt Detection", camera, "no_seatbelt", "high", detection)
            if event:
                generated.append(event)

    # ANPR vehicle and number-plate detections
    anpr = _anpr_detect(frame)
    vehicles = anpr["vehicles"]
    for vehicle in vehicles:
        event = _emit("Vehicle + ANPR", camera, "vehicle_detected", "low", vehicle)
        if event:
            generated.append(event)

    for plate in anpr["plates"]:
        event = _emit("Vehicle + ANPR", camera, "number_plate_detected", "low", plate)
        if event:
            generated.append(event)

    # Geofencing uses HOG for people and the shared ANPR vehicle detections.
    geofence_result, geofence_events = detect_and_alert(frame, camera, vehicles, _emit)
    generated.extend(geofence_events)

    return {
        "camera": camera,
        "events_created": generated,
        "geofencing": geofence_result,
        "models": registry.status,
        "note": (
            "All available YOLO models run through one shared registry. "
            "Missing weight files are reported by GET /models. "
            "Geofencing uses OpenCV HOG person detection, ANPR vehicle detections, "
            "and configured polygon/time rules."
        ),
    }


def decode_image(contents: bytes):
    import numpy as np

    array = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
