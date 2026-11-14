"""One frame enters here; every available model contributes normalized events."""
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2

from .event_store import create_event
from .model_registry import ROOT, registry

COOLDOWN = int(os.getenv("EVENT_COOLDOWN_SECONDS", "30"))
_last_seen: dict[str, float] = {}
_anpr_module = None
_anpr_import_attempted = False


def _emit(module: str, camera: str, event_type: str, severity: str, details: dict, plate: str = ""):
    key = f"{camera}:{module}:{event_type}:{plate}"
    now = time.time()
    if now - _last_seen.get(key, 0) < COOLDOWN:
        return None
    _last_seen[key] = now
    return create_event({
        "module": module, "camera": camera, "event_type": event_type,
        "severity": severity, "plate": plate, "details": json.dumps(details),
    })


def _anpr_detect(frame):
    global _anpr_module, _anpr_import_attempted
    inference_path = ROOT / "anpr" / "inference.py"
    if inference_path.exists() and not _anpr_import_attempted:
        _anpr_import_attempted = True
        try:
            spec = importlib.util.spec_from_file_location("campus_anpr_inference", inference_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _anpr_module = module
        except Exception:
            _anpr_module = None
    if _anpr_module is not None:
        try:
            return _anpr_module.detect(frame)
        except Exception:
            # Fall back to the shared registry if the legacy adapter fails.
            _anpr_module = None
    vehicles = registry.predict("anpr_vehicle", frame)
    plates = registry.predict("anpr_plate", frame)
    return {"vehicles": vehicles, "plates": plates}


def process_frame(frame, camera: str = "Camera 1"):
    if frame is None:
        raise ValueError("Could not decode the uploaded image/frame.")

    generated = []
    for label in registry.predict("helmet", frame):
        name = label["label"]
        if "no_helmet" in name or "without_helmet" in name or "nohelmet" in name:
            event = _emit("Helmet", camera, "no_helmet", "high", label)
            if event: generated.append(event)

    for label in registry.predict("triple_riding", frame):
        name = label["label"]
        if "triple" in name or "three" in name:
            event = _emit("Helmet + Triple Riding", camera, "triple_riding", "high", label)
            if event: generated.append(event)

    anpr = _anpr_detect(frame)
    for vehicle in anpr.get("vehicles", []):
        event = _emit("Vehicle + ANPR", camera, "vehicle_detected", "low", vehicle)
        if event: generated.append(event)

    for plate in anpr.get("plates", []):
        event = _emit("Vehicle + ANPR", camera, "number_plate_detected", "low", plate)
        if event: generated.append(event)

    return {
        "camera": camera,
        "events_created": generated,
        "models": registry.status,
        "note": "Only models with available weights are executed. Missing modules are not simulated.",
    }


def decode_image(contents: bytes):
    import numpy as np
    array = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
