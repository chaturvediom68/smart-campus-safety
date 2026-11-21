"""Weight-free geofencing rules using OpenCV HOG for people and existing ANPR for vehicles.

HOG is OpenCV's built-in people detector: it needs no separate .pt file.
The restricted polygon is stored in normalized image coordinates (0.0 to 1.0).
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

from .event_store import get_geofence_config


class PersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        # HOG is more stable and faster on moderate-sized frames.
        height, width = frame.shape[:2]
        scale = min(1.0, 960.0 / max(width, 1))
        image = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
        boxes, scores = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        detections = []
        inverse = 1.0 / scale
        for (x, y, w, h), score in zip(boxes, scores):
            detections.append({
                "label": "person",
                "confidence": float(score),
                "box": [
                    int(x * inverse), int(y * inverse),
                    int((x + w) * inverse), int((y + h) * inverse),
                ],
            })
        return detections


_person_detector = None


def _get_person_detector():
    global _person_detector
    if _person_detector is None:
        _person_detector = PersonDetector()
    return _person_detector


def _inside_zone(box, polygon, width, height):
    # Use the bottom-centre point: where the person/vehicle touches the ground.
    x1, y1, x2, y2 = box
    point = (int(((x1 + x2) / 2) / width * 10000), int(y2 / height * 10000))
    contour = np.array(
        [[int(px * 10000), int(py * 10000)] for px, py in polygon],
        dtype=np.int32,
    )
    return cv2.pointPolygonTest(contour, point, False) >= 0


def _is_restricted_now(config):
    try:
        now = datetime.now(ZoneInfo(config.get("timezone", "Asia/Kolkata"))).time()
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now().time()
    start = datetime.strptime(config.get("restricted_start", "18:00"), "%H:%M").time()
    end = datetime.strptime(config.get("restricted_end", "23:59"), "%H:%M").time()
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def detect_and_alert(frame, camera, vehicles, emit):
    config = get_geofence_config()
    result = {
        "enabled": bool(config.get("enabled", False)),
        "restricted_time": _is_restricted_now(config),
        "people_detected": 0,
        "vehicles_checked": 0,
        "zone_entries": [],
    }
    if not result["enabled"]:
        return result, []

    height, width = frame.shape[:2]
    polygon = config.get("polygon", [])
    if len(polygon) < 3:
        result["warning"] = "Geofence polygon needs at least three points."
        return result, []

    objects = []
    people = _get_person_detector().detect(frame)
    result["people_detected"] = len(people)
    for person in people:
        objects.append(("person", person))

    for vehicle in vehicles:
        box = vehicle.get("box")
        if box and len(box) == 4:
            objects.append(("vehicle", vehicle))
            result["vehicles_checked"] += 1

    generated = []
    for kind, detection in objects:
        box = detection.get("box")
        if not box or len(box) != 4 or not _inside_zone(box, polygon, width, height):
            continue
        entry = {"object": kind, "box": box, "confidence": detection.get("confidence", detection.get("conf"))}
        result["zone_entries"].append(entry)
        if result["restricted_time"]:
            event = emit(
                "Geofencing",
                camera,
                f"restricted_zone_{kind}_entry",
                "high",
                {
                    **entry,
                    "restricted_start": config.get("restricted_start", "18:00"),
                    "restricted_end": config.get("restricted_end", "23:59"),
                    "timezone": config.get("timezone", "Asia/Kolkata"),
                    "polygon": polygon,
                },
            )
            if event:
                generated.append(event)

    return result, generated
