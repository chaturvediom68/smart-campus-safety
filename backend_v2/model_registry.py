"""Loads model files present in the repository and reports algorithmic modules."""
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.40"))


def resolve_path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else ROOT / value


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, Any] = {}
        self.status: dict[str, dict[str, str]] = {}

    def _load_yolo(self, key: str, env_name: str, default_path: str):
        path = resolve_path(env_name, default_path)
        if not path.exists():
            self.status[key] = {"status": "unavailable", "reason": f"Weight file not found: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"}
            return
        try:
            from ultralytics import YOLO
            self.models[key] = YOLO(str(path))
            self.status[key] = {"status": "ready", "weight": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)}
        except Exception as exc:
            self.status[key] = {"status": "error", "reason": str(exc)}

    def load_all(self):
        self._load_yolo("helmet", "HELMET_MODEL_PATH", "MODEL/weights/helmet_best.pt")
        self._load_yolo("triple_riding", "TRIPLE_RIDING_MODEL_PATH", "MODEL/weights/tripleriding_best.pt")
        self._load_yolo("anpr_vehicle", "ANPR_VEHICLE_MODEL_PATH", "anpr/weights/vehicle_yolov8n.pt")
        self._load_yolo("anpr_plate", "ANPR_PLATE_MODEL_PATH", "anpr/weights/plate_best.pt")
        self.status["weapon"] = {"status": "unavailable", "reason": "Weapon model weights are not present in the repository."}
        self.status["geofencing"] = {
            "status": "ready",
            "method": "OpenCV HOG person detection + existing ANPR vehicle detection + polygon/time rules",
            "separate_weights_required": "no",
        }

    def predict(self, key: str, frame):
        model = self.models.get(key)
        if model is None:
            return []
        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        detections = []
        for result in results:
            names = result.names or {}
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, cls_id)).strip().lower().replace("-", "_").replace(" ", "_")
                confidence = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({"label": label, "confidence": confidence, "box": xyxy})
        return detections


registry = ModelRegistry()
