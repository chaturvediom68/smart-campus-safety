import os
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
vehicle_model = YOLO(os.path.join(BASE, "weights", "vehicle_yolov8n.pt"))
plate_model = YOLO(os.path.join(BASE, "weights", "plate_best.pt"))

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect(frame, conf=0.4):
    """frame: OpenCV image (BGR). Returns vehicles, plates and plate-to-vehicle match."""
    vehicles = []
    v = vehicle_model(frame, classes=list(VEHICLE_CLASSES), conf=conf, verbose=False)[0]
    for box, c, s in zip(v.boxes.xyxy.tolist(), v.boxes.cls.tolist(), v.boxes.conf.tolist()):
        vehicles.append({"type": VEHICLE_CLASSES[int(c)], "box": [int(x) for x in box], "conf": round(s, 2)})

    plates = []
    p = plate_model(frame, conf=conf, verbose=False)[0]
    for box, s in zip(p.boxes.xyxy.tolist(), p.boxes.conf.tolist()):
        x1, y1, x2, y2 = [int(x) for x in box]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        owner, best_area = None, None
        for i, veh in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = veh["box"]
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                area = (vx2 - vx1) * (vy2 - vy1)
                if best_area is None or area < best_area:
                    owner, best_area = i, area
        plates.append({"box": [x1, y1, x2, y2], "conf": round(s, 2), "vehicle_index": owner})

    return {"vehicles": vehicles, "plates": plates}
