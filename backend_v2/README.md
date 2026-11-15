# Smart Campus Safety — Central Backend v2

This backend lives in `backend_v2/`; it does not overwrite the original backend or Streamlit frontend.

## Connected detection modules

- Helmet: `MODEL/weights/helmet_best.pt`
- Triple riding: `MODEL/weights/tripleriding_best.pt`
- Vehicle detection: `anpr/weights/vehicle_yolov8n.pt`
- Number plate detection: `anpr/weights/plate_best.pt`
- Weapon detection: `weapon detection/weights/best.pt` (YOLO checkpoint; configurable with `WEAPON_MODEL_PATH`)
- Geofencing: no separate geofencing weight file is required. People are detected with OpenCV HOG, vehicles with the existing ANPR detector, and a polygon/time rule determines restricted-zone entry.

All YOLO models are loaded by `backend_v2/model_registry.py` and called by the shared frame pipeline. Weapon detections create high-severity `weapon_detected` events. A model showing as `ready` means its checkpoint loaded; it does not guarantee accurate detections, so test it with representative footage.

## Install and run

Run from the repository root:

```bash
pip install -r backend_v2/requirements.txt
python3 -m uvicorn backend_v2.main:app --host 0.0.0.0 --port 8000
```

Open Swagger at `/docs`.

## Weapon checkpoint setup

Upload your trained weapon checkpoint to:

```
weapon detection/weights/best.pt
```

Then restart the backend or call `POST /models/reload`. Check `GET /models` and confirm the weapon status is `ready`. The model's class names are shown in the status response.

If your checkpoint is stored elsewhere, set `WEAPON_MODEL_PATH` to its path before starting the backend. Do not commit private API tokens or secrets.

## Configure the geofence

The default geofence is disabled until you set a real restricted-area polygon.

1. Open `GET /geofence/config` to inspect the current settings.
2. Open `PUT /geofence/config` in Swagger and submit your camera's normalized polygon.

Polygon points are normalized image coordinates from 0.0 to 1.0, measured from the top-left. The time window supports overnight ranges.

Upload an image to `POST /process/image` or a video to `POST /process/video`. The geofence runs on the same sampled frames and saves restricted-time entries in the central events database.

## Important limitations

- OpenCV HOG is a classic upright-person detector, not a modern deep-learning detector. It can miss people at unusual angles, in crowds, at night, or when people are small in the frame.
- Geofencing checks the bottom-centre of each detected bounding box against the polygon. It does not track identities between frames.
- Video uploads are sampled every Nth frame; this backend does not yet connect to live CCTV/RTSP streams.
- Vehicle/plate observations are not proof that a vehicle is authorized. An allow-list is needed for authorization.
- Restrict CORS and protect API routes before public deployment.
