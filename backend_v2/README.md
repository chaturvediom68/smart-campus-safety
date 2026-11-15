# Smart Campus Safety — Central Backend v2

This backend lives in `backend_v2/`; it does not overwrite the original backend or Streamlit frontend.

## Connected detection modules

- Helmet: `MODEL/weights/helmet_best.pt`
- Triple riding: `MODEL/weights/tripleriding_best.pt`
- Seatbelt: `MODEL/seatbelt_best.pt` (configurable with `SEATBELT_MODEL_PATH`)
- Vehicle detection: `anpr/weights/vehicle_yolov8n.pt`
- Number plate detection: `anpr/weights/plate_best.pt`
- Weapon detection: `weapon detection/weights/best.pt` (configurable with `WEAPON_MODEL_PATH`; ensure this is a valid weapon-trained checkpoint)
- Geofencing: no separate geofencing weight file is required. People are detected with OpenCV HOG, vehicles with the existing ANPR detector, and a polygon/time rule determines restricted-zone entry.

All YOLO models are loaded by `backend_v2/model_registry.py` and called by the shared frame pipeline. Seatbelt detections labelled `no_seatbelt`, `without_seatbelt` or `no_seat_belt` create high-severity events. A model showing as `ready` means its checkpoint loaded; it does not guarantee accurate detections, so test it with representative footage.

## Install and run

Run from the repository root:

```bash
pip install -r backend_v2/requirements.txt
python3 -m uvicorn backend_v2.main:app --host 0.0.0.0 --port 8000
```

Open Swagger at `/docs`.

## Model checkpoint setup

Seatbelt checkpoint: `MODEL/seatbelt_best.pt`.
Weapon checkpoint: `weapon detection/weights/best.pt`.

Restart the backend or call `POST /models/reload`, then check `GET /models` for each model's status and class names. Do not use a seatbelt checkpoint as the weapon checkpoint.

## Configure the geofence

Inspect `GET /geofence/config`, then configure the camera's normalized polygon with `PUT /geofence/config`. Polygon points range from 0.0 to 1.0, measured from the top-left. The time window supports overnight ranges.

Upload an image to `POST /process/image` or a video to `POST /process/video`. The geofence runs on the same sampled frames and saves restricted-time entries in the central events database.

## Important limitations

- OpenCV HOG is a classic upright-person detector, not a modern deep-learning detector. It can miss people at unusual angles, in crowds, at night, or when people are small in the frame.
- Geofencing checks the bottom-centre of each detected bounding box against the polygon. It does not track identities between frames.
- Video uploads are sampled every Nth frame; this backend does not yet connect to live CCTV/RTSP streams.
- Vehicle/plate observations are not proof that a vehicle is authorized. An allow-list is needed for authorization.
- Restrict CORS and protect API routes before public deployment.
