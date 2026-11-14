# Smart Campus Safety — Central Backend v2

This is a new backend in `backend_v2/`. It does not overwrite the existing `backend/` or Streamlit frontend.

## What runs now

- Helmet detection: `MODEL/weights/helmet_best.pt`
- Triple-riding detection: `MODEL/weights/tripleriding_best.pt`
- Vehicle detection: `anpr/weights/vehicle_yolov8n.pt`
- Number-plate detection: `anpr/weights/plate_best.pt`
- Central SQLite event storage and API for the existing dashboard

Weapon and geofencing are listed as **unavailable** because their model weights / importable inference code are not in the repository. The backend never creates fake detections for them. Add the missing weights and inference code later, then register their adapters in `model_registry.py` and `pipeline.py`.

## Setup

Run commands from the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend_v2/requirements.txt
uvicorn backend_v2.main:app --reload
```

Open API docs at http://127.0.0.1:8000/docs.

## Connect the existing Streamlit dashboard

In the dashboard's Backend URL field, enter:

```
http://127.0.0.1:8000
```

The backend preserves the dashboard's main routes: `GET /events`, `POST /events`, `PATCH /events/{id}`, and `GET /cameras`.

## Run inference

Upload an image to `POST /process/image` or a video to `POST /process/video`. The video endpoint samples every Nth frame (default 15). You can test both endpoints from Swagger at `/docs`.

## Important notes

- This version processes uploaded images/videos. It does not yet open physical CCTV/RTSP streams automatically.
- `vehicle_detected` and `number_plate_detected` are observations, not proof that a vehicle is authorized or unauthorized. Authorization requires a trusted vehicle allow-list.
- Set model paths with environment variables if your weights are stored elsewhere.
- Restrict CORS and protect API routes before deploying this backend on a public network.
- Rotate any API credentials that were previously committed to the repository.
