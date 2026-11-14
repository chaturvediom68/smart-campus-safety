# Smart Campus Safety — Central Backend v2

This backend lives in `backend_v2/`; it does not overwrite the original backend or Streamlit frontend.

## Available detection modules

- Helmet: `MODEL/weights/helmet_best.pt`
- Triple riding: `MODEL/weights/tripleriding_best.pt`
- Vehicle detection: `anpr/weights/vehicle_yolov8n.pt`
- Number plate detection: `anpr/weights/plate_best.pt`
- Geofencing: **no separate geofencing weight file required**. Vehicles come from the existing ANPR vehicle detector; people are detected with OpenCV's built-in HOG person detector. A polygon and time rule decide whether an object is inside a restricted zone.
- Weapon detection remains unavailable until its model weights and inference code are supplied.

## Setup

Run from the repository root:

```bash
pip install -r backend_v2/requirements.txt
python3 -m uvicorn backend_v2.main:app --reload --host 0.0.0.0
```

Open Swagger at `/docs`.

## Configure the geofence

The default geofence is disabled until you set a real restricted-area polygon.

1. Open `GET /geofence/config` to inspect the current settings.
2. Open `PUT /geofence/config` in Swagger, click **Try it out**, and submit a configuration like this:

```json
{
  "enabled": true,
  "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.85], [0.25, 0.85]],
  "restricted_start": "18:00",
  "restricted_end": "23:59",
  "timezone": "Asia/Kolkata"
}
```

Polygon points are normalized image coordinates from 0.0 to 1.0, measured from the top-left. The sample rectangle is only a placeholder; choose points that match the actual camera view. The time window supports overnight ranges, such as 18:00 to 06:00.

Upload an image to `POST /process/image` or a video to `POST /process/video`. The geofence runs on the same sampled frames and saves restricted-time entries in the central events database.

## Important limitations

- OpenCV HOG is a classic upright-person detector, not a modern deep-learning detector. It can miss people at unusual angles, in crowds, at night, or when people are small in the frame. Test it with your camera footage.
- Geofencing checks the bottom-centre of each detected bounding box against the polygon. It does not track identities between frames.
- Video uploads are sampled every Nth frame; this backend does not yet connect to live CCTV/RTSP streams.
- Vehicle/plate observations are not proof that a vehicle is authorized. An allow-list is needed for authorization.
- Restrict CORS and protect API routes before public deployment. Rotate any API credentials previously committed to the repository.
