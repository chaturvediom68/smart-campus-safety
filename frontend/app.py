import os
import json
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Smart Campus Safety", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.4rem;}
[data-testid="stSidebar"] {background: #0B1F3A;}
[data-testid="stSidebar"] * {color: #E8EEF7;}
.hero {background:linear-gradient(120deg,#0B1F3A,#1E5AA8);color:white;padding:22px 26px;border-radius:14px;margin-bottom:18px;}
.hero h1 {color:white;margin:0;font-size:27px;}
.hero p {margin:5px 0 0 0;opacity:.85;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ Smart Campus Safety</h1><p>Central AI detection and campus monitoring dashboard</p></div>', unsafe_allow_html=True)

DEFAULT_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
st.sidebar.title("🛡️ Control Panel")
backend_url = st.sidebar.text_input("FastAPI backend URL", DEFAULT_URL).strip().rstrip("/")
page = st.sidebar.radio("Navigation", ["AI Detection", "Model Status", "Alerts & Events", "Camera Status"])
st.sidebar.caption("Backend API: " + (backend_url or "Not configured"))

def api_get(path, timeout=10):
    if not backend_url:
        raise RuntimeError("Enter the FastAPI backend URL in the sidebar.")
    response = requests.get(f"{backend_url}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()

def api_post_file(path, uploaded, camera, sample_every=None):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
    data = {"camera": camera}
    if sample_every is not None:
        data["sample_every"] = str(sample_every)
    response = requests.post(f"{backend_url}{path}", files=files, data=data, timeout=600)
    response.raise_for_status()
    return response.json()

def severity_color(value):
    return {"high":"🔴", "medium":"🟠", "low":"🟢"}.get(str(value).lower(), "⚪")

if page == "AI Detection":
    st.subheader("Run all connected AI models")
    st.write("Upload an image or video. The central backend runs the available detection models and saves generated events.")
    try:
        health = api_get("/health")
        st.success("Backend connected")
    except Exception as exc:
        st.error(f"Cannot connect to backend: {exc}")
        st.info("Start FastAPI in a terminal: python3 -m uvicorn backend_v2.main:app --host 0.0.0.0 --port 8000")
        st.stop()

    camera = st.text_input("Camera name", "Camera 1")
    mode = st.radio("Input type", ["Image", "Video"], horizontal=True)
    if mode == "Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])
        if uploaded and st.button("Run detection", type="primary"):
            with st.spinner("Running all available models..."):
                try:
                    result = api_post_file("/process/image", uploaded, camera)
                    st.success("Image processed")
                    st.json(result)
                    st.session_state["last_detection"] = result
                except Exception as exc:
                    st.error(f"Detection request failed: {exc}")
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
        sample_every = st.number_input("Process every Nth frame", min_value=1, max_value=300, value=15)
        if uploaded and st.button("Process video", type="primary"):
            with st.spinner("Processing video. This may take a while..."):
                try:
                    result = api_post_file("/process/video", uploaded, camera, int(sample_every))
                    st.success(f"Video processed — {result.get('frames_sampled', 0)} frames sampled")
                    st.json(result)
                    st.session_state["last_detection"] = result
                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")

    if "last_detection" in st.session_state:
        result = st.session_state["last_detection"]
        st.subheader("Latest detection summary")
        st.metric("Events created in this run", len(result.get("events_created", [])))
        if result.get("events_created"):
            st.dataframe(pd.DataFrame(result["events_created"]), use_container_width=True, hide_index=True)
        else:
            st.info("No new events were generated for this input. This does not necessarily mean every model detected nothing; review the returned model status and test footage.")

elif page == "Model Status":
    st.subheader("Connected model registry")
    if st.button("Refresh model status"):
        st.rerun()
    try:
        models = api_get("/models")
        rows = []
        for name, info in models.items():
            rows.append({"Model": name.replace("_"," ").title(), "Status": info.get("status","unknown"),
                         "Weights": info.get("weight", info.get("method","")),
                         "Classes / Details": info.get("classes", info.get("reason", info.get("method","")))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("READY confirms that a checkpoint loaded; it does not prove detection accuracy. Test each model with suitable footage.")
        if st.button("Reload all model weights"):
            result = requests.post(f"{backend_url}/models/reload", timeout=120)
            result.raise_for_status()
            st.success("Model reload completed")
            st.json(result.json())
    except Exception as exc:
        st.error(f"Could not retrieve model status: {exc}")

elif page == "Alerts & Events":
    st.subheader("Events generated by the central backend")
    limit = st.slider("Maximum events", 20, 1000, 200, 20)
    try:
        events = api_get(f"/events?limit={limit}")
        if events:
            df = pd.DataFrame(events)
            if "timestamp" in df:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
            if "severity" in df:
                df["severity"] = df["severity"].str.lower()
            st.metric("Events retrieved", len(df))
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download events CSV", df.to_csv(index=False), "campus_safety_events.csv", "text/csv")
            st.subheader("Update event status")
            event_ids = df["id"].tolist() if "id" in df else []
            if event_ids:
                selected_id = st.selectbox("Event ID", event_ids)
                new_status = st.selectbox("New status", ["acknowledged","resolved","dismissed","new"])
                if st.button("Save event status"):
                    response = requests.patch(f"{backend_url}/events/{selected_id}", json={"status":new_status}, timeout=15)
                    response.raise_for_status()
                    st.success("Event status updated")
                    st.rerun()
        else:
            st.info("No events have been saved yet. Run an image or video detection first.")
    except Exception as exc:
        st.error(f"Could not load events: {exc}")

elif page == "Camera Status":
    st.subheader("Registered cameras")
    try:
        cameras = api_get("/cameras")
        if cameras:
            df = pd.DataFrame(cameras)
            if "status" in df:
                df["status"] = df["status"].str.title()
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No cameras are registered yet. Uploaded image/video processing can still use the camera name entered in AI Detection.")
    except Exception as exc:
        st.error(f"Could not load camera list: {exc}")
