import os
import tempfile
import time
import cv2
import streamlit as st
import requests
from typing import Optional
from requests.exceptions import RequestException

st.set_page_config(page_title="Smart Pantry - Video Detect", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_fps(video_bytes: bytes, suffix: str) -> Optional[float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None
    finally:
        cap.release()

st.title("Smart Pantry (MVP) — Upload video → Objects detected")

video_file = st.file_uploader("Upload video (mp4/mov/...)", type=None)

video_bytes = None
if video_file is not None:
    video_bytes = video_file.getvalue()
    size_mb = len(video_bytes) / (1024 * 1024)
    st.caption(f"File: {video_file.name} • Size: {size_mb:.2f} MB")
    st.video(video_bytes)

mode = st.radio("Sampling mode", ["every_n_frames", "every_n_seconds"], horizontal=True)

col1, col2, col3 = st.columns(3)

if mode == "every_n_frames":
    every_n_frames = col1.number_input("every_n_frames", min_value=1, value=30, step=1)
else:
    every_n_seconds = col1.number_input("every_n_seconds", min_value=0.1, value=1.0, step=0.1)
    fps = get_fps(video_bytes, os.path.splitext(video_file.name)[1] or ".mp4") if video_bytes else None
    if fps:
        every_n_frames = max(1, int(round(fps * float(every_n_seconds))))
        col1.caption(f"FPS ≈ {fps:.1f} → every_n_frames = {every_n_frames}")
    else:
        every_n_frames = 30
        col1.caption("Cannot read FPS → fallback every_n_frames=30")
max_frames = col2.number_input("max_frames", min_value=1, value=30, step=1)
conf = col3.slider("conf", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Result area
st.divider()
result_placeholder = st.empty()

can_detect = (video_file is not None)

if st.button("Detect", type="primary", disabled=not can_detect):
    files = {"video": (video_file.name, video_bytes, "application/octet-stream")}
    params = {"every_n_frames": int(every_n_frames), "max_frames": int(max_frames), "conf": float(conf)}

    try:
        with st.spinner("Running detection..."):
            t0 = time.perf_counter()
            r = requests.post(f"{API_URL}/detect/video", files=files, params=params, timeout=300)
            dt = time.perf_counter() - t0

        if r.ok:
            data = r.json()
            result_placeholder.success("Done")
            c1, c2 = st.columns(2)
            c1.metric("Latency (s)", f"{dt:.2f}")
            c2.metric("Sampled frames", int(data.get("sampled_frames", 0)))
            st.subheader("Detected labels")
            st.write(data.get("labels", []))

            st.subheader("Frame hits (per-frame)")
            frame_hits = data.get("frame_hits", {})
            frame_hits_sorted = dict(sorted(frame_hits.items(), key=lambda x: x[1], reverse=True))
            st.json(frame_hits_sorted)

            st.subheader("Raw response")
            st.json(data)
        else:
            result_placeholder.error(f"API error {r.status_code}")
            st.text(r.text)
    except RequestException as e:
        result_placeholder.error(
            "Cannot reach API. Start it with: python -m uvicorn src.api.main:app --reload --port 8000"
        )
        st.code(str(e))
elif video_file is None:
    result_placeholder.info("Upload a video to begin.")
