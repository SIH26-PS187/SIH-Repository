server.py — drop this into your backend/ folder alongside testpipeline.py
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Requirements:
    pip install fastapi uvicorn opencv-python ultralytics
"""

import asyncio
import json
import time
import threading
from datetime import datetime
from typing import Generator

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# ── app setup ──────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model ──────────────────────────────────────────────────────────────
model = YOLO("yolo11s.pt")

# ── shared state (updated by detection thread) ─────────────────────────
state = {
    "tracks": [],
    "alerts": [],
    "alert_id": 0,
    "frame": None,
    "stats": {
        "active_cameras": 1,
        "total_cameras": 1,
        "threats_today": 0,
        "persons_tracked": 0,
        "uptime_pct": 99.9,
        "avg_latency_ms": 0,
        "alerts_24h": 0,
    },
}
state_lock = threading.Lock()

# ── queues for WebSocket push ──────────────────────────────────────────
alert_subscribers: list[asyncio.Queue] = []
track_subscribers: list[asyncio.Queue] = []

def push_to_all(subscribers: list[asyncio.Queue], payload):
    for q in subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

# ── detection thread ───────────────────────────────────────────────────
def detection_loop():
    cap = cv2.VideoCapture(0)   # change to RTSP URL string for real camera
    track_counter = {}

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue

        t0 = time.time()
        results = model.track(frame, persist=True, verbose=False)
        latency_ms = int((time.time() - t0) * 1000)

        annotated = results[0].plot()
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        jpeg_bytes = buf.tobytes()

        live_tracks = []
        now_str = datetime.now().strftime("%H:%M:%S")

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = round(float(box.conf[0]) * 100, 1)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_str = f"{x2-x1}x{y2-y1}px"

                tid = int(box.id[0]) if box.id is not None else 0
                tid_str = f"TRK-{tid:03d}"

                if tid not in track_counter:
                    track_counter[tid] = time.time()
                duration_s = int(time.time() - track_counter[tid])

                if cls_name == "person":
                    live_tracks.append({
                        "track_id": tid_str,
                        "camera": "CAM-01",
                        "zone": "Live Feed",
                        "duration_s": duration_s,
                        "confidence": conf,
                        "bbox": bbox_str,
                        "classification": "Civilian",
                    })

                    with state_lock:
                        existing_ids = {a.get("track_ref") for a in state["alerts"]}
                        if tid_str not in existing_ids:
                            state["alert_id"] += 1
                            alert = {
                                "id": str(state["alert_id"]),
                                "time": now_str,
                                "camera": "CAM-01",
                                "type": "Person Detected",
                                "severity": "medium",
                                "location": "Live Feed",
                                "status": "active",
                                "track_ref": tid_str,
                            }
                            state["alerts"].insert(0, alert)
                            state["alerts"] = state["alerts"][:200]
                            state["stats"]["alerts_24h"] += 1
                            push_to_all(alert_subscribers, json.dumps(alert))

        active_ids = {t["track_id"] for t in live_tracks}
        track_counter = {k: v for k, v in track_counter.items()
                         if f"TRK-{k:03d}" in active_ids}

        with state_lock:
            state["frame"] = jpeg_bytes
            state["tracks"] = live_tracks
            state["stats"]["persons_tracked"] = len(live_tracks)
            state["stats"]["avg_latency_ms"] = latency_ms
            if len(live_tracks) > 0:
                state["stats"]["threats_today"] = max(
                    state["stats"]["threats_today"], len(live_tracks)
                )

        push_to_all(track_subscribers, json.dumps(live_tracks))

    cap.release()


threading.Thread(target=detection_loop, daemon=True).start()


# ── MJPEG stream ───────────────────────────────────────────────────────
def mjpeg_generator() -> Generator[bytes, None, None]:
    while True:
        with state_lock:
            frame = state["frame"]
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        time.sleep(0.033)

@app.get("/api/cameras/CAM-01/stream")
def camera_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# ── REST endpoints ─────────────────────────────────────────────────────
@app.get("/api/alerts")
def get_alerts():
    with state_lock:
        return state["alerts"]

@app.get("/api/vehicles")
def get_vehicles():
    return []

@app.get("/api/tracks")
def get_tracks():
    with state_lock:
        return state["tracks"]

@app.get("/api/zones")
def get_zones():
    return []

@app.get("/api/cameras")
def get_cameras():
    with state_lock:
        person_count = len(state["tracks"])
        has_alert = any(t["classification"] in ("Intruder", "Suspect")
                        for t in state["tracks"])
    return [{
        "id": "CAM-01",
        "label": "Webcam — Live",
        "online": True,
        "alert": has_alert,
        "person_count": person_count,
    }]

@app.get("/api/stats")
def get_stats():
    with state_lock:
        return state["stats"]

# ── WebSocket: alerts ──────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    alert_subscribers.append(q)
    try:
        while True:
            msg = await q.get()
            await ws.send_text(msg)
    except WebSocketDisconnect:
        alert_subscribers.remove(q)

# ── WebSocket: tracks ──────────────────────────────────────────────────
@app.websocket("/ws/tracks")
async def ws_tracks(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    track_subscribers.append(q)
    try:
        while True:
            msg = await q.get()
            await ws.send_text(msg)
    except WebSocketDisconnect:
        track_subscribers.remove(q)

# ── WebSocket: vehicles (stub) ─────────────────────────────────────────
@app.websocket("/ws/vehicles")
async def ws_vehicles(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass

# ── WebSocket: zones (stub) ────────────────────────────────────────────
@app.websocket("/ws/zones")
async def ws_zones(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps([]))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
