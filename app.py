"""
╔══════════════════════════════════════════════════════════════╗
║     AI PPE Detection System for Construction Safety          ║
║     Single-file Flask Application                            ║
║     Final Year B.Tech Project                                ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install flask opencv-python ultralytics numpy pillow

Run:
    python app.py
    Open: http://127.0.0.1:5000
"""

import cv2
import numpy as np
import threading
import time
import base64
import json
import os
import logging
from io import BytesIO
from flask import Flask, Response, render_template_string, jsonify, request
from collections import deque

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Try loading YOLOv8 ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO loaded successfully.")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not installed. Running in DEMO mode (simulated detections).")

app = Flask(__name__)

# ─── Global State ─────────────────────────────────────────────────────────────
detection_state = {
    "frame":            None,
    "alerts":           deque(maxlen=50),
    "counts":           {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "person": 0},
    "running":          False,
    "source":           None,
    "fps":              0,
    "model_loaded":     False,
    "frame_count":      0,
    "lock":             threading.Lock(),
    "latest_detections": [],
}

model = None

# ─── YOLO Model Loader ────────────────────────────────────────────────────────
def load_model():
    """Load YOLOv8 model. Falls back to base yolov8n if custom not found."""
    global model
    if not YOLO_AVAILABLE:
        detection_state["model_loaded"] = False
        return
    try:
        # Check for custom-trained model first
        custom_path = "ppe_model.pt"
        if os.path.exists(custom_path):
            model = YOLO(custom_path)
            logger.info(f"Loaded custom model: {custom_path}")
        else:
            model = YOLO("yolov8n.pt")
            logger.info("Loaded base YOLOv8n model (fallback). For best results, train a custom PPE model.")
        detection_state["model_loaded"] = True
    except Exception as e:
        logger.error(f"Model load error: {e}")
        detection_state["model_loaded"] = False

# ─── PPE Class Mapping ────────────────────────────────────────────────────────
# Adjust these to match your custom-trained model's class names
PPE_CLASS_MAP = {
    "helmet":       {"color": (0, 220, 0),   "safe": True},
    "no_helmet":    {"color": (0, 0, 220),   "safe": False},
    "vest":         {"color": (0, 200, 100), "safe": True},
    "no_vest":      {"color": (0, 0, 200),   "safe": False},
    "person":       {"color": (200, 200, 0), "safe": True},
    # YOLOv8 base COCO classes fallback
    "person_coco":  {"color": (200, 200, 0), "safe": True},
}

# ─── Draw Bounding Box ────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, conf, safe):
    color = (0, 200, 0) if safe else (0, 30, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

# ─── Real Detection ───────────────────────────────────────────────────────────
def run_detection(frame):
    """Run YOLO inference and annotate frame. Returns annotated frame + alerts."""
    alerts = []
    counts = {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "person": 0}
    detections = []

    if model is None:
        return frame, alerts, counts, detections

    results = model(
        frame,
        conf=0.35,          # Confidence threshold
        iou=0.45,           # NMS IoU threshold
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls].lower().replace(" ", "_")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            safe = True

            # ── Custom PPE model classes ──────────────────────────────────────
            if label in ("helmet", "hard_hat", "hardhat"):
                counts["helmet"] += 1
                draw_box(frame, x1, y1, x2, y2, "Helmet ✓", conf, True)
                detections.append({"label": "helmet", "conf": conf, "safe": True})

            elif label in ("no_helmet", "no_hard_hat", "without_helmet"):
                counts["no_helmet"] += 1
                alerts.append({"msg": "⚠ No Helmet Detected!", "level": "danger", "time": time.strftime("%H:%M:%S")})
                draw_box(frame, x1, y1, x2, y2, "NO HELMET!", conf, False)
                detections.append({"label": "no_helmet", "conf": conf, "safe": False})

            elif label in ("vest", "safety_vest", "reflective_vest"):
                counts["vest"] += 1
                draw_box(frame, x1, y1, x2, y2, "Vest ✓", conf, True)
                detections.append({"label": "vest", "conf": conf, "safe": True})

            elif label in ("no_vest", "without_vest", "no_safety_vest"):
                counts["no_vest"] += 1
                alerts.append({"msg": "⚠ No Safety Vest Detected!", "level": "warning", "time": time.strftime("%H:%M:%S")})
                draw_box(frame, x1, y1, x2, y2, "NO VEST!", conf, False)
                detections.append({"label": "no_vest", "conf": conf, "safe": False})

            elif label in ("person",):
                counts["person"] += 1
                draw_box(frame, x1, y1, x2, y2, "Person", conf, True)
                detections.append({"label": "person", "conf": conf, "safe": True})

            # ── COCO fallback: only 'person' class ───────────────────────────
            else:
                if cls == 0:   # COCO person class
                    counts["person"] += 1
                    # In COCO mode: flag every person as potentially missing PPE
                    alerts.append({"msg": "⚠ Person: PPE status unknown (no custom model)", "level": "info", "time": time.strftime("%H:%M:%S")})
                    draw_box(frame, x1, y1, x2, y2, "Person?", conf, False)
                    detections.append({"label": "person", "conf": conf, "safe": None})

    return frame, alerts, counts, detections

# ─── Demo / Simulation Mode ───────────────────────────────────────────────────
_demo_phase = 0

def run_demo_detection(frame):
    """
    Simulate detections when YOLO is unavailable.
    Cycles through: safe → no_helmet → no_vest states.
    """
    global _demo_phase
    alerts = []
    counts = {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "person": 0}
    detections = []
    h, w = frame.shape[:2]
    _demo_phase = (detection_state["frame_count"] // 60) % 3

    # Draw simulated person box
    px1, py1 = w // 4, h // 5
    px2, py2 = 3 * w // 4, 9 * h // 10
    counts["person"] += 1

    if _demo_phase == 0:
        # Safe scenario
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)
        cv2.putText(frame, "Person - SAFE (DEMO)", (px1, py1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        draw_box(frame, px1 + 20, py1, px2 - 20, py1 + 60, "Helmet ✓", 0.93, True)
        draw_box(frame, px1, py1 + 70, px2, py2, "Vest ✓", 0.87, True)
        counts["helmet"] += 1
        counts["vest"] += 1
        detections = [
            {"label": "helmet", "conf": 0.93, "safe": True},
            {"label": "vest",   "conf": 0.87, "safe": True},
        ]
    elif _demo_phase == 1:
        # No helmet
        counts["no_helmet"] += 1
        counts["vest"] += 1
        draw_box(frame, px1 + 20, py1, px2 - 20, py1 + 60, "NO HELMET!", 0.89, False)
        draw_box(frame, px1, py1 + 70, px2, py2, "Vest ✓", 0.84, True)
        alerts.append({"msg": "⚠ No Helmet Detected!", "level": "danger", "time": time.strftime("%H:%M:%S")})
        detections = [
            {"label": "no_helmet", "conf": 0.89, "safe": False},
            {"label": "vest",      "conf": 0.84, "safe": True},
        ]
    else:
        # No vest
        counts["helmet"] += 1
        counts["no_vest"] += 1
        draw_box(frame, px1 + 20, py1, px2 - 20, py1 + 60, "Helmet ✓", 0.91, True)
        draw_box(frame, px1, py1 + 70, px2, py2, "NO VEST!", 0.82, False)
        alerts.append({"msg": "⚠ No Safety Vest Detected!", "level": "warning", "time": time.strftime("%H:%M:%S")})
        detections = [
            {"label": "helmet",  "conf": 0.91, "safe": True},
            {"label": "no_vest", "conf": 0.82, "safe": False},
        ]

    # Demo watermark
    cv2.putText(frame, "[ DEMO MODE - No YOLO Model ]", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
    return frame, alerts, counts, detections

# ─── Overlay Helpers ──────────────────────────────────────────────────────────
def draw_overlay(frame, fps, counts):
    """Draw HUD overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, "AI PPE DETECTION", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Status bar bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 32), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

    status_parts = [
        f"Persons: {counts['person']}",
        f"Helmets: {counts['helmet']}",
        f"No-Helmet: {counts['no_helmet']}",
        f"Vests: {counts['vest']}",
        f"No-Vest: {counts['no_vest']}",
    ]
    cv2.putText(frame, "  |  ".join(status_parts), (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame

# ─── Video Processing Thread ──────────────────────────────────────────────────
def process_stream(source):
    """Main video capture + inference loop."""
    state = detection_state

    # Determine source
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source.startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        with state["lock"]:
            state["running"] = False
        return

    fps_timer = time.time()
    frames_since_fps = 0

    while True:
        with state["lock"]:
            if not state["running"]:
                break

        ret, frame = cap.read()
        if not ret:
            # Loop video file
            if source not in ("webcam",) and not source.startswith("rtsp://"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Resize for performance
        frame = cv2.resize(frame, (854, 480))

        # Run detection
        try:
            if YOLO_AVAILABLE and state["model_loaded"]:
                frame, new_alerts, counts, detections = run_detection(frame)
            else:
                frame, new_alerts, counts, detections = run_demo_detection(frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            new_alerts, counts, detections = [], {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "person": 0}, []

        # FPS calc
        frames_since_fps += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frames_since_fps / elapsed
            fps_timer = time.time()
            frames_since_fps = 0
        else:
            fps = state["fps"]

        # Overlay HUD
        frame = draw_overlay(frame, fps, counts)

        # Encode frame to JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with state["lock"]:
            state["frame"]       = buf.tobytes()
            state["fps"]         = fps
            state["counts"]      = counts
            state["frame_count"] += 1
            state["latest_detections"] = detections
            for a in new_alerts:
                state["alerts"].appendleft(a)

    cap.release()
    with state["lock"]:
        state["running"] = False
        state["frame"]   = None
    logger.info("Stream stopped.")

# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/start", methods=["POST"])
def start_stream():
    data   = request.get_json(silent=True) or {}
    source = data.get("source", "webcam")

    with detection_state["lock"]:
        if detection_state["running"]:
            detection_state["running"] = False
            time.sleep(0.4)
        detection_state["running"]    = True
        detection_state["source"]     = source
        detection_state["alerts"]     = deque(maxlen=50)
        detection_state["frame_count"] = 0

    t = threading.Thread(target=process_stream, args=(source,), daemon=True)
    t.start()
    return jsonify({"status": "started", "source": source,
                    "yolo_available": YOLO_AVAILABLE,
                    "model_loaded": detection_state["model_loaded"]})


@app.route("/api/stop", methods=["POST"])
def stop_stream():
    with detection_state["lock"]:
        detection_state["running"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/status")
def status():
    with detection_state["lock"]:
        return jsonify({
            "running":          detection_state["running"],
            "fps":              round(detection_state["fps"], 1),
            "counts":           detection_state["counts"],
            "frame_count":      detection_state["frame_count"],
            "alerts":           list(detection_state["alerts"])[:10],
            "model_loaded":     detection_state["model_loaded"],
            "yolo_available":   YOLO_AVAILABLE,
            "latest_detections": detection_state["latest_detections"],
        })


def gen_frames():
    """MJPEG stream generator."""
    blank = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for stream...", (250, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
    _, blank_buf = cv2.imencode(".jpg", blank)
    blank_bytes  = blank_buf.tobytes()

    while True:
        with detection_state["lock"]:
            frame = detection_state["frame"]
        data = frame if frame else blank_bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)   # ~30 fps max


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── HTML / CSS / JS Template ─────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI PPE Detection System</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
/* ── Reset & Variables ───────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --y:  #FFD600;
  --y2: #FFA000;
  --y3: #FF6F00;
  --k:  #0A0A0A;
  --k2: #111111;
  --k3: #1A1A1A;
  --k4: #242424;
  --w:  #F5F5F5;
  --g:  #888;
  --danger:  #FF3B3B;
  --warn:    #FFB800;
  --ok:      #00E676;
  --info:    #29B6F6;
  --r: 6px;
  --glow: 0 0 18px rgba(255,214,0,.35);
}
html,body{height:100%;background:var(--k);color:var(--w);
  font-family:'Barlow',sans-serif;overflow-x:hidden}

/* ── Particle Canvas ─────────────────────────────────────── */
#particles{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.45}

/* ── Animated gradient noise bg ─────────────────────────── */
body::before{content:'';position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse 80% 60% at 20% 10%,rgba(255,214,0,.08) 0%,transparent 60%),
             radial-gradient(ellipse 60% 80% at 80% 90%,rgba(255,111,0,.07) 0%,transparent 60%);
  animation:bgPulse 8s ease-in-out infinite alternate;pointer-events:none}
@keyframes bgPulse{to{opacity:.5;transform:scale(1.04)}}

/* ── Layout ──────────────────────────────────────────────── */
.app{position:relative;z-index:1;min-height:100vh;
  display:grid;grid-template-rows:auto 1fr auto;gap:0}

/* ── Header ──────────────────────────────────────────────── */
header{
  background:linear-gradient(90deg,var(--k2) 0%,var(--k3) 50%,var(--k2) 100%);
  border-bottom:2px solid var(--y);
  padding:0 2rem;
  display:flex;align-items:center;justify-content:space-between;
  height:72px;position:sticky;top:0;z-index:100;
  box-shadow:0 2px 30px rgba(255,214,0,.15)
}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{width:44px;height:44px;background:var(--y);border-radius:8px;
  display:grid;place-items:center;font-size:1.4rem;
  box-shadow:var(--glow);animation:iconPulse 3s ease-in-out infinite}
@keyframes iconPulse{0%,100%{box-shadow:var(--glow)}50%{box-shadow:0 0 32px rgba(255,214,0,.7)}}
h1{font-family:'Bebas Neue',sans-serif;font-size:1.55rem;letter-spacing:.08em;
   line-height:1.1;color:var(--y)}
h1 span{display:block;font-family:'Barlow Condensed',sans-serif;font-size:.72rem;
  font-weight:600;letter-spacing:.25em;color:var(--g);text-transform:uppercase}
.badge{background:var(--y);color:var(--k);font-size:.65rem;font-weight:700;
  padding:2px 8px;border-radius:20px;letter-spacing:.1em;animation:badgeBlink 2s step-start infinite}
@keyframes badgeBlink{0%,100%{opacity:1}50%{opacity:.5}}

/* ── Main Grid ───────────────────────────────────────────── */
main{display:grid;grid-template-columns:1fr 320px;gap:1.25rem;
  padding:1.25rem 1.5rem;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}

/* ── Video Panel ─────────────────────────────────────────── */
.video-panel{display:flex;flex-direction:column;gap:1rem}
.video-wrap{
  position:relative;border-radius:var(--r);overflow:hidden;
  border:2px solid #2a2a2a;background:#000;
  box-shadow:0 4px 40px rgba(0,0,0,.6);
  aspect-ratio:16/9
}
.video-wrap img{width:100%;height:100%;object-fit:cover;display:block}
.video-overlay-badge{
  position:absolute;top:10px;right:10px;
  background:rgba(0,0,0,.75);border:1px solid var(--y);
  border-radius:4px;padding:3px 10px;
  font-family:'Barlow Condensed',sans-serif;font-size:.8rem;letter-spacing:.12em;
  color:var(--y);backdrop-filter:blur(4px)
}
.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--danger);margin-right:5px;
  animation:liveDot 1s step-start infinite}
@keyframes liveDot{50%{opacity:0}}
.cam-offline{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:8px;background:#050505;
  font-family:'Barlow Condensed',sans-serif;letter-spacing:.1em;color:#444;
  font-size:1rem;transition:opacity .4s}
.cam-offline.hidden{opacity:0;pointer-events:none}
.cam-offline .icon{font-size:3rem;opacity:.3}

/* ── Controls Bar ────────────────────────────────────────── */
.controls{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center}
.source-select{
  flex:1;min-width:180px;background:var(--k3);border:1px solid #333;
  color:var(--w);padding:.5rem .9rem;border-radius:var(--r);
  font-family:'Barlow',sans-serif;font-size:.9rem;cursor:pointer}
.source-select:focus{outline:1px solid var(--y)}
#rtspInput{display:none;flex:2;background:var(--k3);border:1px solid #333;
  color:var(--w);padding:.5rem .9rem;border-radius:var(--r);
  font-family:'Barlow',sans-serif;font-size:.85rem}
#rtspInput:focus{outline:1px solid var(--y)}
#fileInput{display:none}
.file-label{cursor:pointer;background:var(--k4);border:1px solid #333;
  color:var(--g);padding:.5rem .9rem;border-radius:var(--r);
  font-size:.85rem;transition:border-color .2s}
.file-label:hover{border-color:var(--y);color:var(--y)}

.btn{padding:.5rem 1.2rem;border:none;border-radius:var(--r);cursor:pointer;
  font-family:'Barlow Condensed',sans-serif;font-size:.95rem;font-weight:600;
  letter-spacing:.06em;transition:all .2s;text-transform:uppercase}
.btn-start{background:var(--y);color:var(--k)}
.btn-start:hover{background:var(--y2);box-shadow:var(--glow)}
.btn-stop{background:transparent;color:var(--danger);border:1px solid var(--danger)}
.btn-stop:hover{background:rgba(255,59,59,.1)}
.btn:disabled{opacity:.35;cursor:not-allowed}

/* ── Stats Row ───────────────────────────────────────────── */
.stats-row{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem}
.stat-card{
  background:var(--k3);border-radius:var(--r);
  border:1px solid #252525;padding:.7rem .5rem;
  text-align:center;transition:border-color .3s}
.stat-card.danger{border-color:var(--danger)}
.stat-card.ok{border-color:var(--ok)}
.stat-val{font-family:'Bebas Neue',sans-serif;font-size:2rem;
  line-height:1;color:var(--y)}
.stat-lbl{font-size:.65rem;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;color:var(--g);margin-top:2px}

/* ── Right Panel ─────────────────────────────────────────── */
.side-panel{display:flex;flex-direction:column;gap:1rem}

/* ── Model Status ─────────────────────────────────────────── */
.model-status{
  background:var(--k3);border-radius:var(--r);border:1px solid #252525;
  padding:1rem;display:flex;flex-direction:column;gap:.5rem}
.model-row{display:flex;justify-content:space-between;align-items:center;
  font-size:.8rem}
.model-row span:first-child{color:var(--g);text-transform:uppercase;
  letter-spacing:.1em;font-size:.7rem}
.pill{padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:700;
  letter-spacing:.08em}
.pill.ok{background:rgba(0,230,118,.15);color:var(--ok);border:1px solid rgba(0,230,118,.3)}
.pill.warn{background:rgba(255,184,0,.15);color:var(--warn);border:1px solid rgba(255,184,0,.3)}
.pill.off{background:rgba(136,136,136,.1);color:var(--g);border:1px solid #333}
.conf-bar-wrap{margin-top:.2rem}
.conf-bar-wrap label{font-size:.7rem;color:var(--g);letter-spacing:.08em;
  text-transform:uppercase;display:flex;justify-content:space-between}
.conf-bar{height:5px;background:#222;border-radius:3px;margin-top:4px;overflow:hidden}
.conf-bar-fill{height:100%;background:linear-gradient(90deg,var(--y2),var(--y));
  border-radius:3px;transition:width .4s ease}

/* ── Alerts Panel ────────────────────────────────────────── */
.alerts-panel{
  background:var(--k3);border-radius:var(--r);border:1px solid #252525;
  flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:300px}
.panel-header{
  padding:.75rem 1rem;border-bottom:1px solid #252525;
  display:flex;justify-content:space-between;align-items:center}
.panel-title{font-family:'Barlow Condensed',sans-serif;font-size:.9rem;
  font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--y)}
.alert-count{background:var(--danger);color:#fff;border-radius:20px;
  font-size:.65rem;font-weight:700;padding:1px 7px;min-width:20px;text-align:center}
.alerts-list{flex:1;overflow-y:auto;padding:.5rem;
  display:flex;flex-direction:column;gap:.4rem;max-height:400px}
.alerts-list::-webkit-scrollbar{width:4px}
.alerts-list::-webkit-scrollbar-track{background:#111}
.alerts-list::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
.alert-item{
  padding:.55rem .75rem;border-radius:4px;display:flex;flex-direction:column;gap:2px;
  animation:slideIn .25s ease;font-size:.8rem}
@keyframes slideIn{from{transform:translateX(20px);opacity:0}to{transform:none;opacity:1}}
.alert-item.danger{background:rgba(255,59,59,.12);border-left:3px solid var(--danger)}
.alert-item.warning{background:rgba(255,184,0,.1);border-left:3px solid var(--warn)}
.alert-item.info{background:rgba(41,182,246,.08);border-left:3px solid var(--info)}
.alert-msg{font-weight:600;color:var(--w)}
.alert-time{font-size:.65rem;color:var(--g)}
.no-alerts{text-align:center;padding:2rem;color:var(--g);font-size:.8rem}

/* ── Detection Details ───────────────────────────────────── */
.det-panel{
  background:var(--k3);border-radius:var(--r);border:1px solid #252525;padding:.85rem}
.det-title{font-family:'Barlow Condensed',sans-serif;font-size:.85rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;color:var(--y);margin-bottom:.7rem}
.det-list{display:flex;flex-direction:column;gap:.4rem}
.det-item{display:flex;align-items:center;justify-content:space-between;font-size:.78rem}
.det-label{display:flex;align-items:center;gap:.5rem;color:var(--w)}
.det-dot{width:8px;height:8px;border-radius:50%}
.det-conf{font-family:'Barlow Condensed',sans-serif;font-size:.85rem;
  letter-spacing:.06em;font-weight:600}
.det-conf.ok{color:var(--ok)} .det-conf.bad{color:var(--danger)} .det-conf.unk{color:var(--warn)}

/* ── FPS Meter ───────────────────────────────────────────── */
.fps-bar{background:var(--k3);border-radius:var(--r);border:1px solid #252525;
  padding:.6rem 1rem;display:flex;align-items:center;gap:1rem;justify-content:space-between}
.fps-val{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:var(--y);line-height:1}
.fps-lbl{font-size:.65rem;color:var(--g);letter-spacing:.1em;text-transform:uppercase}
.fps-meter{flex:1;height:6px;background:#1a1a1a;border-radius:3px;overflow:hidden}
.fps-fill{height:100%;background:linear-gradient(90deg,#FF3B3B,var(--y),var(--ok));
  border-radius:3px;transition:width .5s ease}

/* ── Contact ──────────────────────────────────────────────── */
.contact-section{
  background:var(--k3);border-radius:var(--r);border:1px solid #252525;padding:.9rem 1rem}
.contact-title{font-family:'Barlow Condensed',sans-serif;font-size:.85rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;color:var(--y);margin-bottom:.75rem}
.contact-links{display:flex;flex-direction:column;gap:.5rem}
.contact-link{display:flex;align-items:center;gap:.6rem;font-size:.78rem;color:var(--g);
  text-decoration:none;transition:color .2s;padding:.25rem 0}
.contact-link:hover{color:var(--y)}
.contact-icon{width:28px;height:28px;border-radius:5px;display:grid;place-items:center;
  font-size:.85rem;flex-shrink:0}
.ci-email{background:rgba(255,214,0,.12);color:var(--y)}
.ci-phone{background:rgba(0,230,118,.12);color:var(--ok)}
.ci-li{background:rgba(10,102,194,.25);color:#0a66c2}

/* ── Footer ──────────────────────────────────────────────── */
footer{border-top:1px solid #1f1f1f;padding:.75rem 1.5rem;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem}
.footer-copy{font-size:.72rem;color:#444;letter-spacing:.05em}
.footer-badge{font-size:.7rem;color:var(--y);letter-spacing:.12em;
  text-transform:uppercase;font-family:'Barlow Condensed',sans-serif}

/* ── Loading spinner ─────────────────────────────────────── */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #333;
  border-top-color:var(--y);border-radius:50%;animation:spin .7s linear infinite;
  vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Toast ───────────────────────────────────────────────── */
.toast{
  position:fixed;bottom:1.5rem;right:1.5rem;z-index:999;
  background:var(--k4);border:1px solid var(--y);border-radius:var(--r);
  padding:.7rem 1.2rem;font-size:.82rem;font-weight:500;color:var(--w);
  box-shadow:0 4px 24px rgba(0,0,0,.5);max-width:320px;
  animation:toastIn .3s ease;display:none}
.toast.show{display:block}
@keyframes toastIn{from{transform:translateY(20px);opacity:0}to{transform:none;opacity:1}}

/* ── Responsive ──────────────────────────────────────────── */
@media(max-width:600px){
  .stats-row{grid-template-columns:repeat(3,1fr)}
  h1{font-size:1.1rem}
  .controls{flex-direction:column}
  .btn{width:100%}
}
</style>
</head>
<body>
<canvas id="particles"></canvas>
<div class="app">

<!-- ════════════ HEADER ════════════ -->
<header>
  <div class="logo">
    <div class="logo-icon">🦺</div>
    <h1>AI PPE Detection System
      <span>Construction Site Safety · Drone Monitoring</span>
    </h1>
  </div>
  <span class="badge" id="modeBadge">DEMO</span>
</header>

<!-- ════════════ MAIN ════════════ -->
<main>

  <!-- Left: Video + Stats -->
  <div class="video-panel">

    <!-- Video Feed -->
    <div class="video-wrap">
      <img id="videoFeed" src="/video_feed" alt="Video Feed"/>
      <div class="video-overlay-badge">
        <span class="live-dot"></span><span id="streamLabel">STREAM</span>
      </div>
      <div class="cam-offline" id="offlinePlaceholder">
        <div class="icon">📷</div>
        <div>No Active Stream</div>
        <div style="font-size:.7rem;margin-top:4px">Select source and click Start</div>
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <select class="source-select" id="sourceSelect" onchange="onSourceChange()">
        <option value="webcam">📷  Webcam (Camera 0)</option>
        <option value="file">📁  Video File Upload</option>
        <option value="rtsp">📡  RTSP Drone Stream</option>
      </select>
      <input type="text" id="rtspInput" placeholder="rtsp://192.168.1.1:554/stream"/>
      <label class="file-label" id="fileLabel" for="fileInput">Choose file</label>
      <input type="file" id="fileInput" accept="video/*" onchange="onFileChange()"/>
      <button class="btn btn-start" id="btnStart" onclick="startStream()">▶ Start</button>
      <button class="btn btn-stop"  id="btnStop"  onclick="stopStream()" disabled>■ Stop</button>
    </div>

    <!-- Stats -->
    <div class="stats-row">
      <div class="stat-card" id="sc-person">
        <div class="stat-val" id="cnt-person">0</div>
        <div class="stat-lbl">Persons</div>
      </div>
      <div class="stat-card ok" id="sc-helmet">
        <div class="stat-val" id="cnt-helmet">0</div>
        <div class="stat-lbl">Helmets ✓</div>
      </div>
      <div class="stat-card" id="sc-nohelmet">
        <div class="stat-val" id="cnt-nohelmet" style="color:var(--danger)">0</div>
        <div class="stat-lbl">No Helmet ⚠</div>
      </div>
      <div class="stat-card ok" id="sc-vest">
        <div class="stat-val" id="cnt-vest">0</div>
        <div class="stat-lbl">Vests ✓</div>
      </div>
      <div class="stat-card" id="sc-novest">
        <div class="stat-val" id="cnt-novest" style="color:var(--warn)">0</div>
        <div class="stat-lbl">No Vest ⚠</div>
      </div>
    </div>

    <!-- FPS Meter -->
    <div class="fps-bar">
      <div>
        <div class="fps-val" id="fpsVal">0</div>
        <div class="fps-lbl">FPS</div>
      </div>
      <div class="fps-meter">
        <div class="fps-fill" id="fpsFill" style="width:0%"></div>
      </div>
      <div id="fpsStatus" style="font-size:.72rem;color:var(--g);letter-spacing:.08em">IDLE</div>
    </div>

  </div><!-- /video-panel -->

  <!-- Right: Side Panel -->
  <div class="side-panel">

    <!-- Model Status -->
    <div class="model-status">
      <div class="panel-title" style="font-family:'Barlow Condensed',sans-serif;font-size:.85rem;
        font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--y);margin-bottom:.3rem">
        System Status
      </div>
      <div class="model-row">
        <span>YOLO Engine</span>
        <span class="pill" id="pillYolo">Checking...</span>
      </div>
      <div class="model-row">
        <span>Model Loaded</span>
        <span class="pill" id="pillModel">—</span>
      </div>
      <div class="model-row">
        <span>Stream</span>
        <span class="pill" id="pillStream">IDLE</span>
      </div>
      <div class="conf-bar-wrap">
        <label>Confidence Threshold <span>35%</span></label>
        <div class="conf-bar"><div class="conf-bar-fill" style="width:35%"></div></div>
      </div>
      <div class="conf-bar-wrap">
        <label>NMS IoU Threshold <span>45%</span></label>
        <div class="conf-bar"><div class="conf-bar-fill" style="width:45%"></div></div>
      </div>
    </div>

    <!-- Live Detections -->
    <div class="det-panel">
      <div class="det-title">Live Detections</div>
      <div class="det-list" id="detList">
        <div class="no-alerts">No detections yet</div>
      </div>
    </div>

    <!-- Alerts -->
    <div class="alerts-panel">
      <div class="panel-header">
        <span class="panel-title">⚠ Alert Log</span>
        <span class="alert-count" id="alertCount">0</span>
      </div>
      <div class="alerts-list" id="alertsList">
        <div class="no-alerts">No alerts — all clear ✓</div>
      </div>
    </div>

    <!-- Contact -->
    <div class="contact-section">
      <div class="contact-title">📬 Developer Contact</div>
      <div class="contact-links">
        <a class="contact-link" href="mailto:avishkarsarang777@gmail.com">
          <div class="contact-icon ci-email">✉</div>
          avishkarsarang777@gmail.com
        </a>
        <a class="contact-link" href="tel:+917588943907">
          <div class="contact-icon ci-phone">📞</div>
          +91 75889 43907
        </a>
        <a class="contact-link" href="https://www.linkedin.com/in/avishkar-sarang-03b107333" target="_blank">
          <div class="contact-icon ci-li">in</div>
          Avishkar Sarang — LinkedIn
        </a>
      </div>
    </div>

  </div><!-- /side-panel -->
</main>

<!-- ════════════ FOOTER ════════════ -->
<footer>
  <span class="footer-copy">
    AI PPE Detection System · Final Year B.Tech Project · Civil &amp; Infrastructure Engineering
  </span>
  <span class="footer-badge">YOLOv8 · Flask · OpenCV</span>
</footer>

</div><!-- /app -->

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- ═══════════════════════════ SCRIPTS ══════════════════════════ -->
<script>
/* ── Particle System ─────────────────────────────────────── */
(function(){
  const c=document.getElementById('particles'),ctx=c.getContext('2d');
  let pts=[];const N=55;
  function resize(){c.width=innerWidth;c.height=innerHeight;init()}
  function init(){pts=[];for(let i=0;i<N;i++)pts.push({
    x:Math.random()*c.width,y:Math.random()*c.height,
    vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,
    r:Math.random()*2+1,a:Math.random()
  })}
  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    pts.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>c.width)p.vx*=-1;
      if(p.y<0||p.y>c.height)p.vy*=-1;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(255,214,0,${p.a*.5})`;ctx.fill();
    });
    // Lines
    for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){
      const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<120){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);
        ctx.strokeStyle=`rgba(255,214,0,${(1-d/120)*.08})`;ctx.lineWidth=1;ctx.stroke()}
    }
    requestAnimationFrame(draw);
  }
  addEventListener('resize',resize);resize();draw();
})();

/* ── State ────────────────────────────────────────────────── */
let isRunning=false, seenAlerts=new Set(), totalAlerts=0;

/* ── Source Change ────────────────────────────────────────── */
function onSourceChange(){
  const v=document.getElementById('sourceSelect').value;
  document.getElementById('rtspInput').style.display   = v==='rtsp'?'flex':'none';
  document.getElementById('fileLabel').style.display   = v==='file'?'inline-flex':'none';
  document.getElementById('fileInput').style.display   = 'none';
}

function onFileChange(){
  const f=document.getElementById('fileInput').files[0];
  if(f)document.getElementById('fileLabel').textContent='📁 '+f.name;
}

/* ── Start / Stop ─────────────────────────────────────────── */
async function startStream(){
  const sel=document.getElementById('sourceSelect').value;
  let src='webcam';
  if(sel==='rtsp'){
    src=document.getElementById('rtspInput').value.trim()||'webcam';
  } else if(sel==='file'){
    const f=document.getElementById('fileInput').files[0];
    if(!f){showToast('Please select a video file first.','warn');return}
    // Upload file first
    const fd=new FormData();fd.append('file',f);
    showToast('⏳ Uploading video...','info');
    const up=await fetch('/api/upload',{method:'POST',body:fd});
    const ud=await up.json();
    if(!ud.path){showToast('Upload failed.','danger');return}
    src=ud.path;
  }
  const res=await fetch('/api/start',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:src})});
  const d=await res.json();
  isRunning=true;
  document.getElementById('btnStart').disabled=true;
  document.getElementById('btnStop').disabled=false;
  document.getElementById('offlinePlaceholder').classList.add('hidden');
  document.getElementById('streamLabel').textContent='LIVE';
  showToast('Stream started ✓','ok');
  seenAlerts.clear();totalAlerts=0;
}

async function stopStream(){
  await fetch('/api/stop',{method:'POST'});
  isRunning=false;
  document.getElementById('btnStart').disabled=false;
  document.getElementById('btnStop').disabled=true;
  document.getElementById('offlinePlaceholder').classList.remove('hidden');
  document.getElementById('streamLabel').textContent='STREAM';
  showToast('Stream stopped','warn');
}

/* ── Poll Status ──────────────────────────────────────────── */
async function pollStatus(){
  try{
    const res=await fetch('/api/status');
    const d=await res.json();

    // Counts
    document.getElementById('cnt-person').textContent  = d.counts.person;
    document.getElementById('cnt-helmet').textContent  = d.counts.helmet;
    document.getElementById('cnt-nohelmet').textContent= d.counts.no_helmet;
    document.getElementById('cnt-vest').textContent    = d.counts.vest;
    document.getElementById('cnt-novest').textContent  = d.counts.no_vest;

    // Danger classes
    document.getElementById('sc-nohelmet').className='stat-card'+(d.counts.no_helmet>0?' danger':'');
    document.getElementById('sc-novest').className='stat-card'+(d.counts.no_vest>0?' danger':'');

    // FPS
    const fps=d.fps;
    document.getElementById('fpsVal').textContent=fps.toFixed(1);
    document.getElementById('fpsFill').style.width=Math.min(fps/30*100,100)+'%';
    document.getElementById('fpsStatus').textContent=d.running?'RUNNING':'IDLE';

    // Pills
    const py=document.getElementById('pillYolo');
    const pm=document.getElementById('pillModel');
    const ps=document.getElementById('pillStream');
    py.textContent=d.yolo_available?'Available':'Not Installed';
    py.className='pill '+(d.yolo_available?'ok':'warn');
    pm.textContent=d.model_loaded?'Loaded':'Demo Mode';
    pm.className='pill '+(d.model_loaded?'ok':'warn');
    ps.textContent=d.running?'ACTIVE':'IDLE';
    ps.className='pill '+(d.running?'ok':'off');

    // Mode badge
    document.getElementById('modeBadge').textContent=d.model_loaded?'LIVE':'DEMO';

    // Alerts
    if(d.alerts && d.alerts.length){
      const list=document.getElementById('alertsList');
      const existing=list.querySelector('.no-alerts');
      if(existing)existing.remove();
      d.alerts.forEach(a=>{
        const key=a.msg+a.time;
        if(!seenAlerts.has(key)){
          seenAlerts.add(key);totalAlerts++;
          const el=document.createElement('div');
          el.className='alert-item '+(a.level||'info');
          el.innerHTML=`<span class="alert-msg">${a.msg}</span><span class="alert-time">${a.time}</span>`;
          list.prepend(el);
          // Keep max 40 items
          while(list.children.length>40)list.removeChild(list.lastChild);
        }
      });
      document.getElementById('alertCount').textContent=Math.min(totalAlerts,99)+(totalAlerts>99?'+':'');
    }

    // Live Detections
    const dl=document.getElementById('detList');
    if(d.latest_detections && d.latest_detections.length){
      dl.innerHTML=d.latest_detections.map(det=>{
        const color=det.safe===true?'var(--ok)':det.safe===false?'var(--danger)':'var(--warn)';
        const cls=det.safe===true?'ok':det.safe===false?'bad':'unk';
        const conf=typeof det.conf==='number'?(det.conf*100).toFixed(0)+'%':'?';
        return `<div class="det-item">
          <span class="det-label">
            <span class="det-dot" style="background:${color}"></span>
            ${det.label.replace(/_/g,' ').toUpperCase()}
          </span>
          <span class="det-conf ${cls}">${conf}</span>
        </div>`;
      }).join('');
    } else if(!d.running){
      dl.innerHTML='<div class="no-alerts">No detections yet</div>';
    }

  }catch(e){console.warn('Poll error',e)}
}
setInterval(pollStatus,800);
pollStatus();

/* ── Toast ────────────────────────────────────────────────── */
let toastTimer;
function showToast(msg,type='info'){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.borderColor=type==='ok'?'var(--ok)':type==='danger'?'var(--danger)':
                       type==='warn'?'var(--warn)':'var(--y)';
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>t.classList.remove('show'),3200);
}

// Init source UI
onSourceChange();
</script>
</body>
</html>"""

# ─── File Upload Route ────────────────────────────────────────────────────────
import tempfile

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    # Save to temp file
    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    return jsonify({"path": tmp.name, "filename": f.filename})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  AI PPE Detection System — Starting Up")
    logger.info("=" * 60)
    load_model()
    logger.info(f"  YOLO Available : {YOLO_AVAILABLE}")
    logger.info(f"  Model Loaded   : {detection_state['model_loaded']}")
    logger.info("  URL            : http://127.0.0.1:5000")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
