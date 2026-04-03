# AI PPE Detection System — Setup Guide
## Final Year B.Tech Project

---

## Quick Start (3 steps)

### 1. Install Dependencies
```bash
pip install flask opencv-python ultralytics numpy pillow
```

### 2. Run the App
```bash
python app.py
```

### 3. Open Browser
```
http://127.0.0.1:5000
```

---

## Usage

| Source | How to Use |
|--------|-----------|
| **Webcam** | Select "Webcam" → Click Start |
| **Video File** | Select "Video File Upload" → Choose .mp4/.avi → Click Start |
| **RTSP Stream** | Select "RTSP Drone Stream" → Enter `rtsp://IP:PORT/stream` → Click Start |

---

## Custom YOLOv8 PPE Model Training

Use `train_ppe_model.py` to train your own model on PPE data.

### Step 1 — Get Dataset
Download from Roboflow (free):
- https://universe.roboflow.com/roboflow-100/ppe-raw-images
- https://universe.roboflow.com/joseph-nelson/hard-hat-workers

Or use any dataset with classes: `helmet`, `no_helmet`, `vest`, `no_vest`, `person`

### Step 2 — Folder Structure
```
dataset/
  images/
    train/   ← training images
    val/     ← validation images
  labels/
    train/   ← YOLO .txt label files
    val/
  data.yaml
```

### Step 3 — data.yaml
```yaml
path: ./dataset
train: images/train
val: images/val
nc: 5
names: ['helmet', 'no_helmet', 'vest', 'no_vest', 'person']
```

### Step 4 — Train
```bash
python train_ppe_model.py
```

### Step 5 — Use Custom Model
Place the output `ppe_model.pt` in the same folder as `app.py`.
The app auto-detects it on startup.

---

## Architecture

```
app.py
 ├── Flask Web Server (port 5000)
 ├── MJPEG Stream  →  /video_feed
 ├── REST API      →  /api/start | /api/stop | /api/status | /api/upload
 ├── YOLOv8 Inference Thread (background)
 ├── OpenCV Frame Capture (webcam / file / RTSP)
 └── Embedded HTML/CSS/JS (render_template_string)
```

---

## Demo Mode
If `ultralytics` is not installed, the system runs in **DEMO mode** — it simulates detections cycling through: safe → no helmet → no vest, so the UI is fully demonstrable without a GPU.

---

## Contact
- **Email:** avishkarsarang777@gmail.com
- **Phone:** +91 75889 43907
- **LinkedIn:** https://www.linkedin.com/in/avishkar-sarang-03b107333
