"""
╔══════════════════════════════════════════════════════════════╗
║   AI PPE Detection System — v2.0 Pinterest Edition          ║
║   Single-file Flask Application · Final Year B.Tech         ║
╚══════════════════════════════════════════════════════════════╝

pip install flask opencv-python ultralytics numpy pillow
python app.py  →  http://127.0.0.1:5000
"""

import cv2, numpy as np, threading, time, base64, json, os, logging, tempfile
from io import BytesIO
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not installed — running in DEMO mode.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ── Global state ───────────────────────────────────────────────────────────────
state = {
    "frame":        None,
    "alerts":       deque(maxlen=80),
    "counts":       {k: 0 for k in [
        "person","helmet","no_helmet","vest","no_vest",
        "harness","no_harness","safety_shoes","no_safety_shoes",
        "gloves","no_gloves","glasses","no_glasses"
    ]},
    "running":      False,
    "source":       None,
    "fps":          0,
    "model_loaded": False,
    "frame_count":  0,
    "lock":         threading.Lock(),
    "detections":   [],
    "mode":         "idle",
    "image_result": None,
}
model = None

# ── PPE class definitions ──────────────────────────────────────────────────────
PPE_CLASSES = {
    "helmet":              ("Helmet ✓",        (34,197,94),   True,  None),
    "no_helmet":           ("No Helmet!",      (239,68,68),   False, "No Helmet detected — head injury risk!"),
    "hard_hat":            ("Helmet ✓",        (34,197,94),   True,  None),
    "no_hard_hat":         ("No Helmet!",      (239,68,68),   False, "No Helmet detected — head injury risk!"),
    "vest":                ("Vest ✓",          (59,130,246),  True,  None),
    "no_vest":             ("No Vest!",        (245,158,11),  False, "No Safety Vest — visibility hazard!"),
    "safety_vest":         ("Vest ✓",          (59,130,246),  True,  None),
    "no_safety_vest":      ("No Vest!",        (245,158,11),  False, "No Safety Vest — visibility hazard!"),
    "harness":             ("Harness ✓",       (168,85,247),  True,  None),
    "no_harness":          ("No Harness!",     (220,38,38),   False, "No Harness — fall protection missing!"),
    "safety_shoes":        ("Shoes ✓",         (16,185,129),  True,  None),
    "no_safety_shoes":     ("No Shoes!",       (234,88,12),   False, "No Safety Shoes — foot hazard!"),
    "gloves":              ("Gloves ✓",        (99,102,241),  True,  None),
    "no_gloves":           ("No Gloves!",      (239,68,68),   False, "No Gloves — hand protection missing!"),
    "glasses":             ("Glasses ✓",       (20,184,166),  True,  None),
    "safety_glasses":      ("Glasses ✓",       (20,184,166),  True,  None),
    "no_glasses":          ("No Glasses!",     (245,158,11),  False, "No Safety Glasses — eye hazard!"),
    "no_safety_glasses":   ("No Glasses!",     (245,158,11),  False, "No Safety Glasses — eye hazard!"),
    "person":              ("Person",          (100,116,139), True,  None),
}

COUNT_MAP = {
    "helmet":"helmet","hard_hat":"helmet","no_helmet":"no_helmet","no_hard_hat":"no_helmet",
    "vest":"vest","safety_vest":"vest","no_vest":"no_vest","no_safety_vest":"no_vest",
    "harness":"harness","no_harness":"no_harness",
    "safety_shoes":"safety_shoes","no_safety_shoes":"no_safety_shoes",
    "gloves":"gloves","no_gloves":"no_gloves",
    "glasses":"glasses","safety_glasses":"glasses",
    "no_glasses":"no_glasses","no_safety_glasses":"no_glasses",
    "person":"person",
}

# ── Model loader ───────────────────────────────────────────────────────────────
def load_model():
    global model
    if not YOLO_AVAILABLE:
        return
    for candidate in ["ppe_model.pt","best.pt","yolov8n.pt"]:
        if os.path.exists(candidate):
            try:
                model = YOLO(candidate)
                state["model_loaded"] = True
                logger.info(f"Loaded: {candidate}")
                return
            except Exception as e:
                logger.error(f"Cannot load {candidate}: {e}")
    try:
        model = YOLO("yolov8n.pt")
        state["model_loaded"] = True
    except Exception as e:
        logger.error(f"Model load failed: {e}")

# ── Draw box ───────────────────────────────────────────────────────────────────
def draw_box(frame, x1,y1,x2,y2, label, conf, color):
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    txt = f"{label} {conf:.0%}"
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+8,y1),color,-1)
    cv2.putText(frame,txt,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

# ── YOLO inference ─────────────────────────────────────────────────────────────
def run_yolo(frame):
    alerts, dets = [], []
    counts = {k:0 for k in state["counts"]}
    if model is None:
        return frame, alerts, counts, dets
    results = model(frame, conf=0.35, iou=0.45, verbose=False)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            raw  = model.names[cls].lower().replace(" ","_")
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            info = PPE_CLASSES.get(raw)
            if info is None:
                if cls == 0:
                    info = PPE_CLASSES["person"]; raw = "person"
                else:
                    continue
            disp, color, safe, alert_msg = info
            draw_box(frame,x1,y1,x2,y2,disp,conf,color)
            ck = COUNT_MAP.get(raw,"person")
            if ck in counts: counts[ck] += 1
            if alert_msg:
                alerts.append({"msg":"⚠ "+alert_msg,"level":"danger","time":time.strftime("%H:%M:%S")})
            dets.append({"label":raw,"display":disp,"conf":conf,"safe":safe,
                         "color":"#{:02x}{:02x}{:02x}".format(*color[::-1])})
    return frame, alerts, counts, dets

# ── Demo simulation ────────────────────────────────────────────────────────────
_demo_tick = 0
def run_demo(frame):
    global _demo_tick
    h,w = frame.shape[:2]
    _demo_tick = (state["frame_count"] // 90) % 4
    alerts, dets = [], []
    counts = {k:0 for k in state["counts"]}
    counts["person"] = 1
    px1,py1,px2,py2 = w//5, h//8, 4*w//5, 7*h//8
    cv2.rectangle(frame,(px1,py1),(px2,py2),(148,163,184),2)
    scenarios = [
        [("helmet",(34,197,94)),("vest",(59,130,246)),("harness",(168,85,247)),
         ("safety_shoes",(16,185,129)),("gloves",(99,102,241)),("glasses",(20,184,166))],
        [("no_helmet",(239,68,68)),("vest",(59,130,246)),("harness",(168,85,247)),
         ("safety_shoes",(16,185,129)),("gloves",(99,102,241)),("no_glasses",(245,158,11))],
        [("helmet",(34,197,94)),("no_vest",(245,158,11)),("no_harness",(220,38,38)),
         ("safety_shoes",(16,185,129)),("no_gloves",(239,68,68)),("glasses",(20,184,166))],
        [("no_helmet",(239,68,68)),("no_vest",(245,158,11)),("no_harness",(220,38,38)),
         ("no_safety_shoes",(234,88,12)),("no_gloves",(239,68,68)),("no_glasses",(245,158,11))],
    ]
    sc = scenarios[_demo_tick]
    positions = [(px1,py1,px2,py1+45),(px1,py1+50,px2,py1+95),(px1,py1+100,px2,py1+145),
                 (px1,py1+150,px2,py1+195),(px1,py1+200,px2,py1+245),(px1,py1+250,px2,py2)]
    for i,(cls,col) in enumerate(sc):
        info = PPE_CLASSES.get(cls,PPE_CLASSES["person"])
        x1,y1,x2,y2 = positions[i]
        conf = 0.82 + 0.12*((i*5+_demo_tick)%3)/3
        draw_box(frame,x1,y1,x2,y2,info[0],conf,col)
        ck = COUNT_MAP.get(cls,"person")
        if ck in counts: counts[ck] += 1
        if info[3]:
            alerts.append({"msg":"⚠ "+info[3],"level":"danger","time":time.strftime("%H:%M:%S")})
        dets.append({"label":cls,"display":info[0],"conf":conf,"safe":info[2],
                     "color":"#{:02x}{:02x}{:02x}".format(*col[::-1])})
    cv2.putText(frame,"[ DEMO MODE — install ultralytics for live detection ]",
                (8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(148,163,184),1)
    return frame, alerts, counts, dets

# ── HUD overlay ────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, counts):
    h,w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,38),(252,250,247),-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.putText(frame,"AI PPE MONITOR",(8,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(99,102,241),2,cv2.LINE_AA)
    cv2.putText(frame,f"FPS {fps:.1f}",(w-80,25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(107,114,128),1,cv2.LINE_AA)
    ov2 = frame.copy()
    cv2.rectangle(ov2,(0,h-26),(w,h),(252,250,247),-1)
    cv2.addWeighted(ov2,0.85,frame,0.15,0,frame)
    parts=(f"P:{counts['person']}  Helm:{counts['helmet']}  Vest:{counts['vest']}  "
           f"Hrn:{counts['harness']}  Sh:{counts['safety_shoes']}  Gl:{counts['gloves']}  Gs:{counts['glasses']}")
    cv2.putText(frame,parts,(6,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(107,114,128),1,cv2.LINE_AA)
    return frame

# ── Stream thread ──────────────────────────────────────────────────────────────
def stream_worker(source):
    cap = cv2.VideoCapture(0 if source=="webcam" else source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        with state["lock"]: state["running"]=False
        return
    fps_t=time.time(); fc=0; cur_fps=0
    while True:
        with state["lock"]:
            if not state["running"]: break
        ret,frame = cap.read()
        if not ret:
            if source not in ("webcam",) and not str(source).startswith("rtsp"):
                cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
            break
        frame = cv2.resize(frame,(854,480))
        try:
            if YOLO_AVAILABLE and state["model_loaded"]:
                frame,new_a,counts,dets = run_yolo(frame)
            else:
                frame,new_a,counts,dets = run_demo(frame)
        except Exception as e:
            logger.error(e); new_a,counts,dets=[],{k:0 for k in state["counts"]},[]
        fc+=1
        el=time.time()-fps_t
        if el>=1.0: cur_fps=fc/el; fps_t=time.time(); fc=0
        frame = draw_hud(frame,cur_fps,counts)
        _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,82])
        with state["lock"]:
            state["frame"]=buf.tobytes(); state["fps"]=cur_fps
            state["counts"]=counts; state["frame_count"]+=1
            state["detections"]=dets
            for a in new_a: state["alerts"].appendleft(a)
    cap.release()
    with state["lock"]: state["running"]=False; state["frame"]=None

# ── Static image detection ─────────────────────────────────────────────────────
def detect_image(img):
    try:
        if YOLO_AVAILABLE and state["model_loaded"]:
            frame,alerts,counts,dets = run_yolo(img.copy())
        else:
            frame,alerts,counts,dets = run_demo(img.copy())
        _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,90])
        return base64.b64encode(buf).decode(), alerts, counts, dets
    except Exception as e:
        logger.error(e); return None,[],{},[]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/start", methods=["POST"])
def api_start():
    d=request.get_json(silent=True) or {}
    src=d.get("source","webcam")
    with state["lock"]:
        state["running"]=False; time.sleep(0.3)
        state.update(running=True,source=src,alerts=deque(maxlen=80),
                     frame_count=0,mode="stream",image_result=None)
    threading.Thread(target=stream_worker,args=(src,),daemon=True).start()
    return jsonify(status="started",source=src,yolo=YOLO_AVAILABLE,model=state["model_loaded"])

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state["lock"]: state["running"]=False; state["mode"]="idle"
    return jsonify(status="stopped")

@app.route("/api/status")
def api_status():
    with state["lock"]:
        return jsonify(running=state["running"],fps=round(state["fps"],1),
                       counts=state["counts"],frame_count=state["frame_count"],
                       alerts=list(state["alerts"])[:15],model_loaded=state["model_loaded"],
                       yolo=YOLO_AVAILABLE,detections=state["detections"],mode=state["mode"])

@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    if "image" not in request.files: return jsonify(error="no image"),400
    f=request.files["image"]
    data=np.frombuffer(f.read(),np.uint8)
    img=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if img is None: return jsonify(error="invalid image"),400
    if max(img.shape[:2])>1600: img=cv2.resize(img,(854,480))
    with state["lock"]: state["running"]=False; state["mode"]="image"
    b64,alerts,counts,dets=detect_image(img)
    with state["lock"]:
        state["image_result"]=b64; state["counts"]=counts
        state["detections"]=dets; state["alerts"]=deque(alerts,maxlen=80)
    return jsonify(image=b64,alerts=alerts,counts=counts,detections=dets)

@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    if "video" not in request.files: return jsonify(error="no video"),400
    f=request.files["video"]
    suf=os.path.splitext(f.filename)[1] or ".mp4"
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=suf)
    f.save(tmp.name)
    with state["lock"]:
        state["running"]=False; time.sleep(0.3)
        state.update(running=True,source=tmp.name,alerts=deque(maxlen=80),
                     frame_count=0,mode="stream",image_result=None)
    threading.Thread(target=stream_worker,args=(tmp.name,),daemon=True).start()
    return jsonify(status="started",filename=f.filename)

def gen_frames():
    blank=np.ones((480,854,3),dtype=np.uint8)*248
    cv2.putText(blank,"Select a source and press Start",(170,240),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(180,170,160),2)
    _,bb=cv2.imencode(".jpg",blank); blank_b=bb.tobytes()
    while True:
        with state["lock"]: fr=state["frame"]
        d=fr if fr else blank_b
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+d+b"\r\n"
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI PPE Detection · Safety Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --cream:#FDFAF7;--warm:#F5EFE6;--blush:#F0E6DC;
  --rose:#E07A5F;--rose2:#D4614A;--sage:#81B29A;
  --slate:#6B7280;--ink:#2D2D2D;--ink2:#4B4B4B;
  --mist:#E5DDD5;--mist2:#C8BFB8;--card:#FFFFFF;--border:#EDE8E3;
  --shadow:rgba(107,85,70,.10);
  --red:#DC2626;--amber:#D97706;--green:#059669;
  --violet:#7C3AED;--sky:#0284C7;--teal:#0D9488;
  --r:8px;--r2:14px;
  --ts:0 1px 3px rgba(107,85,70,.08),0 4px 16px rgba(107,85,70,.06);
  --ts2:0 2px 8px rgba(107,85,70,.12),0 8px 32px rgba(107,85,70,.08);
}
html{font-size:16px}
body{background:var(--cream);color:var(--ink);font-family:'DM Sans',sans-serif;min-height:100vh;
  background-image:radial-gradient(circle at 15% 15%,rgba(224,122,95,.06) 0%,transparent 50%),
                   radial-gradient(circle at 85% 80%,rgba(129,178,154,.07) 0%,transparent 50%)}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.022'/%3E%3C/svg%3E");opacity:.7}
.app{position:relative;z-index:1;min-height:100vh;display:grid;grid-template-rows:auto 1fr auto}

/* Header */
header{background:rgba(253,250,247,.92);backdrop-filter:blur(14px);
  border-bottom:1px solid var(--border);padding:0 2rem;height:66px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;box-shadow:0 1px 0 var(--border),var(--ts)}
.hd-left{display:flex;align-items:center;gap:12px}
.hd-logo{width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,var(--rose),var(--rose2));
  display:grid;place-items:center;font-size:1.2rem;
  box-shadow:0 2px 10px rgba(224,122,95,.35);flex-shrink:0}
.hd-title{font-family:'DM Serif Display',serif;font-size:1.2rem;
  color:var(--ink);letter-spacing:-.01em;line-height:1.1}
.hd-title small{font-family:'DM Sans',sans-serif;font-size:.62rem;
  font-weight:500;color:var(--slate);display:block;letter-spacing:.05em;
  text-transform:uppercase;margin-top:2px}
.hd-right{display:flex;align-items:center;gap:.6rem}
.chip{padding:3px 11px;border-radius:20px;font-size:.68rem;font-weight:600;
  letter-spacing:.05em;text-transform:uppercase}
.chip-rose{background:var(--blush);color:var(--rose2)}
.chip-live{background:rgba(5,150,105,.12);color:var(--green);animation:pulse 2s ease-in-out infinite}
.chip-violet{background:rgba(124,58,237,.1);color:var(--violet)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}

/* Main */
main{display:grid;grid-template-columns:1fr 336px;gap:1.2rem;
  padding:1.2rem 1.5rem;align-items:start}
@media(max-width:960px){main{grid-template-columns:1fr}}

/* Cards */
.card{background:var(--card);border-radius:var(--r2);border:1px solid var(--border);
  box-shadow:var(--ts);overflow:hidden}
.card-head{padding:.85rem 1.1rem;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
.card-title{font-family:'DM Serif Display',serif;font-size:.92rem;color:var(--ink);
  display:flex;align-items:center;gap:.4rem}

/* Tab system */
.tab-bar{display:flex;gap:.35rem;padding:.7rem 1rem;border-bottom:1px solid var(--border);
  background:var(--warm)}
.tab{padding:.38rem .85rem;border-radius:var(--r);border:1px solid transparent;
  font-size:.78rem;font-weight:500;cursor:pointer;transition:all .15s;
  background:transparent;color:var(--slate);font-family:'DM Sans',sans-serif}
.tab.active{background:var(--rose);color:#fff;box-shadow:0 2px 8px rgba(224,122,95,.3)}
.tab:not(.active):hover{background:var(--blush);color:var(--ink)}
.tab-panel{padding:.9rem 1.1rem;display:none}
.tab-panel.active{display:block}

/* Controls */
.input-row{display:flex;gap:.55rem;align-items:center;flex-wrap:wrap}
.sel{background:var(--warm);border:1px solid var(--mist);color:var(--ink);
  padding:.46rem .8rem;border-radius:var(--r);font-family:'DM Sans',sans-serif;
  font-size:.83rem;cursor:pointer;flex:1;min-width:150px;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%236B7280' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right .6rem center;
  padding-right:1.8rem;appearance:none}
.sel:focus{outline:2px solid var(--rose);outline-offset:1px}
.txt-in{flex:2;background:var(--warm);border:1px solid var(--mist);
  color:var(--ink);padding:.46rem .8rem;border-radius:var(--r);
  font-family:'DM Sans',sans-serif;font-size:.8rem;display:none}
.txt-in:focus{outline:2px solid var(--rose);outline-offset:1px}
.txt-in::placeholder{color:var(--mist2)}

/* Drop zones */
.drop-zone{border:2px dashed var(--mist);border-radius:var(--r2);
  padding:1.5rem 1rem;text-align:center;cursor:pointer;
  transition:all .2s;background:var(--warm);position:relative}
.drop-zone:hover,.drop-zone.drag{border-color:var(--rose);background:var(--blush)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:2.2rem;display:block;margin-bottom:.4rem;opacity:.45}
.drop-zone p{font-size:.8rem;color:var(--slate);font-weight:500}
.drop-zone small{font-size:.7rem;color:var(--mist2)}
.file-chosen{font-size:.73rem;color:var(--rose);font-weight:600;margin-top:.35rem}

/* Buttons */
.btn{padding:.46rem 1rem;border:none;border-radius:var(--r);cursor:pointer;
  font-family:'DM Sans',sans-serif;font-size:.83rem;font-weight:600;
  transition:all .16s;display:inline-flex;align-items:center;gap:.35rem;
  white-space:nowrap}
.btn-rose{background:var(--rose);color:#fff;box-shadow:0 2px 8px rgba(224,122,95,.3)}
.btn-rose:hover{background:var(--rose2);transform:translateY(-1px)}
.btn-ghost{background:transparent;color:var(--slate);border:1px solid var(--mist)}
.btn-ghost:hover{background:var(--blush);color:var(--ink)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none!important}
.btn-sm{padding:.32rem .75rem;font-size:.76rem}

/* Progress */
.prog{height:3px;background:var(--mist);border-radius:3px;margin-top:.5rem;
  overflow:hidden;display:none}
.prog-fill{height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--rose),var(--sage));
  animation:shimmer 1.4s infinite;background-size:200% 100%}
@keyframes shimmer{to{background-position:-200% 0}}

/* Video wrap */
.video-wrap{position:relative;border-radius:var(--r);overflow:hidden;
  background:var(--blush);aspect-ratio:16/9;margin:.1rem 1.1rem 1.1rem}
.video-wrap img{width:100%;height:100%;object-fit:cover;display:block}
#imgResult{display:none}
.stream-badge{position:absolute;top:10px;right:10px;z-index:5;
  background:rgba(253,250,247,.9);backdrop-filter:blur(8px);
  border:1px solid var(--border);border-radius:20px;
  padding:3px 11px;font-size:.7rem;font-weight:600;letter-spacing:.06em;
  color:var(--slate);display:flex;align-items:center;gap:5px}
.live-dot{width:6px;height:6px;border-radius:50%;transition:background .3s}
.offline-plate{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:.6rem;background:var(--blush);
  transition:opacity .3s;pointer-events:none}
.offline-plate.hidden{opacity:0}
.offline-plate .big{font-size:3rem;opacity:.3}
.offline-plate p{font-size:.83rem;color:var(--slate);font-weight:500}
.offline-plate small{font-size:.7rem;color:var(--mist2)}

/* Stats */
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;padding:.9rem 1.1rem}
@media(max-width:700px){.stats-grid{grid-template-columns:repeat(3,1fr)}}
.sc{background:var(--warm);border-radius:var(--r);border:1px solid var(--border);
  padding:.65rem .5rem;text-align:center;transition:border-color .25s,background .25s}
.sc.flagged{border-color:rgba(220,38,38,.3);background:rgba(220,38,38,.04)}
.sc.safe-state{border-color:rgba(5,150,105,.25);background:rgba(5,150,105,.03)}
.sv{font-family:'DM Serif Display',serif;font-size:1.7rem;line-height:1;color:var(--ink)}
.sv.ok{color:var(--green)} .sv.bad{color:var(--red)}
.sl{font-size:.6rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--slate);margin-top:2px}

/* FPS strip */
.fps-strip{padding:.55rem 1.1rem;border-top:1px solid var(--border);
  display:flex;align-items:center;gap:.8rem}
.fps-num{font-family:'DM Serif Display',serif;font-size:1.4rem;color:var(--rose);
  line-height:1;min-width:38px}
.fps-lbl{font-size:.6rem;color:var(--slate);letter-spacing:.1em;text-transform:uppercase}
.fps-track{flex:1;height:4px;background:var(--mist);border-radius:3px;overflow:hidden}
.fps-fill{height:100%;background:linear-gradient(90deg,var(--rose),var(--sage));
  border-radius:3px;transition:width .6s ease;width:0%}
#fpsStatus{font-size:.68rem;color:var(--slate);letter-spacing:.05em;text-transform:uppercase}

/* Side panel */
.side-panel{display:flex;flex-direction:column;gap:1rem}

/* System rows */
.sys-rows{padding:.7rem 1rem;display:flex;flex-direction:column;gap:.5rem}
.sys-row{display:flex;justify-content:space-between;align-items:center}
.sys-key{font-size:.7rem;color:var(--slate);letter-spacing:.06em;text-transform:uppercase}
.pill{padding:2px 9px;border-radius:20px;font-size:.65rem;font-weight:700;letter-spacing:.05em}
.pill-ok{background:rgba(5,150,105,.1);color:var(--green);border:1px solid rgba(5,150,105,.2)}
.pill-warn{background:rgba(217,119,6,.1);color:var(--amber);border:1px solid rgba(217,119,6,.2)}
.pill-off{background:var(--warm);color:var(--slate);border:1px solid var(--mist)}
.pill-live{background:rgba(220,38,38,.1);color:var(--red);border:1px solid rgba(220,38,38,.2);animation:pulse 2s ease-in-out infinite}
.conf-wrap{padding:0 1rem .7rem;display:flex;flex-direction:column;gap:.45rem}
.conf-lbl{font-size:.68rem;color:var(--slate);letter-spacing:.06em;text-transform:uppercase;
  display:flex;justify-content:space-between}
.conf-track{height:4px;background:var(--mist);border-radius:3px;margin-top:3px;overflow:hidden}
.conf-fill{height:100%;background:linear-gradient(90deg,var(--rose),var(--sage));border-radius:3px}

/* Detections */
.det-wrap{padding:.65rem 1rem;display:flex;flex-direction:column;gap:.35rem;min-height:55px}
.det-item{display:flex;align-items:center;justify-content:space-between;
  padding:.32rem .5rem;border-radius:var(--r);background:var(--warm);animation:popIn .2s ease}
@keyframes popIn{from{transform:scale(.96);opacity:0}}
.det-left{display:flex;align-items:center;gap:.45rem;font-size:.77rem;font-weight:500}
.det-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.det-cf{font-size:.75rem;font-weight:700;font-family:'DM Serif Display',serif}
.det-cf.ok{color:var(--green)} .det-cf.bad{color:var(--red)} .det-cf.unk{color:var(--amber)}
.no-dets{font-size:.77rem;color:var(--mist2);text-align:center;padding:.4rem 0}

/* Alerts */
.alerts-wrap{padding:.45rem;display:flex;flex-direction:column;gap:.3rem;
  max-height:260px;overflow-y:auto}
.alerts-wrap::-webkit-scrollbar{width:3px}
.alerts-wrap::-webkit-scrollbar-thumb{background:var(--mist);border-radius:2px}
.al{padding:.45rem .65rem;border-radius:var(--r);font-size:.76rem;
  animation:slideR .22s ease;border-left:3px solid}
@keyframes slideR{from{transform:translateX(14px);opacity:0}}
.al.danger{background:rgba(220,38,38,.06);border-color:var(--red)}
.al.warning{background:rgba(217,119,6,.06);border-color:var(--amber)}
.al.info{background:rgba(2,132,199,.05);border-color:var(--sky)}
.a-msg{font-weight:600;color:var(--ink)}
.a-time{font-size:.63rem;color:var(--slate);margin-top:1px}
.no-alerts{font-size:.77rem;color:var(--mist2);text-align:center;padding:.65rem 0}

/* PPE legend */
.ppe-grid{padding:.65rem 1rem;display:grid;grid-template-columns:1fr 1fr;gap:.35rem}
.ppe-badge{display:flex;align-items:center;gap:.38rem;font-size:.7rem;
  padding:.28rem .45rem;border-radius:var(--r);background:var(--warm);font-weight:500}
.ppe-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}

/* Contact */
.contact-wrap{padding:.65rem 1rem;display:flex;flex-direction:column;gap:.35rem}
.c-link{display:flex;align-items:center;gap:.55rem;font-size:.77rem;
  color:var(--slate);text-decoration:none;padding:.28rem 0;transition:color .15s}
.c-link:hover{color:var(--rose)}
.c-icon{width:26px;height:26px;border-radius:6px;display:grid;place-items:center;
  font-size:.82rem;flex-shrink:0}
.ci-e{background:rgba(224,122,95,.12);color:var(--rose)}
.ci-p{background:rgba(5,150,105,.1);color:var(--green)}
.ci-l{background:rgba(2,132,199,.1);color:var(--sky)}

/* Footer */
footer{border-top:1px solid var(--border);padding:.7rem 1.5rem;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.4rem;
  background:rgba(253,250,247,.8);font-size:.68rem}
.ft-copy{color:var(--mist2)}
.ft-tag{color:var(--rose);font-weight:600;letter-spacing:.05em}

/* Toast */
#toast{position:fixed;bottom:1.4rem;right:1.4rem;z-index:9999;
  background:var(--card);border:1px solid var(--border);border-radius:var(--r2);
  padding:.6rem 1rem;font-size:.8rem;font-weight:500;color:var(--ink);
  box-shadow:var(--ts2);display:none;max-width:300px;animation:toastIn .22s ease}
@keyframes toastIn{from{transform:translateY(10px);opacity:0}}
#toast.show{display:block}
</style>
</head>
<body>
<div class="app">

<!-- HEADER -->
<header>
  <div class="hd-left">
    <div class="hd-logo">🦺</div>
    <div class="hd-title">
      AI PPE Detection
      <small>Construction Safety · Drone Monitoring</small>
    </div>
  </div>
  <div class="hd-right">
    <span class="chip chip-violet" id="chipModel">YOLOv8</span>
    <span class="chip chip-rose"   id="chipMode">IDLE</span>
  </div>
</header>

<!-- MAIN -->
<main>
<div style="display:flex;flex-direction:column;gap:1rem">

  <!-- Primary card -->
  <div class="card">
    <!-- Tabs -->
    <div class="tab-bar">
      <button class="tab active" onclick="switchTab('stream',this)">📡 Live Stream</button>
      <button class="tab"        onclick="switchTab('image',this)">🖼 Image Upload</button>
      <button class="tab"        onclick="switchTab('video',this)">🎬 Video Upload</button>
    </div>

    <!-- Stream tab -->
    <div class="tab-panel active" id="panel-stream">
      <div class="input-row">
        <select class="sel" id="srcSel" onchange="onSrcChange()">
          <option value="webcam">📷  Webcam (Camera 0)</option>
          <option value="rtsp">📡  RTSP / Drone Stream</option>
        </select>
        <input class="txt-in" id="rtspUrl" placeholder="rtsp://192.168.x.x:554/stream"/>
        <button class="btn btn-rose" id="btnStart" onclick="startStream()">▶ Start</button>
        <button class="btn btn-ghost" id="btnStop" onclick="stopStream()" disabled>■ Stop</button>
      </div>
      <div class="prog" id="pStream"><div class="prog-fill"></div></div>
    </div>

    <!-- Image tab -->
    <div class="tab-panel" id="panel-image">
      <div class="drop-zone" id="imgDrop"
           ondragover="dov(event,'imgDrop')" ondragleave="dlv('imgDrop')" ondrop="dodrp(event,'img')">
        <input type="file" accept="image/*" id="imgFile" onchange="onImgFile()"/>
        <span class="drop-icon">🖼</span>
        <p>Drop image here or click to browse</p>
        <small>JPG · PNG · WEBP — up to 20 MB</small>
        <div class="file-chosen" id="imgChosen"></div>
      </div>
      <div style="display:flex;gap:.5rem;margin-top:.65rem;flex-wrap:wrap">
        <button class="btn btn-rose" id="btnAnalyze" onclick="analyzeImage()" disabled>🔍 Detect PPE</button>
        <button class="btn btn-ghost btn-sm" onclick="clearImg()">✕ Clear</button>
      </div>
      <div class="prog" id="pImg"><div class="prog-fill"></div></div>
    </div>

    <!-- Video tab -->
    <div class="tab-panel" id="panel-video">
      <div class="drop-zone" id="vidDrop"
           ondragover="dov(event,'vidDrop')" ondragleave="dlv('vidDrop')" ondrop="dodrp(event,'vid')">
        <input type="file" accept="video/*" id="vidFile" onchange="onVidFile()"/>
        <span class="drop-icon">🎬</span>
        <p>Drop video here or click to browse</p>
        <small>MP4 · AVI · MOV — up to 200 MB</small>
        <div class="file-chosen" id="vidChosen"></div>
      </div>
      <div style="display:flex;gap:.5rem;margin-top:.65rem;flex-wrap:wrap">
        <button class="btn btn-rose" id="btnVidStart" onclick="startVideo()" disabled>▶ Process Video</button>
        <button class="btn btn-ghost" id="btnVidStop" onclick="stopStream()" disabled>■ Stop</button>
      </div>
      <div class="prog" id="pVid"><div class="prog-fill"></div></div>
    </div>

    <!-- Video/Image feed -->
    <div class="video-wrap">
      <img id="videoFeed" src="/video_feed" alt="Stream"/>
      <img id="imgResult" src="" alt="Detection result"/>
      <div class="stream-badge">
        <span class="live-dot" id="liveDot" style="background:var(--mist2)"></span>
        <span id="streamLabel">READY</span>
      </div>
      <div class="offline-plate" id="offlinePlate">
        <div class="big">🏗️</div>
        <p>No active source</p>
        <small>Choose a tab above and start detection</small>
      </div>
    </div>

    <!-- Stats (13 cards: person + 6 pairs) -->
    <div class="stats-grid">
      <div class="sc" id="sc-person">
        <div class="sv" id="cnt-person">0</div><div class="sl">Persons</div></div>
      <div class="sc" id="sc-helmet">
        <div class="sv ok" id="cnt-helmet">0</div><div class="sl">Helmet ✓</div></div>
      <div class="sc" id="sc-nohelmet">
        <div class="sv" id="cnt-nohelmet">0</div><div class="sl">No Helmet ⚠</div></div>
      <div class="sc" id="sc-vest">
        <div class="sv ok" id="cnt-vest">0</div><div class="sl">Vest ✓</div></div>
      <div class="sc" id="sc-novest">
        <div class="sv" id="cnt-novest">0</div><div class="sl">No Vest ⚠</div></div>
      <div class="sc" id="sc-harness">
        <div class="sv ok" id="cnt-harness">0</div><div class="sl">Harness ✓</div></div>
      <div class="sc" id="sc-noharness">
        <div class="sv" id="cnt-noharness">0</div><div class="sl">No Harness ⚠</div></div>
      <div class="sc" id="sc-shoes">
        <div class="sv ok" id="cnt-shoes">0</div><div class="sl">Shoes ✓</div></div>
      <div class="sc" id="sc-noshoes">
        <div class="sv" id="cnt-noshoes">0</div><div class="sl">No Shoes ⚠</div></div>
      <div class="sc" id="sc-gloves">
        <div class="sv ok" id="cnt-gloves">0</div><div class="sl">Gloves ✓</div></div>
      <div class="sc" id="sc-nogloves">
        <div class="sv" id="cnt-nogloves">0</div><div class="sl">No Gloves ⚠</div></div>
      <div class="sc" id="sc-glasses">
        <div class="sv ok" id="cnt-glasses">0</div><div class="sl">Glasses ✓</div></div>
      <div class="sc" id="sc-noglasses">
        <div class="sv" id="cnt-noglasses">0</div><div class="sl">No Glasses ⚠</div></div>
    </div>

    <!-- FPS -->
    <div class="fps-strip">
      <div><div class="fps-num" id="fpsNum">0</div><div class="fps-lbl">FPS</div></div>
      <div class="fps-track"><div class="fps-fill" id="fpsFill"></div></div>
      <span id="fpsStatus">IDLE</span>
    </div>
  </div><!-- /card -->

</div><!-- /left -->

<!-- SIDE -->
<div class="side-panel">

  <!-- System -->
  <div class="card">
    <div class="card-head">
      <span class="card-title">⚙️ System Status</span>
    </div>
    <div class="sys-rows">
      <div class="sys-row"><span class="sys-key">YOLO</span><span class="pill" id="pillYolo">—</span></div>
      <div class="sys-row"><span class="sys-key">Model</span><span class="pill" id="pillModel">—</span></div>
      <div class="sys-row"><span class="sys-key">Stream</span><span class="pill" id="pillStream">IDLE</span></div>
      <div class="sys-row"><span class="sys-key">Mode</span><span class="pill" id="pillMode">—</span></div>
    </div>
    <div class="conf-wrap">
      <div><div class="conf-lbl">Confidence Threshold<span>35%</span></div>
        <div class="conf-track"><div class="conf-fill" style="width:35%"></div></div></div>
      <div><div class="conf-lbl">NMS IoU Threshold<span>45%</span></div>
        <div class="conf-track"><div class="conf-fill" style="width:45%"></div></div></div>
    </div>
  </div>

  <!-- Live detections -->
  <div class="card">
    <div class="card-head">
      <span class="card-title">🎯 Live Detections</span>
      <span class="chip chip-rose" id="detCount">0</span>
    </div>
    <div class="det-wrap" id="detList"><div class="no-dets">No detections yet</div></div>
  </div>

  <!-- Alerts -->
  <div class="card">
    <div class="card-head">
      <span class="card-title">⚠️ Alert Log</span>
      <span class="chip chip-rose" id="alertBadge">0</span>
    </div>
    <div class="alerts-wrap" id="alertsList">
      <div class="no-alerts">All clear — no violations ✓</div>
    </div>
  </div>

  <!-- PPE legend -->
  <div class="card">
    <div class="card-head"><span class="card-title">📋 PPE Categories</span></div>
    <div class="ppe-grid">
      <div class="ppe-badge"><div class="ppe-dot" style="background:#059669"></div>Helmet</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#DC2626"></div>No Helmet</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#0284C7"></div>Safety Vest</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#D97706"></div>No Vest</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#7C3AED"></div>Harness</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#DC2626"></div>No Harness</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#10B981"></div>Safety Shoes</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#EA580C"></div>No Shoes</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#6366F1"></div>Gloves</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#DC2626"></div>No Gloves</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#0D9488"></div>Glasses</div>
      <div class="ppe-badge"><div class="ppe-dot" style="background:#D97706"></div>No Glasses</div>
    </div>
  </div>

  <!-- Contact -->
  <div class="card">
    <div class="card-head"><span class="card-title">📬 Developer</span></div>
    <div class="contact-wrap">
      <a class="c-link" href="mailto:avishkarsarang777@gmail.com">
        <div class="c-icon ci-e">✉</div>avishkarsarang777@gmail.com</a>
      <a class="c-link" href="tel:+917588943907">
        <div class="c-icon ci-p">📞</div>+91 75889 43907</a>
      <a class="c-link" href="https://www.linkedin.com/in/avishkar-sarang-03b107333" target="_blank">
        <div class="c-icon ci-l">in</div>Avishkar Sarang — LinkedIn</a>
    </div>
  </div>

</div><!-- /side-panel -->
</main>

<footer>
  <span class="ft-copy">AI PPE Detection System · Final Year B.Tech · Civil & Infrastructure Engineering</span>
  <span class="ft-tag">YOLOv8 · Flask · OpenCV</span>
</footer>
</div><!-- /app -->
<div id="toast"></div>

<script>
/* ── State ── */
let imgFile=null, vidFile=null, seenAlerts=new Set(), totalAlerts=0;

/* ── Tabs ── */
function switchTab(n,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-'+n).classList.add('active');
}

/* ── Source ── */
function onSrcChange(){
  const v=document.getElementById('srcSel').value;
  document.getElementById('rtspUrl').style.display=v==='rtsp'?'block':'none';
}

/* ── Drag & drop ── */
function dov(e,id){e.preventDefault();document.getElementById(id).classList.add('drag')}
function dlv(id){document.getElementById(id).classList.remove('drag')}
function dodrp(e,t){
  e.preventDefault();
  const id=t==='img'?'imgDrop':'vidDrop';
  document.getElementById(id).classList.remove('drag');
  const f=e.dataTransfer.files[0];if(!f)return;
  if(t==='img'){setImgFile(f);}else{setVidFile(f);}
}
function onImgFile(){const f=document.getElementById('imgFile').files[0];if(f)setImgFile(f);}
function onVidFile(){const f=document.getElementById('vidFile').files[0];if(f)setVidFile(f);}
function setImgFile(f){imgFile=f;document.getElementById('imgChosen').textContent='📎 '+f.name;document.getElementById('btnAnalyze').disabled=false;}
function setVidFile(f){vidFile=f;document.getElementById('vidChosen').textContent='📎 '+f.name;document.getElementById('btnVidStart').disabled=false;}
function clearImg(){
  imgFile=null;document.getElementById('imgChosen').textContent='';
  document.getElementById('btnAnalyze').disabled=true;
  document.getElementById('imgResult').style.display='none';
  document.getElementById('videoFeed').style.display='block';
  document.getElementById('offlinePlate').classList.remove('hidden');
  resetCounts();resetDets();
}

/* ── Stream start/stop ── */
async function startStream(){
  const sel=document.getElementById('srcSel').value;
  let src='webcam';
  if(sel==='rtsp'){
    src=document.getElementById('rtspUrl').value.trim();
    if(!src){toast('Enter an RTSP URL','warn');return;}
  }
  show('pStream');
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})});
  hide('pStream');
  setLive(true,'LIVE');
  document.getElementById('offlinePlate').classList.add('hidden');
  document.getElementById('videoFeed').style.display='block';
  document.getElementById('imgResult').style.display='none';
  document.getElementById('btnStart').disabled=true;
  document.getElementById('btnStop').disabled=false;
  seenAlerts.clear();totalAlerts=0;
  toast('Stream started ✓','ok');
}
async function stopStream(){
  await fetch('/api/stop',{method:'POST'});
  setLive(false,'READY');
  ['btnStart','btnVidStart'].forEach(id=>document.getElementById(id).disabled=false);
  ['btnStop','btnVidStop'].forEach(id=>document.getElementById(id).disabled=true);
  document.getElementById('offlinePlate').classList.remove('hidden');
  toast('Stopped','warn');
}
function setLive(on,lbl){
  document.getElementById('liveDot').style.background=on?'var(--red)':'var(--mist2)';
  document.getElementById('streamLabel').textContent=lbl;
  const cm=document.getElementById('chipMode');
  cm.textContent=on?'LIVE':'IDLE';cm.className='chip '+(on?'chip-live':'chip-rose');
}

/* ── Image analysis ── */
async function analyzeImage(){
  if(!imgFile)return;
  const btn=document.getElementById('btnAnalyze');
  btn.disabled=true;btn.textContent='⏳ Analyzing…';show('pImg');
  const fd=new FormData();fd.append('image',imgFile);
  try{
    const r=await fetch('/api/upload_image',{method:'POST',body:fd});
    const d=await r.json();hide('pImg');
    if(d.image){
      document.getElementById('imgResult').src='data:image/jpeg;base64,'+d.image;
      document.getElementById('imgResult').style.display='block';
      document.getElementById('videoFeed').style.display='none';
      document.getElementById('offlinePlate').classList.add('hidden');
      updateCounts(d.counts);updateDets(d.detections);updateAlerts(d.alerts);
      setLive(false,'IMAGE');toast('Detection complete ✓','ok');
    }
  }catch(e){toast('Upload failed','danger')}
  btn.disabled=false;btn.textContent='🔍 Detect PPE';
}

/* ── Video upload ── */
async function startVideo(){
  if(!vidFile)return;
  const btn=document.getElementById('btnVidStart');
  btn.disabled=true;btn.textContent='⏳ Uploading…';show('pVid');
  const fd=new FormData();fd.append('video',vidFile);
  try{
    const r=await fetch('/api/upload_video',{method:'POST',body:fd});
    const d=await r.json();hide('pVid');
    if(d.status==='started'){
      setLive(true,'PROCESSING');
      document.getElementById('offlinePlate').classList.add('hidden');
      document.getElementById('videoFeed').style.display='block';
      document.getElementById('imgResult').style.display='none';
      document.getElementById('btnVidStop').disabled=false;
      seenAlerts.clear();totalAlerts=0;
      toast('Video processing started ✓','ok');
    }
  }catch(e){toast('Upload failed','danger')}
  btn.textContent='▶ Process Video';
}

/* ── Count update ── */
const CK={
  person:'person',helmet:'helmet',no_helmet:'nohelmet',
  vest:'vest',no_vest:'novest',harness:'harness',no_harness:'noharness',
  safety_shoes:'shoes',no_safety_shoes:'noshoes',
  gloves:'gloves',no_gloves:'nogloves',
  glasses:'glasses',no_glasses:'noglasses'
};
const BAD=new Set(['no_helmet','no_vest','no_harness','no_safety_shoes','no_gloves','no_glasses']);
const GOOD=new Set(['helmet','vest','harness','safety_shoes','gloves','glasses']);

function updateCounts(c){
  if(!c)return;
  for(const[k,id] of Object.entries(CK)){
    const el=document.getElementById('cnt-'+id);
    const card=document.getElementById('sc-'+id);
    if(!el)continue;
    const v=c[k]||0;
    el.textContent=v;
    el.className='sv'+(BAD.has(k)&&v>0?' bad':GOOD.has(k)&&v>0?' ok':'');
    if(card)card.className='sc'+(BAD.has(k)&&v>0?' flagged':GOOD.has(k)&&v>0?' safe-state':'');
  }
}
function resetCounts(){updateCounts(Object.fromEntries(Object.keys(CK).map(k=>[k,0])));}

/* ── Detections ── */
function updateDets(dets){
  const dl=document.getElementById('detList'),dc=document.getElementById('detCount');
  if(!dets||!dets.length){
    if(!dl.querySelector('.no-dets'))dl.innerHTML='<div class="no-dets">No detections</div>';
    dc.textContent='0';return;
  }
  dl.innerHTML=dets.map(d=>{
    const cls=d.safe===true?'ok':d.safe===false?'bad':'unk';
    const conf=(d.conf*100).toFixed(0)+'%';
    return `<div class="det-item"><span class="det-left"><span class="det-dot" style="background:${d.color||'#999'}"></span>${(d.display||d.label).replace(/_/g,' ')}</span><span class="det-cf ${cls}">${conf}</span></div>`;
  }).join('');
  dc.textContent=dets.length;
}
function resetDets(){document.getElementById('detList').innerHTML='<div class="no-dets">No detections yet</div>';document.getElementById('detCount').textContent='0';}

/* ── Alerts ── */
function updateAlerts(alerts){
  if(!alerts||!alerts.length)return;
  const list=document.getElementById('alertsList');
  const noa=list.querySelector('.no-alerts');if(noa)noa.remove();
  alerts.forEach(a=>{
    const key=a.msg+a.time;if(seenAlerts.has(key))return;
    seenAlerts.add(key);totalAlerts++;
    const el=document.createElement('div');el.className='al '+(a.level||'info');
    el.innerHTML=`<div class="a-msg">${a.msg}</div><div class="a-time">${a.time}</div>`;
    list.prepend(el);
    while(list.children.length>40)list.removeChild(list.lastChild);
  });
  document.getElementById('alertBadge').textContent=Math.min(totalAlerts,99)+(totalAlerts>99?'+':'');
}

/* ── Poll ── */
async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    const set=(id,txt,cls)=>{const e=document.getElementById(id);if(e){e.textContent=txt;e.className='pill '+cls;}};
    set('pillYolo',d.yolo?'Available':'Not Installed',d.yolo?'pill-ok':'pill-warn');
    set('pillModel',d.model_loaded?'Loaded':'Demo Mode',d.model_loaded?'pill-ok':'pill-warn');
    set('pillStream',d.running?'ACTIVE':'IDLE',d.running?'pill-live':'pill-off');
    set('pillMode',(d.mode||'idle').toUpperCase(),d.running?'pill-live':'pill-off');
    document.getElementById('chipModel').textContent=d.model_loaded?'Custom PPE':'YOLOv8 Base';
    const fps=d.fps||0;
    document.getElementById('fpsNum').textContent=fps.toFixed(1);
    document.getElementById('fpsFill').style.width=Math.min(fps/30*100,100)+'%';
    document.getElementById('fpsStatus').textContent=d.running?'RUNNING':'IDLE';
    if(d.mode==='stream'||d.mode==='idle')updateCounts(d.counts);
    if(d.mode==='stream')updateDets(d.detections);
    updateAlerts(d.alerts);
  }catch(e){}
}
setInterval(poll,900);poll();

/* ── Helpers ── */
function show(id){document.getElementById(id).style.display='block';}
function hide(id){document.getElementById(id).style.display='none';}
let _tt;
function toast(msg,type='info'){
  const t=document.getElementById('toast');
  const c={ok:'var(--sage)',warn:'var(--amber)',danger:'var(--red)',info:'var(--rose)'};
  t.textContent=msg;t.style.borderColor=c[type]||c.info;
  t.classList.add('show');clearTimeout(_tt);_tt=setTimeout(()=>t.classList.remove('show'),3000);
}
onSrcChange();
</script>
</body>
</html>"""

if __name__=="__main__":
    logger.info("="*55)
    logger.info("  AI PPE Detection System v2.0 — Pinterest Edition")
    logger.info("="*55)
    load_model()
    logger.info(f"  YOLO: {YOLO_AVAILABLE}  |  Model: {state['model_loaded']}")
    logger.info("  → http://127.0.0.1:5000")
    logger.info("="*55)
   import os

if __name__ == "__main__":
    logger.info("="*55)
    logger.info(" AI PPE Detection System v2.0 - Pinterest Edition")
    logger.info("="*55)

    load_model()

    logger.info(f" YOLO: {YOLO_AVAILABLE} | Model: {state['model_loaded']}")
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f" Running on port: {port}")

    app.run(host="0.0.0.0", port=port)
