"""
train_ppe_model.py
──────────────────
Custom YOLOv8 PPE Model Trainer
Final Year B.Tech Project — AI PPE Detection System

Requirements:
    pip install ultralytics albumentations

Usage:
    python train_ppe_model.py
"""

from ultralytics import YOLO
import yaml, os, shutil, urllib.request

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "model_base":   "yolov8n.pt",        # Start from YOLOv8 Nano (fast, lightweight)
    "epochs":        50,                  # Increase to 100+ for production
    "batch":         16,
    "imgsz":         640,
    "lr0":           0.01,
    "lrf":           0.01,
    "momentum":      0.937,
    "weight_decay":  0.0005,
    "warmup_epochs": 3,
    "conf_thresh":   0.35,
    "iou_thresh":    0.45,
    "output_name":   "ppe_model.pt",
    "data_yaml":     "dataset/data.yaml",
    # Data Augmentation (applied automatically by Ultralytics)
    "augment": True,
    "fliplr":  0.5,    # 50% horizontal flip
    "flipud":  0.0,
    "degrees": 5.0,    # ±5° rotation
    "translate": 0.1,
    "scale":   0.5,
    "hsv_h":   0.015,  # HSV hue augment
    "hsv_s":   0.7,    # HSV saturation augment
    "hsv_v":   0.4,    # HSV brightness augment
}

# ─── Sample data.yaml (edit to match your dataset) ───────────────────────────
SAMPLE_YAML = """
path: ./dataset
train: images/train
val: images/val

nc: 5
names:
  0: helmet
  1: no_helmet
  2: vest
  3: no_vest
  4: person

# Augmentation hints (Ultralytics handles these automatically)
"""

def create_sample_yaml():
    os.makedirs("dataset", exist_ok=True)
    if not os.path.exists("dataset/data.yaml"):
        with open("dataset/data.yaml", "w") as f:
            f.write(SAMPLE_YAML.strip())
        print("Created dataset/data.yaml — edit paths to match your dataset.")

def train():
    print("=" * 55)
    print("  YOLOv8 PPE Model Training")
    print("=" * 55)

    # Check dataset
    if not os.path.exists(CONFIG["data_yaml"]):
        create_sample_yaml()
        print("\n⚠  dataset/data.yaml not found — a sample was created.")
        print("   Please:")
        print("   1. Add images to dataset/images/train and dataset/images/val")
        print("   2. Add YOLO labels to dataset/labels/train and dataset/labels/val")
        print("   3. Run this script again")
        return

    # Load base model
    model = YOLO(CONFIG["model_base"])
    print(f"\n✓ Loaded base model: {CONFIG['model_base']}")
    print(f"  Epochs: {CONFIG['epochs']}, Batch: {CONFIG['batch']}, ImgSz: {CONFIG['imgsz']}")
    print(f"  Augmentation: fliplr={CONFIG['fliplr']}, hsv_v={CONFIG['hsv_v']}, degrees={CONFIG['degrees']}\n")

    # Train
    results = model.train(
        data       = CONFIG["data_yaml"],
        epochs     = CONFIG["epochs"],
        batch      = CONFIG["batch"],
        imgsz      = CONFIG["imgsz"],
        lr0        = CONFIG["lr0"],
        lrf        = CONFIG["lrf"],
        momentum   = CONFIG["momentum"],
        weight_decay = CONFIG["weight_decay"],
        warmup_epochs = CONFIG["warmup_epochs"],
        augment    = CONFIG["augment"],
        fliplr     = CONFIG["fliplr"],
        flipud     = CONFIG["flipud"],
        degrees    = CONFIG["degrees"],
        translate  = CONFIG["translate"],
        scale      = CONFIG["scale"],
        hsv_h      = CONFIG["hsv_h"],
        hsv_s      = CONFIG["hsv_s"],
        hsv_v      = CONFIG["hsv_v"],
        project    = "ppe_training",
        name       = "ppe_run",
        exist_ok   = True,
        verbose    = True,
    )

    # Copy best weights
    best = "ppe_training/ppe_run/weights/best.pt"
    if os.path.exists(best):
        shutil.copy(best, CONFIG["output_name"])
        print(f"\n✓ Custom model saved: {CONFIG['output_name']}")
        print("  Place ppe_model.pt alongside app.py and restart the server.")
    else:
        print("\n⚠ Training complete but best.pt not found. Check ppe_training/ folder.")

    # Validate
    print("\n── Validation ────────────────────────────────────")
    val_model = YOLO(CONFIG["output_name"] if os.path.exists(CONFIG["output_name"]) else best)
    val_model.val(
        data     = CONFIG["data_yaml"],
        conf     = CONFIG["conf_thresh"],
        iou      = CONFIG["iou_thresh"],
        imgsz    = CONFIG["imgsz"],
        verbose  = True,
    )
    print("\n✓ Training complete!")

if __name__ == "__main__":
    train()
