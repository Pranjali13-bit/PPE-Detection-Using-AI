"""
download_model.py
─────────────────
Run this ONCE to download the best available full-PPE YOLOv8 model.
It detects: Helmet, No-Helmet, Safety Vest, No-Vest, Gloves, Glasses/Mask, Boots, Person

Usage:
    python download_model.py

The model is saved as ppe_model.pt in the same folder.
Then just run:  python app.py
"""

import urllib.request, os, sys

MODELS = [
    # Best: detects helmet, vest, gloves, mask, boots, person
    ("https://huggingface.co/keremberke/yolov8m-ppe-detection/resolve/main/best.pt",
     "Full PPE (medium, ~50MB) — helmet/vest/gloves/mask/boots"),
    # Lighter: same classes, faster on CPU
    ("https://huggingface.co/keremberke/yolov8n-ppe-detection/resolve/main/best.pt",
     "Full PPE (nano, ~6MB) — helmet/vest/gloves/mask/boots, faster"),
    # Fallback: helmet only
    ("https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt",
     "Helmet only (nano, ~6MB) — helmet/head detection"),
]

OUTPUT = "ppe_model.pt"

def reporthook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%  ", end="", flush=True)

def try_download(url, fname):
    try:
        print(f"\n  Downloading from:\n  {url}")
        urllib.request.urlretrieve(url, fname, reporthook)
        print()
        size = os.path.getsize(fname)
        if size < 500_000:
            print(f"  ⚠ File too small ({size} bytes), skipping.")
            os.remove(fname)
            return False
        print(f"  ✓ Saved: {fname} ({size // 1024 // 1024}MB)")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if os.path.exists(fname):
            os.remove(fname)
        return False

def main():
    print("=" * 55)
    print("  PPE Model Downloader")
    print("=" * 55)

    if os.path.exists(OUTPUT):
        print(f"\n  ✓ {OUTPUT} already exists ({os.path.getsize(OUTPUT)//1024}KB)")
        ans = input("  Re-download? (y/N): ").strip().lower()
        if ans != "y":
            print("  Using existing model.")
            return

    for url, desc in MODELS:
        print(f"\n  Trying: {desc}")
        if try_download(url, OUTPUT):
            # Print what classes this model detects
            try:
                from ultralytics import YOLO
                m = YOLO(OUTPUT)
                print(f"\n  Model classes: {list(m.names.values())}")
            except Exception:
                pass
            print(f"\n  ✓ Done! Now run:  python app.py\n")
            return

    print("\n  ✗ All downloads failed.")
    print("  Manual option: Go to https://universe.roboflow.com")
    print("  Search 'PPE detection YOLOv8', download best.pt, rename to ppe_model.pt\n")
    sys.exit(1)

if __name__ == "__main__":
    main()
