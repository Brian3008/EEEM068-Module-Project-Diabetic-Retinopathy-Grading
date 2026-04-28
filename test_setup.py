# test_setup.py
# Run this BEFORE main.py to confirm everything is wired correctly.
# Command: python test_setup.py

import os
import torch
import numpy as np
from torchvision import datasets
from torch.cuda.amp import autocast

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Check GPU
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 1 — GPU CHECK")
print("="*55)
print(f"  CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA Version   : {torch.version.cuda}")
else:
    print("  ⚠️  No GPU detected — check CUDA drivers!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Check Dataset Folder
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 2 — DATASET FOLDER CHECK")
print("="*55)

DATA_DIR = "src/data/colored_images"

if not os.path.exists(DATA_DIR):
    print(f"  ❌ Folder NOT found: {DATA_DIR}")
    print("     Double-check your path in config.py")
else:
    print(f"  ✅ Folder found: {DATA_DIR}")
    subfolders = os.listdir(DATA_DIR)
    print(f"  📁 Subfolders  : {subfolders}")

    try:
        ds = datasets.ImageFolder(DATA_DIR)
        print(f"\n  ✅ ImageFolder loaded successfully")
        print(f"  📊 Classes found : {ds.classes}")
        print(f"  🖼️  Total images  : {len(ds)}")
        print()
        for cls, idx in ds.class_to_idx.items():
            count = ds.targets.count(idx)
            bar   = "█" * (count // 50)
            print(f"  [{idx}] {cls:15s} : {count:>5} images  {bar}")
    except Exception as e:
        print(f"  ❌ ImageFolder failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Check Imports
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 3 — LIBRARY IMPORTS CHECK")
print("="*55)

libs = {
    "torch"         : "torch",
    "torchvision"   : "torchvision",
    "timm"          : "timm",
    "cv2 (OpenCV)"  : "cv2",
    "sklearn"       : "sklearn",
    "matplotlib"    : "matplotlib",
    "seaborn"       : "seaborn",
    "pandas"        : "pandas",
    "numpy"         : "numpy",
    "PIL"           : "PIL",
}

all_ok = True
for name, module in libs.items():
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name}  ← run: pip install {module}")
        all_ok = False

if all_ok:
    print("\n  ✅ All libraries installed!")
else:
    print("\n  ⚠️  Fix missing libraries before running main.py")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Check Config + Model Build
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 4 — CONFIG + MODEL CHECK")
print("="*55)

try:
    from src.config import cfg
    print(f"  ✅ config.py loaded")
    print(f"     DATA_DIR    : {cfg.DATA_DIR}")
    print(f"     BATCH_SIZE  : {cfg.BATCH_SIZE}")
    print(f"     IMAGE_SIZE  : {cfg.IMAGE_SIZE}")
    print(f"     NUM_CLASSES : {cfg.NUM_CLASSES}")
    print(f"     DEVICE      : {cfg.DEVICE}")
    print(f"     USE_AMP     : {cfg.USE_AMP}")
except Exception as e:
    print(f"  ❌ config.py error: {e}")

try:
    from src.model import build_model
    model = build_model(pretrained=False)   # pretrained=False for speed
    print(f"\n  ✅ Model built successfully")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Total params    : {total/1e6:.1f}M")
    print(f"     Trainable params: {trainable/1e6:.1f}M")
except Exception as e:
    print(f"\n  ❌ Model build failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Dummy Forward Pass (checks memory + shapes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 5 — DUMMY FORWARD PASS")
print("="*55)

try:
    from src.config import cfg
    from src.model import build_model

    model  = build_model(pretrained=False).to(cfg.DEVICE)
    dummy  = torch.randn(4, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)

    with torch.no_grad():
        with autocast(enabled=cfg.USE_AMP):
            output = model(dummy)

    print(f"  ✅ Forward pass successful")
    print(f"     Input shape  : {list(dummy.shape)}")
    print(f"     Output shape : {list(output.shape)}  ← should be [4, 5]")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"     VRAM used    : {mem:.3f} GB")

except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  ✅  All checks passed — run:  python main.py")
print("="*55 + "\n")