import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import cfg


def crop_retina(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Crop image to the bounding box of the retina (mask of pixels > threshold)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image[y0:y1 + 1, x0:x1 + 1]


def ben_graham(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    image = crop_retina(image)

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target_size - new_h) // 2
    pad_bot = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    image = cv2.copyMakeBorder(
        image, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    sigma = target_size / 30.0
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)

    circle_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(circle_mask, (target_size // 2, target_size // 2),
               int(target_size * 0.49), 1, thickness=-1)
    enhanced = enhanced * circle_mask[..., None] + 128 * (1 - circle_mask[..., None])
    return enhanced.astype(np.uint8)


def _process_one(args):
    src, dst, target_size = args
    if dst.exists():
        return True
    try:
        img = cv2.imread(str(src))
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = ben_graham(img, target_size)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"  ! failed {src.name}: {e}")
        return False


def preprocess_all():
    raw_dir = Path(cfg.RAW_DATA_DIR)
    out_dir = Path(cfg.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data dir not found: {raw_dir}\n"
            f"Download train.zip.001..005 + trainLabels.csv from "
            f"https://www.kaggle.com/competitions/diabetic-retinopathy-detection "
            f"and extract train/*.jpeg into {raw_dir}"
        )

    labels_df = pd.read_csv(cfg.LABELS_CSV)
    print(f"  Found {len(labels_df)} entries in {cfg.LABELS_CSV}")
    print(f"  Class distribution:")
    for cls, count in labels_df['level'].value_counts().sort_index().items():
        print(f"    class {cls}: {count:6d}  ({100*count/len(labels_df):.1f}%)")

    tasks = []
    missing = 0
    for _, row in labels_df.iterrows():
        src = raw_dir / f"{row['image']}.jpeg"
        dst = out_dir / f"{row['image']}.jpeg"
        if src.exists():
            tasks.append((src, dst, cfg.PROCESSED_SIZE))
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing} images listed in CSV but missing on disk")

    print(f"\n  Processing {len(tasks)} images @ {cfg.PROCESSED_SIZE}×{cfg.PROCESSED_SIZE}")
    print(f"  Writing to: {out_dir}")
    print(f"  Using {cfg.NUM_WORKERS} worker processes")

    with ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as ex:
        results = list(tqdm(
            ex.map(_process_one, tasks, chunksize=8),
            total=len(tasks),
            desc="preprocess"
        ))

    success = sum(results)
    print(f"\n  Done. {success}/{len(tasks)} images processed successfully.")


if __name__ == "__main__":
    preprocess_all()