from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import cfg


class DRDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.image_dir / f"{row['image']}.jpeg"

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = int(row["level"])
        return img, torch.tensor(label, dtype=torch.long)



def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=cfg.NUM_CLASSES)
    total = int(labels.shape[0])
    weights = total / (cfg.NUM_CLASSES * counts.clip(min=1))
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders():
    csv_path = Path(cfg.LABELS_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep only images that exist in PROCESSED_DIR
    image_dir = Path(cfg.PROCESSED_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Processed image dir not found: {image_dir}\n"
            f"Run `python -m src.preprocess` first."
        )

    existing = {p.stem for p in image_dir.glob("*.jpeg")}
    df = df[df["image"].isin(existing)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(
            f"No preprocessed images matched entries in {csv_path}. "
            f"Did preprocessing complete?"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.VAL_SPLIT,
        random_state=cfg.SEED,
        stratify=df["level"],
        shuffle=True,
    )

    train_dataset = DRDataset(train_df, cfg.PROCESSED_DIR,
                              transform=get_train_transforms())
    val_dataset = DRDataset(val_df, cfg.PROCESSED_DIR,
                            transform=get_val_transforms())

    print(f"\n  Dataset: Kaggle DR-2015 (training split)")
    print(f"  Image dir: {cfg.PROCESSED_DIR}")
    print(f"  Total: {len(df)}  |  Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_labels = train_df["level"].values.astype(np.int64)
    train_counts = np.bincount(train_labels, minlength=cfg.NUM_CLASSES)
    print("\n  Train class distribution:")
    for i, name in enumerate(cfg.CLASS_NAMES):
        pct = 100 * train_counts[i] / len(train_df)
        print(f"    {name:18s}: {int(train_counts[i]):6d}  ({pct:.1f}%)")

    sampler = None
    shuffle = True
    if cfg.USE_WEIGHTED_SAMPLER:
        per_class_w = 1.0 / train_counts.clip(min=1)
        sample_w = per_class_w[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    return train_loader, val_loader, cfg.CLASS_NAMES, train_labels