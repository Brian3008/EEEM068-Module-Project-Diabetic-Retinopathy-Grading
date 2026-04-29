import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


def get_transforms(image_size: int = 512, mode: str = "train"):
    """
    No Ben Graham preprocessing — already applied in processed dataset.
    Train: augmentation to improve generalisation.
    Val: clean resize only.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


class DRDataset(Dataset):
    """
    Loads DR images using CSV file with image names and labels.
    Images are already Ben Graham preprocessed at 512x512.
    CSV columns: image (filename without extension), level (0-4)
    """
    def __init__(self, csv_path: str, image_dir: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.samples   = []

        for _, row in self.df.iterrows():
            img_name = str(row["image"])
            label    = int(row["level"])
            # Try jpeg first then jpg then png
            for ext in [".jpeg", ".jpg", ".png"]:
                img_path = os.path.join(image_dir, img_name + ext)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                    break

        print(f"  Dataset loaded: {len(self.samples)} images total")
        self._print_class_dist()

    def _print_class_dist(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        names  = {0: "No DR", 1: "Mild", 2: "Moderate",
                  3: "Severe", 4: "Proliferative"}
        for i in range(5):
            print(f"    Class {i} ({names[i]}): {counts.get(i, 0)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]


def get_sampler(labels: list) -> WeightedRandomSampler:
    """
    Sqrt inverse frequency sampler.
    Boosts minority classes without completely drowning No DR.
    Best balance found across all previous runs.
    """
    counts         = np.bincount(labels)
    class_weights  = 1.0 / np.sqrt(counts.astype(float))
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )