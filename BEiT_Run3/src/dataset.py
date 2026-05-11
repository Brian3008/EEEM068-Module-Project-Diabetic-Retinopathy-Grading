import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


def ben_graham_preprocessing(image: np.ndarray, sigma: int = 10) -> np.ndarray:
    """
    Ben Graham's preprocessing: subtract local mean colour to enhance
    retinal features (used by top Kaggle DR solutions).
    """
    blurred   = cv2.GaussianBlur(image, (0, 0), sigma)
    processed = cv2.addWeighted(image, 4, blurred, -4, 128)
    return processed


class BenGrahamTransform:
    """Wraps Ben Graham preprocessing as a callable transform."""
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = ben_graham_preprocessing(arr)
        return Image.fromarray(arr)


def get_transforms(image_size: int = 224, mode: str = "train"):
    ben = BenGrahamTransform()
    if mode == "train":
        return transforms.Compose([
            ben,
            transforms.Resize((256, 256)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.05),
        ])
    else:
        return transforms.Compose([
            ben,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


FOLDER_TO_LABEL = {
    "No_DR":          0,
    "Mild":           1,
    "Moderate":       2,
    "Severe":         3,
    "Proliferate_DR": 4,
}


class DRFolderDataset(Dataset):
    """
    Loads DR images directly from class-named subfolders.
    No CSV required. Folder names map to DR severity grades 0-4.
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples   = []
        self.transform = transform

        for folder_name, label in FOLDER_TO_LABEL.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  Warning: folder not found → {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder_path, fname), label)
                    )

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
    Balanced sampler using cube-root inverse frequency.
    Gentler than sqrt — gives minority classes a boost without
    making the model ignore No_DR completely.
    """
    counts         = np.bincount(labels)
    class_weights  = 1.0 / np.cbrt(counts.astype(float))  # cube root
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )