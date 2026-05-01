import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from.config import cfg


def _severity_class_names():
    return list(cfg.CLASS_NAMES)


def _build_label_remap(imagefolder_classes):
    folder_to_severity = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4,
    }

    missing = [c for c in imagefolder_classes if c not in folder_to_severity]
    if missing:
        raise ValueError(
            "Unknown class folder(s) in DATA_DIR: "
            f"{missing}. Expected: {sorted(folder_to_severity.keys())}"
        )

    return [folder_to_severity[c] for c in imagefolder_classes]


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class DRFolderDataset(Dataset):
    def __init__(self, root: str, transform=None, use_clahe: bool = True):
        self.base = datasets.ImageFolder(root)
        self.transform = transform
        self.use_clahe = use_clahe

        self._label_remap = _build_label_remap(self.base.classes)
        self.classes = _severity_class_names()

        self.targets = [self._label_remap[t] for t in self.base.targets]
        self.samples = self.base.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, old_label = self.base.samples[idx]
        label = self._label_remap[old_label]

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_clahe:
            image = apply_clahe(image)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders():
    full_base = datasets.ImageFolder(cfg.DATA_DIR)
    total = len(full_base)

    remap = _build_label_remap(full_base.classes)
    severity_targets = np.array([remap[t] for t in full_base.targets], dtype=np.int64)

    all_indices = np.arange(total)
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=cfg.VAL_SPLIT,
        random_state=cfg.SEED,
        shuffle=True,
        stratify=severity_targets,
    )
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()
    train_size = len(train_indices)
    val_size = len(val_indices)

    train_dataset = DRFolderDataset(
        root=cfg.DATA_DIR,
        transform=get_train_transforms(),
        use_clahe=True,
    )
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = DRFolderDataset(
        root=cfg.DATA_DIR,
        transform=get_val_transforms(),
        use_clahe=True,
    )
    val_subset = Subset(val_dataset, val_indices)

    print(f"\n📂 Dataset loaded from: {cfg.DATA_DIR}")
    print(f"   Folders found : {full_base.classes}")
    print(f"   Severity order: {_severity_class_names()}")
    print(f"   Total images  : {total}")
    counts = np.bincount(severity_targets, minlength=cfg.NUM_CLASSES)
    for idx, name in enumerate(_severity_class_names()):
        print(f"   {name:15s}: {int(counts[idx])} images")

    sampler = None
    if getattr(cfg, "USE_WEIGHTED_SAMPLER", False):
        train_labels = severity_targets[np.array(train_indices, dtype=np.int64)]
        train_counts = np.bincount(train_labels, minlength=cfg.NUM_CLASSES)
        per_class_weight = 1.0 / train_counts
        sample_weights = per_class_weight[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    print(f"\n   Train samples : {train_size}")
    print(f"   Val samples   : {val_size}")

    return train_loader, val_loader, _severity_class_names()
