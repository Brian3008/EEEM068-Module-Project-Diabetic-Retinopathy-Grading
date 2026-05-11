# ============================================================
# PART 1 — Local paths and folders 
# ============================================================
import os

DATA_DIR = "/scratch/processed"

CSV_PATH = os.path.join(DATA_DIR, "trainLabels.csv")
IMAGE_DIR = DATA_DIR

OUTPUT_DIR = os.path.join(DATA_DIR, "outputs_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
if os.path.exists(DATA_DIR):
    print("Contents of DATA_DIR:", os.listdir(DATA_DIR))

print("CSV_PATH:", CSV_PATH)
print("IMAGE_DIR:", IMAGE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Subfolders in OUTPUT_DIR:", os.listdir(OUTPUT_DIR))

# ============================================================
# PART 2 — Imports 
# ============================================================
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import transforms

from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, classification_report,
    f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import timm

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# PART 3 — Configuration
# Student Name: Balaji Periyadurai | URN: 6962046
# Model: EfficientNetV2-S
# ============================================================
SEED            = 42
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LR_BACKBONE     = 3e-6
LR_HEAD         = 3e-5
WEIGHT_DECAY    = 0.01
VAL_SPLIT       = 0.15
NUM_WORKERS     = 1
LABEL_SMOOTHING = 0.1
PATIENCE        = 7
ACCUMULATION    = 2

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

print("Device:", device)
print("Student: Balaji Periyadurai | URN: 6962046")
print("CSV_PATH:", CSV_PATH)
print("IMAGE_DIR:", IMAGE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# ============================================================
# PART 4 — Dataset visualisation: class distribution
# ============================================================
df = pd.read_csv(CSV_PATH)
print(f"Total images: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

counts = df["level"].value_counts().sort_index()
print("\nClass distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  Class {i} ({name}): {counts.get(i, 0)} images "
          f"({counts.get(i, 0)/len(df)*100:.1f}%)")

colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Class Distribution — Diabetic Retinopathy\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=13, fontweight="bold")

bars = axes[0].bar(CLASS_NAMES, counts.values, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("Image Count per DR Grade")
axes[0].set_xlabel("DR Severity Grade")
axes[0].set_ylabel("Number of Images")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 str(val), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=CLASS_NAMES, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="black", linewidth=0.5))
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Class distribution plot saved!")

# ============================================================
# PART 5 — Sample images per DR grade
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Sample Retinal Images — One Per DR Severity Grade\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=13, fontweight="bold")

colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

for grade in range(5):
    subset = df[df["level"] == grade]
    img_name = subset.iloc[0]["image"]

    img_path = None
    for ext in [".jpeg", ".jpg", ".png"]:
        candidate = os.path.join(IMAGE_DIR, img_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"Warning: no image found for {img_name} (grade {grade})")
        axes[grade].axis("off")
        continue

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axes[grade].imshow(image)
    axes[grade].set_title(f"Grade {grade}\n{CLASS_NAMES[grade]}",
                          fontsize=11, fontweight="bold",
                          color=colors[grade])
    axes[grade].axis("off")

    for spine in axes[grade].spines.values():
        spine.set_edgecolor(colors[grade])
        spine.set_linewidth(3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/sample_images.png", dpi=150, bbox_inches="tight")
plt.close()
print("Sample images plot saved!")

# ============================================================
# PART 6 — Dataset class, transforms, loaders
# ============================================================
from torchvision import transforms

def get_transforms(image_size=IMAGE_SIZE, mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
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


class DRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.samples   = []

        for _, row in self.df.iterrows():
            img_name = str(row["image"])
            label    = int(row["level"])
            for ext in [".jpeg", ".jpg", ".png"]:
                img_path = os.path.join(image_dir, img_name + ext)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                    break

        print(f"Dataset loaded: {len(self.samples)} images")
        counts = Counter(label for _, label in self.samples)
        for i, name in enumerate(CLASS_NAMES):
            print(f"  Class {i} ({name}): {counts.get(i, 0)}")

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


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_sampler(labels):
    counts         = np.bincount(labels)
    class_weights  = 1.0 / np.sqrt(counts.astype(float))
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


print("Loading dataset...")
full_dataset = DRDataset(CSV_PATH, IMAGE_DIR,
                         transform=get_transforms(IMAGE_SIZE, "train"))

val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_ds_clean = SubsetWithTransform(val_ds, get_transforms(IMAGE_SIZE, "val"))

train_labels = [full_dataset.samples[i][1] for i in train_ds.indices]
sampler      = get_sampler(train_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds_clean, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"\nTrain: {train_size} images | Val: {val_size} images")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ============================================================
# PART 7 — Model: EfficientNetV2-S
# ============================================================
def build_efficientnetv2_model(num_classes=5, dropout=0.3):
    model = timm.create_model(
        "tf_efficientnetv2_s",     # <- tf_ prefix
        pretrained=True,           # this usually has weights
        num_classes=num_classes
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_efficientnetv2_model(NUM_CLASSES, dropout=0.3).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model: EfficientNetV2-S")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable:,}")

# ============================================================
# PART 8 — Loss, optimiser, scheduler (single LR)
# ============================================================
all_labels   = full_dataset.get_labels()
label_counts = [Counter(all_labels)[i] for i in range(NUM_CLASSES)]
print("Label counts:", label_counts)

counts_arr    = torch.tensor(label_counts, dtype=torch.float)
class_weights = 1.0 / torch.sqrt(counts_arr)
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)
print("Class weights:", class_weights.cpu().numpy().round(3))

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=WEIGHT_DECAY)

warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
cosine    = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - 3, eta_min=1e-7)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

from torch.cuda.amp import GradScaler
scaler = GradScaler(enabled=(device.type == "cuda"))

print("Loss, optimiser and scheduler ready!")

# ============================================================
# PART 9 — Training and validation functions
# ============================================================
from torch import autocast

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels) / ACCUMULATION

        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


print("Training functions defined!")

# ============================================================
# PART 10 — Training loop with early stopping
# ============================================================
best_qwk      = -1.0
no_improve    = 0
train_losses, val_losses, val_qwks = [], [], []
val_preds_final = val_labels_final = val_probs_final = None

print(f"Starting training — max {NUM_EPOCHS} epochs, early stopping patience={PATIENCE}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")

    train_loss, _, _ = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scaler
    )
    val_loss, val_preds, val_labels_ep, val_probs = validate(
        model, val_loader, criterion, device
    )
    scheduler.step()

    qwk      = cohen_kappa_score(val_labels_ep, val_preds, weights="quadratic")
    f1_macro = f1_score(val_labels_ep, val_preds, average="macro", zero_division=0)
    acc      = (val_preds == val_labels_ep).mean()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_qwks.append(qwk)

    print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    print(f"  QWK: {qwk:.4f} | F1 macro: {f1_macro:.4f} | Acc: {acc:.4f}")

    if qwk > best_qwk:
        best_qwk         = qwk
        no_improve       = 0
        val_preds_final  = val_preds
        val_labels_final = val_labels_ep
        val_probs_final  = val_probs
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "qwk": qwk,
        }, f"{OUTPUT_DIR}/checkpoints/best_efficientnetv2_s.pth")
        print(f"  ✓ New best QWK {best_qwk:.4f} — saved!")
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nTraining complete! Best QWK: {best_qwk:.4f}")

# ============================================================
# PART 11 — Training curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EfficientNetV2-S Training Curves — Diabetic Retinopathy Grading\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=13, fontweight="bold")

epochs_range = range(1, len(train_losses) + 1)

axes[0].plot(epochs_range, train_losses, "b-o", markersize=4, label="Train Loss")
axes[0].plot(epochs_range, val_losses,   "r-o", markersize=4, label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, val_qwks, "g-o", markersize=4, label="Val QWK")
axes[1].axhline(y=best_qwk, color="darkgreen", linestyle="--",
                label=f"Best QWK = {best_qwk:.4f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("QWK")
axes[1].set_title("Validation Quadratic Weighted Kappa")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Training curves saved!")

# ============================================================
# PART 12 — Confusion matrix & classification report
# ============================================================
cm      = confusion_matrix(val_labels_final, val_preds_final)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Confusion Matrix — EfficientNetV2-S DR Grading\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=13, fontweight="bold")

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
    ["d", ".2f"]
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── Classification Report ──")
print(classification_report(val_labels_final, val_preds_final,
                             target_names=CLASS_NAMES, zero_division=0))
acc = (val_preds_final == val_labels_final).mean()
print(f"Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Best QWK         : {best_qwk:.4f}")
print(f"Macro F1         : {f1_score(val_labels_final, val_preds_final, average='macro', zero_division=0):.4f}")

# ============================================================
# PART 13 — ROC Curve + AUC
# ============================================================
y_true_bin = label_binarize(val_labels_final, classes=list(range(NUM_CLASSES)))

fig, ax = plt.subplots(figsize=(10, 7))
colors  = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

auc_scores = []
for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], val_probs_final[:, i])
    auc_val     = auc(fpr, tpr)
    auc_scores.append(auc_val)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"ROC Curves — EfficientNetV2-S DR Grading\n"
             f"Student: Balaji Periyadurai | URN: 6962046\n"
             f"Mean AUC = {np.mean(auc_scores):.3f}",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"ROC curves saved! Mean AUC: {np.mean(auc_scores):.4f}")

# ============================================================
# PART 14 — Per-class metrics
# ============================================================
report = classification_report(val_labels_final, val_preds_final,
                                target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)

metrics_data = {
    "Precision": [report[c]["precision"] for c in CLASS_NAMES],
    "Recall":    [report[c]["recall"]    for c in CLASS_NAMES],
    "F1-Score":  [report[c]["f1-score"]  for c in CLASS_NAMES],
}

x      = np.arange(len(CLASS_NAMES))
width  = 0.25
colors = ["#3498db", "#e74c3c", "#2ecc71"]

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metric, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric,
                  color=colors[i], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xlabel("DR Severity Grade", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Precision, Recall & F1-Score — EfficientNetV2-S\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/per_class_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Per-class metrics chart saved!")

# ============================================================
# PART 15 — GradCAM Attention Heatmaps 
# ============================================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load best EfficientNetV2-S checkpoint (ensure path matches PART 10)
ckpt_path = f"{OUTPUT_DIR}/checkpoints/best_efficientnetv2_s.pth"
print("Loading checkpoint for GradCAM:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Target layer for CNN: last conv layer before classifier
target_layers = [model.conv_head]

cam = GradCAM(
    model=model,
    target_layers=target_layers,   # no reshape_transform needed for CNN
)

val_transform = get_transforms(IMAGE_SIZE, "val")
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle("GradCAM Attention Maps — Where EfficientNetV2-S Looks for DR Features\n"
             "Student: Balaji Periyadurai | URN: 6962046",
             fontsize=14, fontweight="bold")

colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

for grade in range(5):
    subset = df[df["level"] == grade]
    if len(subset) == 0:
        print(f"No samples found for grade {grade}")
        for row in range(3):
            axes[row, grade].axis("off")
        continue

    # Use 4th image if available, otherwise last one
    idx = min(3, len(subset) - 1)
    img_name = subset.iloc[idx]["image"]

    # Find image file
    img_path = None
    for ext in [".jpeg", ".jpg", ".png"]:
        candidate = os.path.join(IMAGE_DIR, img_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"Warning: no image found for {img_name} (grade {grade})")
        for row in range(3):
            axes[row, grade].axis("off")
        continue

    # Load original image
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE))
    orig_float   = orig_resized.astype(np.float32) / 255.0

    # Prepare tensor
    input_tensor = val_transform(
        Image.fromarray(orig_resized)
    ).unsqueeze(0).to(device)

    # Generate CAM
    targets   = [ClassifierOutputTarget(grade)]
    grayscale = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(orig_float, grayscale[0], use_rgb=True)

    # Row 1 — original
    axes[0, grade].imshow(orig_resized)
    axes[0, grade].set_title(f"Grade {grade}: {CLASS_NAMES[grade]}",
                              fontweight="bold", color=colors[grade])
    axes[0, grade].axis("off")

    # Row 2 — GradCAM overlay
    axes[1, grade].imshow(cam_image)
    axes[1, grade].set_title("GradCAM Overlay", fontsize=9)
    axes[1, grade].axis("off")

    # Row 3 — heatmap only
    axes[2, grade].imshow(grayscale[0], cmap="jet")
    axes[2, grade].set_title("Attention Heatmap", fontsize=9)
    axes[2, grade].axis("off")

# Row labels
for row, label in enumerate(["Original Image", "GradCAM Overlay", "Attention Heatmap"]):
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                             rotation=90, labelpad=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/gradcam_heatmaps_efficientnetv2_s.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("GradCAM heatmaps saved!")

# ============================================================
# PART 16 — Final Summary
# ============================================================
acc_final      = (val_preds_final == val_labels_final).mean()
f1_macro_final = f1_score(val_labels_final, val_preds_final, average="macro", zero_division=0)
f1_per_final   = f1_score(val_labels_final, val_preds_final, average=None,   zero_division=0)

print("=" * 60)
print("FINAL RESULTS SUMMARY — EfficientNetV2-S")
print("Student: Balaji Periyadurai | URN: 6962046")
print("Model: EfficientNetV2-S")
print("Dataset: Processed Kaggle DR Dataset")
print("=" * 60)
print(f"Best QWK (primary metric) : {best_qwk:.4f}")
print(f"Overall Accuracy          : {acc_final:.4f} ({acc_final*100:.2f}%)")
print(f"Macro F1                  : {f1_macro_final:.4f}")
print(f"Mean AUC                  : {np.mean(auc_scores):.4f}")
print("-" * 60)
print("Per-class F1 scores:")
for i, (name, score) in enumerate(zip(CLASS_NAMES, f1_per_final)):
    print(f"  Class {i} ({name:15s}): {score:.4f}")
print("-" * 60)
print("Imbalance handling strategies:")
print("  1. Sqrt inverse frequency WeightedRandomSampler")
print("  2. Sqrt inverse frequency class weights in CrossEntropy")
print("  3. Label smoothing = 0.1")
print("  4. LR warmup 3 epochs + cosine annealing")
print("=" * 60)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/plots/")
print("Files saved:")
for f in sorted(os.listdir(os.path.join(OUTPUT_DIR, "plots"))):
    print(f"  {f}")
