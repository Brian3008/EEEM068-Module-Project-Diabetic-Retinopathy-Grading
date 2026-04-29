import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from collections import Counter
import numpy as np

from src.dataset import DRDataset, get_transforms, get_sampler
from src.model import build_beit_model, get_class_weights
from src.train import train_one_epoch, validate
from src.metrics import compute_all_metrics, plot_confusion_matrix, plot_training_curves
from src.utils import set_seed, save_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH     = "data/processed/trainLabels.csv"
IMAGE_DIR    = "data/processed"
OUTPUT_DIR   = "outputs_v7"
SEED         = 42
IMAGE_SIZE   = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 25
LR           = 2e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT    = 0.15
NUM_WORKERS  = 0
# ─────────────────────────────────────────────────────────────────────────────


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        import cv2
        from PIL import Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    full_dataset = DRDataset(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        transform=get_transforms(IMAGE_SIZE, "train")
    )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_ds_clean = SubsetWithTransform(
        val_ds, get_transforms(IMAGE_SIZE, "val")
    )

    train_labels = [full_dataset.samples[i][1] for i in train_ds.indices]
    sampler      = get_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,     batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds_clean, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding BEiT model...")
    model = build_beit_model(num_classes=5).to(device)

    all_labels   = full_dataset.get_labels()
    label_counts = [Counter(all_labels)[i] for i in range(5)]
    print(f"Label counts: {label_counts}")

    class_weights = get_class_weights(label_counts, device, soft=True)
    criterion     = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.1
    )

    head_params     = list(model.classifier.parameters())
    head_ids        = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = AdamW([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": head_params,     "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    warmup    = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=3
    )
    cosine    = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - 3, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[3]
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_qwk      = -1.0
    train_losses, val_losses, val_qwks = [], [], []
    metrics       = {}
    val_preds     = []
    val_labels_ep = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer,
            criterion, device, scaler
        )
        val_loss, val_preds, val_labels_ep = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        metrics = compute_all_metrics(val_labels_ep, val_preds)
        qwk     = metrics["qwk"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_qwks.append(qwk)

        print(f"  Train loss : {train_loss:.4f} | Val loss : {val_loss:.4f}")
        print(f"  Val QWK    : {qwk:.4f}       | Val F1   : {metrics['f1_macro']:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            save_checkpoint(
                model, optimizer, epoch, qwk,
                f"{OUTPUT_DIR}/checkpoints/best_beit.pth"
            )
            print(f"  New best QWK {best_qwk:.4f} — checkpoint saved.")

    # ── Final results ──────────────────────────────────────────────────────────
    print("\n── Final Classification Report ──")
    print(metrics.get("report", "No report generated."))

    plot_confusion_matrix(
        val_labels_ep, val_preds,
        save_path=f"{OUTPUT_DIR}/plots/confusion_matrix.png"
    )
    plot_training_curves(
        train_losses, val_losses, val_qwks,
        save_path=f"{OUTPUT_DIR}/plots/training_curves.png"
    )

    print(f"\nBest Validation QWK : {best_qwk:.4f}")
    print("Done. Outputs saved to outputs_v7/")


if __name__ == "__main__":
    main()