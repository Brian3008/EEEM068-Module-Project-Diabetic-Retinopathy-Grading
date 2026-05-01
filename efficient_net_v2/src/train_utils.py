import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from.config import cfg


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float], str, list, list]:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    model.to(device)

    train_losses = []
    val_losses = []
    val_qwk_scores = []

    best_qwk = -1.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")

    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_qwk_scores.append(qwk)

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{cfg.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {qwk:.4f}"
        )

        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with QWK: {best_qwk:.4f}")

    print("\nClassification Report (last epoch):\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            digits=4
        )
    )

    return train_losses, val_losses, val_qwk_scores, best_model_path, all_labels, all_preds


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_qwk_scores: List[float],
):
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = range(1, len(train_losses) + 1)

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_qwk_scores, color="green", label="Val QWK")
    plt.title("Validation Quadratic Weighted Kappa")
    plt.xlabel("Epoch")
    plt.ylabel("QWK")
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(cfg.PLOTS_DIR, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_confusion_matrices(
    all_labels,
    all_preds,
    class_names,
):
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix (counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix (normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    save_path = os.path.join(cfg.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plots saved to: {save_path}")
