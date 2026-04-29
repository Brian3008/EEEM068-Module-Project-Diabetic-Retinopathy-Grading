# plots.py — Shared plotting utilities

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import cfg


def _ensure_plots_dir():
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    _ensure_plots_dir()
    if save_path is None:
        save_path = os.path.join(cfg.PLOTS_DIR, "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def plot_training_curves(history, save_path=None):
    _ensure_plots_dir()
    if save_path is None:
        save_path = os.path.join(cfg.PLOTS_DIR, "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#3b82f6")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#ef4444")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_kappa"], label="Train Kappa", color="#22c55e")
    axes[1].plot(epochs, history["val_kappa"],   label="Val Kappa",   color="#f97316")
    axes[1].set_title("Quadratic Weighted Kappa", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("QWK Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
