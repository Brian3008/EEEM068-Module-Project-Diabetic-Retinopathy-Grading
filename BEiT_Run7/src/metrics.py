import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    f1_score,
)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(y_true, y_pred) -> dict:
    qwk          = quadratic_weighted_kappa(y_true, y_pred)
    f1_macro     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None,     zero_division=0)
    report       = classification_report(
        y_true, y_pred,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        zero_division=0
    )
    return {
        "qwk":          qwk,
        "f1_macro":     f1_macro,
        "f1_per_class": f1_per_class.tolist(),
        "report":       report,
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    cm          = confusion_matrix(y_true, y_pred)
    cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(train_losses, val_losses, val_qwks,
                         save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(val_losses,   label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    axes[1].plot(val_qwks, color="green", label="Val QWK")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK")
    axes[1].set_title("Validation Quadratic Weighted Kappa")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()