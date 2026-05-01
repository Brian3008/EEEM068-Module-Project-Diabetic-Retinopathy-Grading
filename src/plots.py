import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrices(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_qwks, val_qwks, save_path):
    epochs = range(len(train_losses))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # QWK plot
    axes[1].plot(epochs, train_qwks, label="Train QWK")
    axes[1].plot(epochs, val_qwks, label="Val QWK")
    axes[1].set_title("QWK (Train vs Validation)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()