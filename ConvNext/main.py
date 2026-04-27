import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.dataset import get_data_loaders
from src.model import get_model
from src.train import train_one_epoch
from src.eval import evaluate
from src.plots import plot_confusion_matrices, plot_training_curves


def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders()

    # Load model
    print("Loading model...")
    model = get_model(num_classes=5).to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15

    train_losses = []
    val_losses = []
    train_qwks = []
    val_qwks = []

    best_qwk = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_qwk = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_qwk, y_true, y_pred = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_qwks.append(train_qwk)
        val_qwks.append(val_qwk)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Train QWK: {train_qwk:.4f}")
        print(f"Val QWK: {val_qwk:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), "outputs/output_V1/best_model.pth")

    print("\nTraining Complete!")
    print(f"Best QWK: {best_qwk:.4f}")

    # Final evaluation + plots
    os.makedirs("outputs/output_V1/plots/", exist_ok=True)

    val_loss, val_acc, val_qwk, y_true, y_pred = evaluate(
    model, val_loader, criterion, device
)

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    plot_confusion_matrices(
        y_true, y_pred,
        class_names,
        "outputs/output_V1/plots/confusion_matrix.png"
    )

    plot_training_curves(
    train_losses,
    val_losses,
    train_qwks,
    val_qwks,
    "outputs/output_V1/plots/training_curves.png"
)

    print("Plots saved in outputs/output_V1/plots/")


if __name__ == "__main__":
    main()