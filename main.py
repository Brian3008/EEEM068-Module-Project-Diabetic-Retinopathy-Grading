import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_data_loaders
from src.train import train_one_epoch
from src.eval import evaluate

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data...")
    train_loader, val_loader = get_data_loaders()

    print("Loading model...")
    model, _ = get_model(num_classes=5)
    model = model.to(device)

    # FIXED LOSS (NO FOCAL LOSS)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_qwk = 0

    epochs = 15

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        acc, qwk = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {acc:.4f}")
        print(f"QWK: {qwk:.4f}")

        # save best model
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("Saved Best Model!")

    print("\nTraining Complete!")
    print("Best QWK:", best_qwk)


if __name__ == "__main__":
    main()