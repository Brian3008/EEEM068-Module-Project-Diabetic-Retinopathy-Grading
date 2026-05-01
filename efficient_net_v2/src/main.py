import torch

from.dataset import get_dataloaders
from.model import create_efficientnetv2
from.train_utils import (
    train_model,
    plot_training_curves,
    plot_confusion_matrices,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, class_names = get_dataloaders()
    print("Classes:", class_names)

    model_name = "efficientnetv2_s"
    print(f"Using model: {model_name}")

    model = create_efficientnetv2(model_name=model_name).to(device)

    (
        train_losses,
        val_losses,
        val_qwk_scores,
        best_model_path,
        all_labels,
        all_preds,
    ) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    print(f"Best model saved to: {best_model_path}")

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_qwk_scores=val_qwk_scores,
    )

    plot_confusion_matrices(
        all_labels=all_labels,
        all_preds=all_preds,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
