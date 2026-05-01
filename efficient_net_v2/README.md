# EfficientNetV2 – Diabetic Retinopathy Grading

This subproject implements diabetic retinopathy grading using **EfficientNetV2** on the 2015 Kaggle Diabetic Retinopathy dataset.

It is part of the team repository:
`EEEM068-Module-Project-Diabetic-Retinopathy-Grading`.

---

## Project Structure

```text
efficient_net_v2/
├── organize_images.py        # Script to organize flat images into class folders using trainLabels.csv
├── processed.zip             # (local only) Preprocessed images (flat) – not tracked in git
├── trainLabels.csv           # Kaggle labels: image, level (0–4)
├── outputs/
│   └── output_V1/
│       ├── best_model.pth    # Best model (by validation QWK)
│       └── plots/
│           ├── training_curves.png
│           └── confusion_matrix.png
└── src/
    ├── config.py             # Paths, hyperparameters, output dirs
    ├── dataset.py            # DRFolderDataset, CLAHE, transforms, dataloaders
    ├── main.py               # Training entry point (uses EfficientNetV2)
    ├── model.py              # EfficientNetV2 model definition
    ├── train_utils.py        # Training loop, QWK, plots, confusion matrix
    └── data/
        └── colored_images/   # Organized images in class folders (not tracked in git)
            ├── No_DR/
            ├── Mild/
            ├── Moderate/
            ├── Severe/
            └── Proliferate_DR/
