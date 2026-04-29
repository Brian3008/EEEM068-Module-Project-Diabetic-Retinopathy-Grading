# Diabetic Retinopathy Grading - Baseline Model (V1)

## Overview

This project implements a baseline deep learning model for diabetic
retinopathy classification using retinal fundus images. The model
predicts one of five severity levels:

-   No DR
-   Mild
-   Moderate
-   Severe
-   Proliferative DR

The implementation uses a pretrained ConvNeXt-Tiny architecture and
evaluates performance using Accuracy and Quadratic Weighted Kappa (QWK).

------------------------------------------------------------------------

## Project Structure
```
ConvNext/ 
    ├── main.py 
    ├── requirements.txt 
    ├──README.md 
    ├── src/ 
    │ ├── dataset.py 
    │ ├── model.py 
    │ ├── train.py 
    │ ├── eval.py 
    │ └── plots.py 
    │ ├── data/ 
    │   ├── colored_images/ 
    │   └──trainLabels.csv 
    └── outputs/output_V1/ 
      ├── best_model.pth 
      └── plots/ 
        ├── confusion_matrix.png 
        └── training_curves.png
```
------------------------------------------------------------------------

## Installation

-  python3 -m venv env 
-  source env/bin/activate 
-  pip install -r requirements.txt

------------------------------------------------------------------------

## Running the Project

python3 main.py

------------------------------------------------------------------------

## Model Details

-   Architecture: ConvNeXt-Tiny (pretrained)
-   Input size: 224 x 224
-   Optimizer: Adam
-   Learning rate: 1e-4
-   Loss function: CrossEntropyLoss
-   Epochs: 15

------------------------------------------------------------------------

## Evaluation Metrics

-   Accuracy
-   Quadratic Weighted Kappa (QWK)

------------------------------------------------------------------------

## Outputs

-   Confusion Matrix: outputs/output_V1/plots/confusion_matrix.png
-   Training Curves: outputs/output_V1//plots/training_curves.png
-   Best Model: outputs/output_V1/best_model.pth

------------------------------------------------------------------------

## Notes

Baseline_model_training_V1
-   Baseline version without class balancing
-   Validation loss approximated
-   Used for future improvements
-   Best QWK: 0.6618

------------------------------------------------------------------------

## Author

-   https://github.com/gitnerd109
-   Name: Bilimagga Ganesh Suhas
-   University ID: 6970288
-   @ University Of Surrey, UK
