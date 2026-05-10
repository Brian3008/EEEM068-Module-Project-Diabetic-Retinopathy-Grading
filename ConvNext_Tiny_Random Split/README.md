# Diabetic Retinopathy Grading using ConvNeXt-Tiny
## Experiment started with Random Split of Data for Understanding

## Overview
This project implements an end-to-end deep learning pipeline for automated diabetic retinopathy (DR) grading using retinal fundus images. The model is based on ConvNeXt-Tiny (224×224 pretrained on ImageNet-1K) and is fine-tuned to classify images into 5 DR severity levels.

The workflow includes data preprocessing, class imbalance handling, model training, evaluation, and interpretability using GradCAM.

## Dataset
- Source: Kaggle Diabetic Retinopathy Dataset  
- Total Images: ~35,126  
- Classes:
  0: No DR  
  1: Mild  
  2: Moderate  
  3: Severe  
  4: Proliferative DR  

Images: data/processed/  
Labels: data/trainLabels.csv  

## Model
- ConvNeXt-Tiny (HuggingFace pretrained)
- Modified classification head (5 classes)
- ~28M parameters

## Key Techniques
- Random split
- WeightedRandomSampler
- Data augmentation (flip, rotation, color jitter)
- Label smoothing
- AdamW optimizer
- Mixed precision training (AMP)
- Warmup + cosine scheduler

## Evaluation Metrics
- QWK (primary metric)
- Accuracy
- Macro F1-score
- ROC-AUC
- Confusion matrix

## Explainability
- GradCAM visualisations to highlight retinal regions influencing predictions

## Outputs
Saved in outputs_final/:
- training curves
- confusion matrix
- ROC curves
- per-class metrics
- GradCAM heatmaps

## Author
Bilimagga Ganesh Suhas  
URN: 6970288  
Model: ConvNeXt-Tiny
