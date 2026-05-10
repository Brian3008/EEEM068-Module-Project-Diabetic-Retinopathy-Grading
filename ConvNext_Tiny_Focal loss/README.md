# Diabetic Retinopathy Grading using ConvNeXt-Tiny
## Experiment - Focal Loss addition

## Overview
This project implements an end-to-end deep learning pipeline for automated diabetic retinopathy (DR) grading using retinal fundus images. The model is based on ConvNeXt-Tiny (224×224 pretrained on ImageNet-1K) and is fine-tuned to classify images into 5 DR severity levels.

The workflow includes data preprocessing, class imbalance handling, model training, evaluation, and interpretability using GradCAM.

## Experiment Changes
- Addition of Focal Loss for class imbalance handling instead of WeightedRandomSampler

## Observations
- Focal loss does not perform better than weighted Random Sampler in this experiment setting though it increases the qwk and metric scores it dails in handling the class imbalance which can be observed in confusion matrix


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
Train-validation split (85% / 15%)
Stratified Split
Data augmentation
ImageNet normalisation
Focal Loss
criterion
AdamW optimiser
Layer-wise learning rates
Mixed precision training (AMP)
Gradient accumulation
Gradient clipping
Warmup + cosine annealing scheduler
Early stopping
Transfer learning (ConvNeXt-Tiny)
GradCAM interpretability

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
