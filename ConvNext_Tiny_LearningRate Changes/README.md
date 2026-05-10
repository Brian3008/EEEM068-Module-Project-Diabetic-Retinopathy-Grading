# Diabetic Retinopathy Grading using ConvNeXt-Tiny
## Experiment - Learning Rate Changes and other hypermeters tuning

## Overview
This project implements an end-to-end deep learning pipeline for automated diabetic retinopathy (DR) grading using retinal fundus images. The model is based on ConvNeXt-Tiny (224×224 pretrained on ImageNet-1K) and is fine-tuned to classify images into 5 DR severity levels.

The workflow includes data preprocessing, class imbalance handling, model training, evaluation, and interpretability using GradCAM.

## Experiment Changes
- The LR values and the batch sizes were mended to gain performance using WeightedRandomSampler for class imbalance
- LR were set as:
  LR_BACKBONE     = 1e-5 from 3e-6    
  LR_HEAD         = 3e-4 from 3e-5
- Batch Size was changed to 16 from 32

## Observations
- The Change in LR and BS values resultant in high qwk but lacked handling class imbalance.

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
weighted Random Sampler
ImageNet normalisation
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
