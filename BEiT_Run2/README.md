\# BEiT Run 2 — Sqrt Sampler + Layer-wise LR

Student: Nyi Nyi Myo Zin | URN: 6955918

Model: BEiT-base-patch16-224

Module: EEEM068 Applied Machine Learning



\## Purpose

First attempt at fixing class imbalance and overfitting

from Run 1 by introducing weighted sampling, class weights

in loss, and layer-wise learning rate decay.



\## Key Settings

\- Dataset: Colored images (224x224 folder-based)

\- Epochs: 30

\- LR Backbone: 3e-6 (10x lower than head)

\- LR Head: 3e-5

\- Sampler: Sqrt inverse frequency WeightedRandomSampler

\- Loss: CrossEntropy + sqrt class weights + label smoothing 0.1

\- Scheduler: LinearLR warmup 3 epochs + CosineAnnealingLR

\- Batch size: 32



\## Key Changes from Run 1

\- Added sqrt weighted sampler — minority classes seen more often

\- Added class weights in CrossEntropy loss

\- Added layer-wise LR — backbone 10x slower than head

\- Added warmup scheduler — stabilises early training



\## Results

| Metric | Value |

|---|---|

| QWK | 0.50 |

| Accuracy | 50% |

| Macro F1 | 0.41 |



\## Per-Class Recall

| Class | Recall |

|---|---|

| No DR | 0.51 |

| Mild | 0.45 |

| Moderate | 0.43 |

| Severe | 0.61 |

| Proliferative | 0.67 |



\## Key Observation

Significant improvement over Run 1. Overfitting was resolved

with train and val loss now converging together. Best class

balance across all runs — Severe recall 0.61 and Proliferative

recall 0.67 are the highest achieved. QWK improved from

0.34 to 0.50. The most balanced confusion matrix of all runs.

