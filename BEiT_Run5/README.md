\# BEiT Run 5 — MixUp Augmentation Experiment

Student: Nyi Nyi Myo Zin | URN: 6955918

Model: BEiT-base-patch16-224

Module: EEEM068 Applied Machine Learning



\## Purpose

Experiment using MixUp augmentation to create smoother

decision boundaries between adjacent DR grades, particularly

between No DR and Mild which look visually similar.



\## Key Settings

\- Dataset: Colored images (224x224 folder-based)

\- Epochs: 25

\- LR Backbone: 3e-6

\- LR Head: 3e-5

\- Sampler: Power 0.7 inverse frequency sampler

\- Loss: KLDivLoss (for soft MixUp labels) during training

\- Loss: CrossEntropy during validation

\- MixUp alpha: 0.3

\- Scheduler: LinearLR warmup 2 epochs + CosineAnnealingLR

\- Batch size: 32



\## Key Changes from Run 4

\- Added MixUp augmentation (alpha=0.3)

\- Switched to KLDivLoss to handle soft mixed labels

\- Changed sampler power from cube-root to 0.7

\- Removed Focal Loss — back to CrossEntropy for val



\## Results

| Metric | Value |

|---|---|

| QWK | 0.57 |

| Accuracy | 77% |

| Macro F1 | 0.45 |



\## Per-Class Recall

| Class | Recall |

|---|---|

| No DR | 0.97 |

| Mild | 0.00 |

| Moderate | 0.28 |

| Severe | 0.39 |

| Proliferative | 0.46 |



\## Key Observation

MixUp caused Mild recall to collapse completely to 0.00 —

the model could not detect Mild DR at all. While accuracy

jumped to 77% and No DR recall reached 0.97, the model

essentially ignored Mild entirely. MixUp blended Mild images

with No DR images during training making the model unable

to distinguish them. Despite high accuracy, this result

is clinically unacceptable as Mild DR would go undetected.

