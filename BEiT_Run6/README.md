\# BEiT Run 6 — Perfect Balanced Sampler Experiment

Student: Nyi Nyi Myo Zin | URN: 6955918

Model: BEiT-base-patch16-224

Module: EEEM068 Applied Machine Learning



\## Purpose

Experiment using a perfectly balanced sampler where every

class is sampled equally regardless of original class size,

to force the model to learn all 5 DR grades equally.



\## Key Settings

\- Dataset: Colored images (224x224 folder-based)

\- Epochs: 25

\- LR Backbone: 2e-5

\- LR Head: 2e-4 (reduced from previous runs)

\- Sampler: Pure inverse frequency — equal samples per class

\- num\_samples: minority\_count x 5 x num\_classes = 17,700

\- Loss: CrossEntropy + pure inverse class weights

\- Label smoothing: 0.1

\- Scheduler: LinearLR warmup 3 epochs + CosineAnnealingLR

\- Batch size: 32



\## Key Changes from Run 5

\- Perfectly balanced sampler — each class seen equally

\- Reduced num\_samples to 17,700 (5x minority class)

\- Pure inverse class weights (most aggressive)

\- Lower LR for stability



\## Results

| Metric | Value |

|---|---|

| QWK | 0.27 |

| Accuracy | 9% |

| Macro F1 | 0.14 |



\## Per-Class Recall

| Class | Recall |

|---|---|

| No DR | 0.00 |

| Mild | 0.85 |

| Moderate | 0.55 |

| Severe | 0.79 |

| Proliferative | 0.57 |



\## Key Observation

Worst overall result of all runs. The perfectly balanced

sampler completely destroyed No DR detection — recall

collapsed to 0.00 meaning the model never predicted No DR

despite it being 73.5% of real cases. By showing No DR

so rarely during training (equal to only 708 Proliferative

images per epoch), the model forgot No DR exists entirely.

This demonstrates that perfect class balance is harmful

for severely imbalanced datasets — the natural class

distribution carries important information that the model

needs to learn. Sqrt weighting (Run 3) proved to be the

optimal balance between minority class boosting and

majority class preservation.

