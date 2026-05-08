\# BEiT Run 4 — Focal Loss Experiment

Student: Nyi Nyi Myo Zin | URN: 6955918

Model: BEiT-base-patch16-224

Module: EEEM068 Applied Machine Learning



\## Purpose

Experiment replacing CrossEntropy with Focal Loss to

specifically target hard-to-classify minority classes

like Mild DR which looks visually similar to No DR.



\## Key Settings

\- Dataset: Colored images (224x224 folder-based)

\- Epochs: 20

\- LR Backbone: 3e-6

\- LR Head: 3e-5

\- Sampler: Cube-root inverse frequency sampler

\- Loss: Focal Loss (gamma=2.0) + sqrt class weights

\- Scheduler: LinearLR warmup 3 epochs + CosineAnnealingLR

\- Batch size: 32

\- Dropout: 0.3



\## Key Changes from Run 3

\- Replaced CrossEntropy with Focal Loss (gamma=2.0)

\- Focal Loss down-weights easy correct predictions

\- Changed sampler from sqrt to cube-root (gentler)

\- Reduced epochs from 25 to 20



\## Results

| Metric | Value |

|---|---|

| QWK | 0.42 |

| Accuracy | 28% |

| Macro F1 | 0.36 |



\## Per-Class Recall

| Class | Recall |

|---|---|

| No DR | 0.22 |

| Mild | 0.81 |

| Moderate | 0.29 |

| Severe | 0.51 |

| Proliferative | 0.47 |



\## Key Observation

Focal Loss caused over-correction — Mild recall jumped to

0.81 (best across all runs) but overall accuracy collapsed

to 28% and No DR recall dropped to 0.22. The model became

obsessed with predicting minority classes at the expense of

the majority class. QWK dropped from Run 3's 0.58 to 0.42.

Conclusion: Focal Loss alone without careful tuning

destabilises training on severely imbalanced medical datasets.

