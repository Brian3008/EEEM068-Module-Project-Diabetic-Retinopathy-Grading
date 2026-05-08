\# BEiT Run 1 — Baseline (No Imbalance Handling)

Student: Nyi Nyi Myo Zin | URN: 6955918

Model: BEiT-base-patch16-224

Module: EEEM068 Applied Machine Learning



\## Purpose

Baseline run with no class imbalance handling to establish

a performance reference point before applying any corrections.



\## Key Settings

\- Dataset: Colored images (224x224 folder-based)

\- Epochs: 25

\- LR: 2e-5 (single LR for all layers)

\- Sampler: None — standard random sampling

\- Loss: CrossEntropy with NO class weights

\- Scheduler: CosineAnnealingLR only (no warmup)

\- Batch size: 16



\## What Was Wrong

\- No weighted sampler — model saw mostly No DR images every batch

\- No class weights in loss — all classes penalised equally

\- No layer-wise LR decay — backbone trained too fast

\- No warmup — unstable early training



\## Results

| Metric | Value |

|---|---|

| QWK | 0.34 |

| Accuracy | 35% |

| Macro F1 | 0.31 |



\## Per-Class Recall

| Class | Recall |

|---|---|

| No DR | 0.26 |

| Mild | 0.39 |

| Moderate | 0.72 |

| Severe | 0.44 |

| Proliferative | 0.54 |



\## Key Observation

Severe overfitting — train loss dropped to 0.4 but val loss

stayed at 2.5. Model learned training data but failed to

generalise. No DR recall was only 0.26 showing the model

was barely predicting the majority class correctly despite

it being 73.5% of the data. This confirmed that class

imbalance handling was essential.

