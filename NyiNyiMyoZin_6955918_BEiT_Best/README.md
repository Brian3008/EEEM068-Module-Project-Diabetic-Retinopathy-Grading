\# BEiT Run 9 — Final Best Combined Settings

Student: Nyi Nyi Myo Zin | URN: 6955918



\## Results

| Metric | Value |

|---|---|

| QWK | 0.5583 |

| Accuracy | 54.93% |

| Macro F1 | 0.4519 |

| Mean AUC | 0.8173 |



\## Per-class F1

| Class | F1 |

|---|---|

| No DR | 0.7000 |

| Mild | 0.1764 |

| Moderate | 0.4179 |

| Severe | 0.4351 |

| Proliferative | 0.5300 |



\## Key settings

\- Dataset: Processed Kaggle DR (Rudra's dataset)

\- Split: Stratified 85/15

\- Sampler: Sqrt weighted

\- Loss: CrossEntropy + sqrt weights + smoothing 0.1

\- LR backbone: 3e-6 / head: 3e-5

\- Warmup 3 epochs + cosine annealing

\- Early stopping patience 8

