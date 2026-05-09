\# BEiT Run 8 — Advanced Architecture Experiment

Student: Nyi Nyi Myo Zin | URN: 6955918



\## Key Changes

\- 3-group layer-wise LR decay

\- Deeper head: LayerNorm + Linear + GELU + Linear

\- Stronger dropout 0.4

\- Stronger weight decay 0.05

\- Longer warmup 5 epochs

\- Stratified split



\## Results

| Metric | Value |

|---|---|

| QWK | 0.5318 |

| Accuracy | 61.11% |

| Macro F1 | 0.4265 |

| Mean AUC | 0.7922 |



\## Conclusion

Over-regularisation caused lower performance than Run 3.

Deeper head added too much complexity.

Run 3 remains the best overall result (QWK 0.58).

