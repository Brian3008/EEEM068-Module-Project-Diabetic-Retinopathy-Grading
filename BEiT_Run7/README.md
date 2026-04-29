\# BEiT Run 7 — Diabetic Retinopathy Grading



Model: BEiT-base-patch16-224

Student: \[Your Name]

Module: EEEM068 Applied Machine Learning



\## Dataset

Processed dataset — 512x512 JPEG images with Ben Graham

preprocessing already applied. 35,126 images total.



\## Results — Best Class Balance Across All Runs



| Metric | Value |

|---|---|

| Accuracy | 54% |

| QWK | 0.5618 |

| Macro F1 | 0.45 |



\## Per-class recall — best balanced result



| Class | Recall |

|---|---|

| No DR | 0.57 |

| Mild | 0.41 |

| Moderate | 0.40 |

| Severe | 0.68 |

| Proliferative | 0.67 |



\## Key improvement over Run 3

Run 7 uses the processed dataset with higher quality images.

Mild recall improved from 0.19 to 0.41.

Moderate recall improved from 0.28 to 0.40.

Severe recall improved from 0.48 to 0.68.



\## How to run

pip install -r requirements.txt

python main.py



\## Class imbalance handling

1\. Sqrt inverse frequency WeightedRandomSampler

2\. Soft class weights in CrossEntropy loss

3\. Label smoothing 0.1

4\. Layer-wise learning rate decay

5\. LR warmup 3 epochs then cosine decay

