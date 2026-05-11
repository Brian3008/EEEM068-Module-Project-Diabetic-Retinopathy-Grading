\# BEiT — Diabetic Retinopathy Grading



Model: BEiT-base-patch16-224 (Masked Image Modelling pretraining)  

Student: Nyi Nyi Myo Zin 

Module: EEEM068 Applied Machine Learning  



\## Results (Best Run — Run 3)



| Metric | Value |

|---|---|

| Accuracy | 70.79% |

| QWK | 0.5801 |

| Macro F1 | 0.4751 |



\## Training runs



| Run | Key change | QWK |

|---|---|---|

| Run 1 | Baseline | 0.34 |

| Run 2 | Sqrt sampler + layer-wise LR | 0.50 |

| Run 3 | Stronger augmentation — best | 0.58 |

| Run 4 | Focal loss experiment | 0.42 |



\## How to run



```bash

pip install -r requirements.txt

python main.py

```



\## Project structure

