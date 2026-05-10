# Diabetic Retinopathy Classification using ConvNeXt

## 1. Overview

This project implements a deep learning pipeline for automated classification of diabetic retinopathy severity from retinal fundus images. A pretrained ConvNeXt-Tiny model is fine-tuned to classify images into five disease stages.

---

## 2. Problem Definition

Diabetic retinopathy is a leading cause of blindness. Early detection through automated systems can assist clinicians in large-scale screening.

This task is formulated as a multi-class classification problem with ordered labels.

---

## 3. Dataset

Dataset used:
Diabetic Retinopathy 2015 (Colored and Resized)

The dataset contains retinal images categorized into the following classes:

* 0: No DR
* 1: Mild
* 2: Moderate
* 3: Severe
* 4: Proliferative DR

### Download Instructions

```bash
kaggle datasets download -d sovitrath/diabetic-retinopathy-2015-data-colored-resized
unzip diabetic-retinopathy-2015-data-colored-resized.zip
```

### Directory Structure

Place the dataset in the following path:

```
data/colored_images/colored_images/
```

---

## 4. Project Structure

```
.
├── main.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
├── data/
├── outputs/
├── README.md
└── requirements.txt
```

---

## 5. Model Architecture

* Backbone: ConvNeXt-Tiny (pretrained on ImageNet)
* Final layer modified for 5-class classification
* Transfer learning applied for feature extraction and fine-tuning

---

## 6. Training Configuration

* Loss Function: CrossEntropyLoss with class weights
* Optimizer: Adam
* Learning Rate: reduced for stable convergence
* Epochs: 15–20
* Batch Size: 32

---

## 7. Evaluation Metrics

The model is evaluated using:

* Accuracy
* Quadratic Weighted Kappa (QWK)

QWK is used as the primary metric due to the ordinal nature of the classification task.

---

## 8. Results

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | 0.77  |
| QWK Score           | 0.75  |

The model demonstrates strong ordinal agreement with ground truth labels.

---

## 9. Running the Project

### Environment Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Training

```bash
python3 main.py
```

---

## 11. Key Features

* Modular code structure
* Transfer learning using ConvNeXt
* Class imbalance handling
* Validation-based checkpointing
* Use of ordinal-aware evaluation metric (QWK)

---

## 12. Notes

* The best model is selected based on validation QWK
* The dataset is not included in the repository due to size constraints
* Ensure correct directory structure before running the code

---

## 13. Author

Name: Bilimagga Ganesh Suhas