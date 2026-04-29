import torch
import torch.nn as nn
from transformers import BeitForImageClassification


def build_beit_model(num_classes: int = 5,
                     pretrained_name: str = "microsoft/beit-base-patch16-224",
                     dropout: float = 0.3) -> nn.Module:
    """
    Loads pretrained BEiT-base-224 and replaces classifier head
    for 5-class DR grading.
    BEiT uses masked image modelling pretraining — strong for
    fine-grained medical image features.
    """
    model = BeitForImageClassification.from_pretrained(
        pretrained_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_class_weights(label_counts: list, device: torch.device,
                      soft: bool = False) -> torch.Tensor:
    """
    soft=False: pure inverse frequency (aggressive)
    soft=True:  sqrt inverse frequency (gentler)
    """
    counts = torch.tensor(label_counts, dtype=torch.float)
    if soft:
        weights = 1.0 / torch.sqrt(counts)
    else:
        weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return weights.to(device)