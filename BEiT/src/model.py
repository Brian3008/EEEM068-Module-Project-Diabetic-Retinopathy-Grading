import torch
import torch.nn as nn
from transformers import BeitForImageClassification


def build_beit_model(num_classes: int = 5,
                     pretrained_name: str = "microsoft/beit-base-patch16-224",
                     dropout: float = 0.3) -> nn.Module:
    """
    Loads pretrained BEiT-base and replaces the classifier head
    for 5-class DR grading.
    BEiT uses masked image modelling pretraining (like BERT for images),
    making it strong for fine-grained medical image features.
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
    Computes class weights for loss function.
    soft=False: pure inverse frequency (aggressive)
    soft=True:  square-root inverse frequency (gentler, avoids
                over-correcting the heavy No_DR class dominance)
    """
    counts = torch.tensor(label_counts, dtype=torch.float)
    if soft:
        weights = 1.0 / torch.sqrt(counts)
    else:
        weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return weights.to(device)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Focuses training on hard misclassified examples (like Mild DR)
    by down-weighting easy correct predictions (like obvious No DR).
    gamma=2.0 is standard for medical imaging tasks.
    Combined with class weights for double imbalance correction.
    """
    def __init__(self, weight=None, gamma: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt     = torch.exp(-ce_loss)
        focal  = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()