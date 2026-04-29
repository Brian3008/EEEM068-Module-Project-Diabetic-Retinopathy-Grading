import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def get_model(num_classes=5):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model