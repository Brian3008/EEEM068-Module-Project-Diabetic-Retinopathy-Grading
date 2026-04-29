import timm
import torch.nn as nn
from.config import cfg


def create_efficientnetv2(model_name: str = "efficientnetv2_s"):
    num_classes = cfg.NUM_CLASSES

    model = timm.create_model(model_name, pretrained=False)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model
