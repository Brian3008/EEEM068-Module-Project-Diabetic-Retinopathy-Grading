import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

def evaluate(model, loader, device):
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            # QWK-friendly prediction (VERY IMPORTANT)
            expected = torch.sum(probs * torch.arange(5).to(device), dim=1)
            pred = torch.round(expected)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")

    return acc, qwk