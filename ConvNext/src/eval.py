import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score

def evaluate(model, loader, criterion, device):
    model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    return avg_loss, acc, qwk, all_labels, all_preds