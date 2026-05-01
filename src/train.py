import torch
from sklearn.metrics import cohen_kappa_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0
    all_preds = []
    all_labels = []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    train_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    return avg_loss, train_qwk