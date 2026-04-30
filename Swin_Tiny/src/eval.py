import numpy as np
import torch
from sklearn.metrics import (
    classification_report, cohen_kappa_score, f1_score, roc_auc_score,
)
from torch.amp import autocast

from .config import cfg
from .dataset import get_dataloaders
from .model import build_model
from .plots import (
    plot_confusion_matrix, plot_training_curves,
)


def probabilities_to_scores(probabilities: np.ndarray) -> np.ndarray:
    class_ids = np.arange(cfg.NUM_CLASSES, dtype=np.float32)
    return probabilities @ class_ids


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, thresholds).astype(np.int64)


def load_best_model():
    model = build_model(pretrained=False)
    try:
        ckpt = torch.load(cfg.BEST_MODEL, map_location=cfg.DEVICE)
    except Exception as e:
        if "Weights only load failed" in str(e):
            ckpt = torch.load(
                cfg.BEST_MODEL, map_location=cfg.DEVICE, weights_only=False
            )
        else:
            raise
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  |  "
          f"val_kappa={ckpt['val_kappa']:.4f}")
    return model, ckpt


@torch.no_grad()
def get_predictions(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(cfg.DEVICE)
        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def full_evaluation(history=None):
    _, val_loader, class_names, _ = get_dataloaders()
    model, ckpt = load_best_model()
    y_true, y_pred_argmax, y_prob = get_predictions(model, val_loader)

    thresholds = ckpt.get("qwk_thresholds", None)
    scores = probabilities_to_scores(y_prob)
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float32)
        y_pred = scores_to_labels(scores, thresholds)
        print(f"  Using tuned thresholds: "
              f"{np.round(thresholds, 3).tolist()}")
    else:
        y_pred = y_pred_argmax

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = float((y_true == y_pred).mean())
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print("\n" + "=" * 55)
    print("  DIABETIC RETINOPATHY GRADING — FINAL RESULTS")
    print("=" * 55)
    print(f"  Quadratic Weighted Kappa : {kappa:.4f}")
    print(f"  Macro F1 Score           : {f1:.4f}")
    print(f"  Accuracy                 : {acc:.4f}")
    print(f"  Macro AUC-ROC (OvR)      : {auc:.4f}")
    print("=" * 55)
    print("\n  Per-Class Report:")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))

    plot_confusion_matrix(y_true, y_pred, class_names=class_names)
    
    if history:
        plot_training_curves(history)


if __name__ == "__main__":
    full_evaluation()