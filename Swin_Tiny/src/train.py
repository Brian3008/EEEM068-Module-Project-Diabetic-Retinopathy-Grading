import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, cohen_kappa_score, f1_score,
)
from torch.amp import GradScaler, autocast

from .config import cfg
from .dataset import compute_class_weights, get_dataloaders
from .model import build_model, get_optimizer, get_scheduler


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def probabilities_to_scores(probabilities: np.ndarray) -> np.ndarray:
    class_ids = np.arange(cfg.NUM_CLASSES, dtype=np.float32)
    return probabilities @ class_ids


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, thresholds).astype(np.int64)


def optimize_qwk_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    initial_thresholds=None,
    iters: int = 3,
    grid_size: int = 100,
):
    eps = 1e-6
    if initial_thresholds is None:
        thresholds = np.arange(
            0.5, cfg.NUM_CLASSES - 0.5, 1.0, dtype=np.float32
        )
    else:
        thresholds = np.array(initial_thresholds, dtype=np.float32).copy()

    best_preds = scores_to_labels(scores, thresholds)
    best_kappa = quadratic_weighted_kappa(y_true, best_preds)

    for _ in range(max(1, iters)):
        improved = False
        for i in range(cfg.NUM_CLASSES - 1):
            low = thresholds[i - 1] + eps if i > 0 else -0.5
            high = (thresholds[i + 1] - eps
                    if i < cfg.NUM_CLASSES - 2
                    else (cfg.NUM_CLASSES - 0.5))
            if high <= low:
                continue
            candidates = np.linspace(low, high, num=grid_size, dtype=np.float32)
            local_best_t = thresholds[i]
            local_best_k = best_kappa
            for t in candidates:
                trial = thresholds.copy()
                trial[i] = t
                preds = scores_to_labels(scores, trial)
                kappa = quadratic_weighted_kappa(y_true, preds)
                if kappa > local_best_k:
                    local_best_k = kappa
                    local_best_t = t
            if local_best_k > best_kappa:
                thresholds[i] = local_best_t
                best_kappa = local_best_k
                improved = True
        if not improved:
            break

    best_preds = scores_to_labels(scores, thresholds)
    return thresholds, best_preds, best_kappa


def print_gpu_memory():
    if cfg.DEVICE == "cuda":
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: alloc={a:.2f}GB / reserved={r:.2f}GB / total={t:.2f}GB")


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels = [], []

    for step, (images, labels) in enumerate(loader):
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  

        preds = logits.argmax(dim=1)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 50 == 0:
            print(f"    step {step+1:>4}/{len(loader)}  "
                  f"loss={total_loss/total_samples:.4f}  "
                  f"acc={total_correct/total_samples:.4f}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, kappa, f1


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits.float(), dim=1)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return (avg_loss, accuracy, kappa, f1,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def save_checkpoint(model, optimizer, epoch, val_kappa, val_acc, class_names,
                    qwk_thresholds=None, val_kappa_raw=None):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_kappa": float(val_kappa),
        "val_kappa_raw": float(val_kappa
                                if val_kappa_raw is None else val_kappa_raw),
        "val_acc": float(val_acc),
        "class_names": class_names,
        "qwk_thresholds": (None if qwk_thresholds is None
                           else np.asarray(qwk_thresholds,
                                           dtype=np.float32).tolist()),
    }, cfg.BEST_MODEL)
    print(f"  Saved checkpoint → {cfg.BEST_MODEL}")


def train():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print("=" * 60)
    print("  Diabetic Retinopathy Grading — Swin-Tiny  (Kaggle DR-2015)")
    print("=" * 60)
    if cfg.DEVICE == "cuda":
        gp = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {gp.name}  ({gp.total_memory/1e9:.1f} GB)")
    else:
        print("  GPU         : NONE — running on CPU (very slow)")
    print(f"  image_size  : {cfg.IMAGE_SIZE}")
    print(f"  batch_size  : {cfg.BATCH_SIZE}")
    print(f"  epochs      : {cfg.NUM_EPOCHS}  (warmup={cfg.WARMUP_EPOCHS})")
    print(f"  AMP         : {cfg.USE_AMP}")
    print(f"  sampler     : {'weighted' if cfg.USE_WEIGHTED_SAMPLER else 'shuffle'}")
    print(f"  loss weights: {cfg.USE_LOSS_WEIGHTS}")

    if cfg.USE_WEIGHTED_SAMPLER and cfg.USE_LOSS_WEIGHTS:
        raise ValueError(
            "Don't use BOTH USE_WEIGHTED_SAMPLER and USE_LOSS_WEIGHTS — "
            "this double-balances and ruins the prediction distribution."
        )

    train_loader, val_loader, class_names, train_labels = get_dataloaders()

    val_imgs, val_lbls = next(iter(val_loader))
    print(f"\n  Val batch shape: {tuple(val_imgs.shape)}")
    print(f"  Val pixel mean/std: {val_imgs.mean():.3f} / {val_imgs.std():.3f}")

    if cfg.USE_LOSS_WEIGHTS:
        class_weights = compute_class_weights(train_labels).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING
        )
        print(f"\n  class weights: "
              f"{class_weights.cpu().numpy().round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    model = build_model(pretrained=True)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, steps_per_epoch=len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg.USE_AMP)
    early_stop = EarlyStopping(patience=8)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.1f}M")

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_kappa", "val_kappa", "val_kappa_tuned",
        "train_acc", "val_acc",
        "train_f1", "val_f1", "lr",
    ]}

    best_kappa = -1.0
    best_thresholds = np.arange(
        0.5, cfg.NUM_CLASSES - 0.5, 1.0, dtype=np.float32
    )

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch}/{cfg.NUM_EPOCHS}  |  lr(head)={current_lr:.2e}")

        print("  [TRAIN]")
        tr_loss, tr_acc, tr_kappa, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler
        )

        print("  [VAL]")
        val_loss, val_acc, val_kappa, val_f1, y_true, y_pred, y_prob = (
            validate(model, val_loader, criterion)
        )

        scores = probabilities_to_scores(y_prob)
        print("  Pred-score distribution by true class:")
        for c in range(cfg.NUM_CLASSES):
            mask = y_true == c
            if mask.sum() == 0:
                continue
            s = scores[mask]
            print(f"    {class_names[c]:18s}  "
                  f"mean={s.mean():.3f}  std={s.std():.3f}  "
                  f"n={int(mask.sum())}")

        tuned_kappa = val_kappa
        tuned_acc = val_acc
        tuned_f1 = val_f1
        tuned_preds = y_pred
        if cfg.TUNE_QWK_THRESHOLDS:
            best_thresholds, tuned_preds, tuned_kappa = optimize_qwk_thresholds(
                y_true=y_true, scores=scores,
                initial_thresholds=best_thresholds,
                iters=cfg.QWK_THRESHOLD_ITERS,
                grid_size=cfg.QWK_THRESHOLD_GRID_SIZE,
            )
            tuned_acc = float((tuned_preds == y_true).mean())
            tuned_f1 = f1_score(y_true, tuned_preds,
                                average="macro", zero_division=0)

        elapsed = time.time() - t0
        print(f"\n  time={elapsed:.0f}s")
        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"kappa={tr_kappa:.4f}  f1={tr_f1:.4f}")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}  "
              f"kappa={val_kappa:.4f}  f1={val_f1:.4f}")
        if cfg.TUNE_QWK_THRESHOLDS:
            print(f"  tuned  acc={tuned_acc:.4f}  kappa={tuned_kappa:.4f}  "
                  f"f1={tuned_f1:.4f}")
            print(f"  thresholds: {np.round(best_thresholds, 3).tolist()}")
        print_gpu_memory()

        metric = tuned_kappa if cfg.TUNE_QWK_THRESHOLDS else val_kappa
        if metric > best_kappa:
            best_kappa = metric
            save_checkpoint(model, optimizer, epoch, metric, tuned_acc,
                            class_names, qwk_thresholds=best_thresholds,
                            val_kappa_raw=val_kappa)
            print(f"  ★ new best kappa: {best_kappa:.4f}")

        if epoch % 5 == 0:
            print("\n  Per-class report:")
            print(classification_report(
                y_true, tuned_preds,
                target_names=class_names, zero_division=0
            ))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_kappa"].append(tr_kappa)
        history["val_kappa"].append(val_kappa)
        history["val_kappa_tuned"].append(tuned_kappa)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        if early_stop(metric):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print(f"  Done. Best Val Kappa: {best_kappa:.4f}")
    print(f"  Best model: {cfg.BEST_MODEL}")
    print("=" * 60)
    return history


if __name__ == "__main__":
    history = train()