# train.py
# ─────────────────────────────────────────────────────────────────────────────
# MENTOR NOTE:
#   This version works with folder-based dataset (no CSV).
#   Key upgrades for your A4000 16GB setup:
#     ✅ WeightedRandomSampler  → handles class imbalance at data level
#     ✅ Mixed Precision (AMP)  → fp16, faster + less VRAM
#     ✅ Gradient Clipping      → stable training
#     ✅ QWK as primary metric  → clinical standard for DR grading
#     ✅ Early Stopping         → avoids overfitting
#     ✅ Per-epoch GPU memory   → monitor your A4000 usage
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    classification_report
)

from .config import cfg
from .model import build_model, get_optimizer, get_scheduler
from .dataset import get_dataloaders, compute_class_weights


# ─────────────────────────────────────────────────────────────────────────────
# METRIC: Quadratic Weighted Kappa
# ─────────────────────────────────────────────────────────────────────────────
# WHY QWK and not accuracy?
#   DR grading is ORDINAL — predicting "Severe" when truth is "Mild" is worse
#   than predicting "Moderate". QWK penalises bigger mistakes more heavily.
#   It's the standard metric in every Kaggle DR competition.
# ─────────────────────────────────────────────────────────────────────────────
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def probabilities_to_scores(probabilities: np.ndarray) -> np.ndarray:
    class_ids = np.arange(cfg.NUM_CLASSES, dtype=np.float32)
    return probabilities @ class_ids


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # np.digitize maps scores into 0..NUM_CLASSES-1 when bins has NUM_CLASSES-1 elements
    return np.digitize(scores, thresholds).astype(np.int64)


def optimize_qwk_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    initial_thresholds=None,
    iters: int = 2,
    grid_size: int = 80,
):
    """Coordinate-descent threshold tuning to maximize QWK on validation."""
    eps = 1e-6
    if initial_thresholds is None:
        thresholds = np.arange(0.5, cfg.NUM_CLASSES - 0.5, 1.0, dtype=np.float32)
    else:
        thresholds = np.array(initial_thresholds, dtype=np.float32).copy()

    best_preds = scores_to_labels(scores, thresholds)
    best_kappa = quadratic_weighted_kappa(y_true, best_preds)

    for _ in range(max(1, iters)):
        improved = False
        for i in range(cfg.NUM_CLASSES - 1):
            low = thresholds[i - 1] + eps if i > 0 else -0.5
            high = thresholds[i + 1] - eps if i < cfg.NUM_CLASSES - 2 else (cfg.NUM_CLASSES - 0.5)

            if high <= low:
                continue

            candidates = np.linspace(low, high, num=max(10, grid_size), dtype=np.float32)
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
    best_kappa = quadratic_weighted_kappa(y_true, best_preds)
    return thresholds, best_preds, best_kappa


# ─────────────────────────────────────────────────────────────────────────────
# GPU MEMORY MONITOR  (A4000 specific — helps you tune batch size)
# ─────────────────────────────────────────────────────────────────────────────
def print_gpu_memory():
    if cfg.DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   🖥️  GPU Memory → Allocated: {allocated:.2f}GB | "
              f"Reserved: {reserved:.2f}GB | Total: {total:.2f}GB")


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training if val_kappa doesn't improve for `patience` epochs.
    Saves best model automatically.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_kappa: float) -> bool:
        if self.best_score is None:
            self.best_score = val_kappa
        elif val_kappa < self.best_score + self.min_delta:
            self.counter += 1
            print(f"   ⏳ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_kappa
            self.counter    = 0
        return self.stop


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    for step, (images, labels) in enumerate(loader):
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        # ── Forward pass with AMP ─────────────────────────────────────────
        with autocast(device_type="cuda", enabled=cfg.USE_AMP):

            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # Gradient clipping — prevents exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # ── Accumulate metrics ────────────────────────────────────────────
        preds          = logits.argmax(dim=1)
        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ── Step-level log (every 20 steps) ──────────────────────────────
        if (step + 1) % 20 == 0:
            running_loss = total_loss / total_samples
            running_acc  = total_correct / total_samples
            print(f"   Step [{step+1:>3}/{len(loader)}] "
                  f"Loss: {running_loss:.4f} | "
                  f"Acc: {running_acc:.4f}")

    # ── Epoch-level metrics ───────────────────────────────────────────────
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, kappa, f1


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []
    all_probs     = []

    for images, labels in loader:
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):

            logits = model(images)
            loss   = criterion(logits, labels)

        preds          = logits.argmax(dim=1)
        probs          = torch.softmax(logits, dim=1)
        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, kappa, f1, all_labels, all_preds, np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_kappa,
    val_acc,
    class_names,
    qwk_thresholds=None,
    val_kappa_raw=None,
):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_kappa":       val_kappa,
        "val_kappa_raw":   val_kappa if val_kappa_raw is None else val_kappa_raw,
        "val_acc":         val_acc,
        "class_names":     class_names,
        # Keep this as plain Python list for torch.load(weights_only=True) compatibility.
        "qwk_thresholds":  None if qwk_thresholds is None else np.asarray(qwk_thresholds, dtype=np.float32).tolist(),
    }, cfg.BEST_MODEL)
    print(f"   💾 Checkpoint saved → {cfg.BEST_MODEL}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train():
    # ── Setup ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # ── GPU Info ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  🏥  Diabetic Retinopathy Grading — Swin Transformer Tiny")
    print("="*60)
    if cfg.DEVICE == "cuda":
        print(f"  🖥️  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  🔋 VRAM : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  No GPU found — training on CPU (will be slow!)")
    print(f"  📦 Batch size : {cfg.BATCH_SIZE}")
    print(f"  🔁 Epochs     : {cfg.NUM_EPOCHS}")
    print(f"  ⚡ AMP        : {cfg.USE_AMP}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders()

    # ── Sanity check val batch ────────────────────────────────────────────────
    val_images, val_labels = next(iter(val_loader))
    print(f"\n  🔍 Val batch check:")
    print(f"     Image shape : {val_images.shape}")
    print(f"     Label dist  : {torch.bincount(val_labels, minlength=cfg.NUM_CLASSES).tolist()}")
    print(f"     Pixel mean  : {val_images.mean():.4f}  ← should be near 0.0")
    print(f"     Pixel std   : {val_images.std():.4f}   ← should be near 1.0")

    # ── Loss with class weights ────────────────────────────────────────────
    class_weights = compute_class_weights(class_names).to(cfg.DEVICE)

    print(f"\n  ⚖️  Class weights:")
    for name, w in zip(class_names, class_weights.cpu().numpy()):
        print(f"     {name:15s}: {w:.4f}")

    criterion = nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = cfg.LABEL_SMOOTHING
    )

    # ── Model, Optimizer, Scheduler ───────────────────────────────────────
    model         = build_model(pretrained=True)
    optimizer     = get_optimizer(model)
    scheduler     = get_scheduler(optimizer)
    scaler        = GradScaler("cuda", enabled=cfg.USE_AMP)
    early_stopper = EarlyStopping(patience=7)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  🤖 Model       : Swin Transformer Tiny")
    print(f"  📐 Total params: {total_params/1e6:.1f}M")
    print(f"  🎯 Trainable   : {trainable_params/1e6:.1f}M")

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss"  : [],
        "val_loss"    : [],
        "train_kappa" : [],
        "val_kappa"   : [],
        "train_acc"   : [],
        "val_acc"     : [],
        "train_f1"    : [],
        "val_f1"      : [],
        "val_kappa_tuned": [],
        "lr"          : [],
    }

    best_kappa = -1.0
    best_thresholds = np.arange(0.5, cfg.NUM_CLASSES - 0.5, 1.0, dtype=np.float32)
    print("\n" + "="*60)
    print("  🚀 Starting Training...")
    print("="*60)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[1]["lr"]  # head LR

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch:02d}/{cfg.NUM_EPOCHS}  |  LR (head): {current_lr:.2e}")
        print(f"{'─'*60}")

        # ── Train ─────────────────────────────────────────────────────────
        print("  [TRAIN]")
        tr_loss, tr_acc, tr_kappa, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )

        # ── Validate ──────────────────────────────────────────────────────
        print("  [VALID]")
        val_loss, val_acc, val_kappa, val_f1, y_true, y_pred, y_prob = validate(
            model, val_loader, criterion
        )

        y_true_np = np.asarray(y_true, dtype=np.int64)
        tuned_kappa = val_kappa
        tuned_acc = val_acc
        tuned_f1 = val_f1
        tuned_preds = np.asarray(y_pred, dtype=np.int64)

        if cfg.TUNE_QWK_THRESHOLDS:
            val_scores = probabilities_to_scores(y_prob)
            best_thresholds, tuned_preds, tuned_kappa = optimize_qwk_thresholds(
                y_true=y_true_np,
                scores=val_scores,
                initial_thresholds=best_thresholds,
                iters=cfg.QWK_THRESHOLD_ITERS,
                grid_size=cfg.QWK_THRESHOLD_GRID_SIZE,
            )
            tuned_acc = float((tuned_preds == y_true_np).mean())
            tuned_f1 = f1_score(y_true_np, tuned_preds, average="macro", zero_division=0)

        # ── Scheduler step ────────────────────────────────────────────────
        scheduler.step()

        # ── Epoch Summary ─────────────────────────────────────────────────
        elapsed = time.time() - t0
        print(f"\n  ⏱️  Time     : {elapsed:.0f}s")
        print(f"  📈 Train    → Loss: {tr_loss:.4f} | "
              f"Acc: {tr_acc:.4f} | Kappa: {tr_kappa:.4f} | F1: {tr_f1:.4f}")
        print(f"  📉 Val      → Loss: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f} | F1: {val_f1:.4f}")
        if cfg.TUNE_QWK_THRESHOLDS:
            print(f"  🎯 Val Tuned→ Acc: {tuned_acc:.4f} | Kappa: {tuned_kappa:.4f} | F1: {tuned_f1:.4f}")
            print(f"  🧭 Thresholds: {np.round(best_thresholds, 3).tolist()}")

        # GPU memory report
        print_gpu_memory()

        # ── Save Best Model ───────────────────────────────────────────────
        metric_for_selection = tuned_kappa if cfg.TUNE_QWK_THRESHOLDS else val_kappa
        if metric_for_selection > best_kappa:
            best_kappa = metric_for_selection
            save_checkpoint(model, optimizer, epoch,
                            metric_for_selection, tuned_acc, class_names,
                            qwk_thresholds=best_thresholds,
                            val_kappa_raw=val_kappa)
            print(f"   🏆 New best Kappa: {best_kappa:.4f}")

        # ── Per-class report every 5 epochs ──────────────────────────────
        if epoch % 5 == 0:
            print(f"\n  📋 Per-Class Report (Epoch {epoch}):")
            print(classification_report(
                y_true_np, tuned_preds,
                target_names = class_names,
                zero_division = 0
            ))

        # ── Log history ───────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_kappa"].append(tr_kappa)
        history["val_kappa"].append(val_kappa)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)
        history["val_kappa_tuned"].append(tuned_kappa)
        history["lr"].append(current_lr)

        # ── Early Stopping ────────────────────────────────────────────────
        if early_stopper(metric_for_selection):
            print(f"\n  🛑 Early stopping triggered at epoch {epoch}")
            break

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅  Training Complete!")
    print(f"  🏆 Best Validation Kappa : {best_kappa:.4f}")
    print(f"  💾 Best model saved at   : {cfg.BEST_MODEL}")
    print("="*60)

    return history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = train()