import os
import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """Reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, optimizer, epoch, qwk, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "qwk":                  qwk,
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["qwk"]


def get_label_counts(csv_path: str) -> list:
    import pandas as pd
    df     = pd.read_csv(csv_path)
    counts = df["level"].value_counts().sort_index()
    return counts.tolist()