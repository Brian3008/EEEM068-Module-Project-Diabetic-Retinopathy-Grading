import timm
import torch
import torch.nn as nn

from .config import cfg


def build_model(pretrained: bool = True) -> nn.Module:
    if cfg.IMAGE_SIZE == 384:
        model_name = "swin_tiny_patch4_window12_384"
    elif cfg.IMAGE_SIZE == 224:
        model_name = "swin_tiny_patch4_window7_224"
    else:
        raise ValueError(
            f"Unsupported IMAGE_SIZE={cfg.IMAGE_SIZE}. Use 224 or 384."
        )

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=cfg.NUM_CLASSES,
        drop_rate=cfg.DROPOUT,
    )
    return model.to(cfg.DEVICE)


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
   
    head_param_ids = set()
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if classifier is not None:
            head_param_ids = {id(p) for p in classifier.parameters()}

    head_params, backbone_params = [], []
    for p in model.parameters():
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.LR_BACKBONE},
            {"params": head_params, "lr": cfg.LR_HEAD},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )


def get_scheduler(optimizer, steps_per_epoch: int):
    """Linear warmup → cosine decay, scheduled per-step."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, LinearLR, SequentialLR,
    )

    warmup_steps = max(1, cfg.WARMUP_EPOCHS * steps_per_epoch)
    cosine_steps = max(
        1, (cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS) * steps_per_epoch
    )

    warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=cfg.ETA_MIN
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )