# model.py
# ─────────────────────────────────────────────────────────────────────────────
# MENTOR NOTE — Why Swin Transformer?
#   CNNs (ResNet, EfficientNet) use local convolutions. They miss long-range
#   dependencies (e.g., haemorrhages spread across the whole retina).
#   Swin-T uses shifted window self-attention:
#     • Divides image into 7×7 windows → local attention (efficient)
#     • Shifts windows each layer → cross-window communication
#     • Hierarchical feature maps → works like a CNN backbone
#   Result: 28M params, faster than ViT, beats ResNet-50 on ImageNet.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from .config import cfg


class SwinTDRGrader(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()

        # ── Load pretrained Swin-T ────────────────────────────────────────
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)

        # ── Replace the classifier head ───────────────────────────────────
        # Original head: Linear(768 → 1000) for ImageNet
        # New head: we add dropout + Linear(768 → num_classes)
        in_features = backbone.head.in_features   # 768 for Swin-T

        self.backbone = backbone
        self.backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(pretrained: bool = True) -> SwinTDRGrader:
    model = SwinTDRGrader(
        num_classes = cfg.NUM_CLASSES,
        dropout     = cfg.DROPOUT,
        pretrained  = pretrained,
    )
    return model.to(cfg.DEVICE)


def get_optimizer(model: SwinTDRGrader) -> torch.optim.Optimizer:
    """
    MENTOR NOTE — Differential learning rates:
    Pretrained backbone layers already have good weights → small LR.
    New classifier head starts random → larger LR.
    This is called 'discriminative fine-tuning'.
    """
    head_params     = list(model.backbone.head.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_param_ids]

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params,     "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.T_MAX, eta_min=cfg.ETA_MIN
    )