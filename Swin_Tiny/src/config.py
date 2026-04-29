import torch

class Config:
    # ── Dataset ──────────────────────────────────────────────────────────
    DATA_DIR    = "src/data/colored_images"
    NUM_CLASSES = 5
    IMAGE_SIZE  = 224

    # IMPORTANT:
    # QWK assumes labels are ordinal (increasing severity).
    # We therefore use *clinical severity* order as the canonical class index.
    # Folder names under DATA_DIR are remapped to this order inside dataset.py.
    CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

    # ── NVIDIA A4000 16GB — Tuned Settings ───────────────────────────────
    BATCH_SIZE      = 64        # A4000 can handle 64 comfortably at 224×224
    NUM_EPOCHS      = 30
    NUM_WORKERS     = 8         # A4000 workstation typically has 8+ cores
    PIN_MEMORY      = True
    USE_AMP         = True      # fp16 — cuts memory in half, 2× faster

    # ── Imbalance handling ───────────────────────────────────────────────
    USE_WEIGHTED_SAMPLER = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    LR_HEAD         = 1e-4
    LR_BACKBONE     = 1e-5
    WEIGHT_DECAY    = 1e-2

    # ── Scheduler ─────────────────────────────────────────────────────────
    T_MAX           = 30
    ETA_MIN         = 1e-6

    # ── Regularisation ────────────────────────────────────────────────────
    LABEL_SMOOTHING = 0.05
    DROPOUT         = 0.3

    # ── QWK optimization ──────────────────────────────────────────────────
    # Tune decision thresholds on validation expected-severity scores.
    # This often improves QWK over raw argmax predictions.
    TUNE_QWK_THRESHOLDS      = True
    QWK_THRESHOLD_ITERS      = 2
    QWK_THRESHOLD_GRID_SIZE  = 80

    # ── Misc ──────────────────────────────────────────────────────────────
    SEED        = 42
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR    = "outputs/"
    BEST_MODEL  = "outputs/best_model.pth"
    PLOTS_DIR   = "outputs/plots"
    VAL_SPLIT   = 0.2           # 80/20 train-val split

cfg = Config()