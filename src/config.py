import torch
from pathlib import Path


class Config:

    RAW_DATA_DIR  = "src/data/raw/train"             
    LABELS_CSV    = "src/data/raw/trainLabels.csv"   
    PROCESSED_DIR = "src/data/processed"             

    NUM_CLASSES = 5
    CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

    
    IMAGE_SIZE     = 224
    PROCESSED_SIZE = 512   

    BATCH_SIZE   = 32          
    NUM_EPOCHS   = 25
    NUM_WORKERS  = 8
    PIN_MEMORY   = True
    USE_AMP      = True

    
    USE_WEIGHTED_SAMPLER = True
    USE_LOSS_WEIGHTS     = False

    LR_HEAD       = 1e-4
    LR_BACKBONE   = 1e-5
    WEIGHT_DECAY  = 1e-2
    WARMUP_EPOCHS = 2

    ETA_MIN = 1e-6

    LABEL_SMOOTHING = 0.05
    DROPOUT         = 0.2

    TUNE_QWK_THRESHOLDS     = True
    QWK_THRESHOLD_ITERS     = 3
    QWK_THRESHOLD_GRID_SIZE = 100

    SEED       = 42
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR   = "outputs/"
    BEST_MODEL = "outputs/best_model.pth"
    PLOTS_DIR  = "outputs/plots"
    VAL_SPLIT  = 0.15           


cfg = Config()