import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data", "colored_images")

    CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
    NUM_CLASSES = len(CLASS_NAMES)

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    SEED = 42

    NUM_WORKERS = 4
    PIN_MEMORY = True

    USE_WEIGHTED_SAMPLER = True

    LR = 3e-5
    EPOCHS = 20

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "output_V1")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


cfg = Config()
