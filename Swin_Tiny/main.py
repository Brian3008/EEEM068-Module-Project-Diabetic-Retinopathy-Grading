from pathlib import Path

from src.config import cfg
from src.eval import full_evaluation
from src.preprocess import preprocess_all
from src.train import train


def main():
    print("=" * 55)
    print("  DR Grading: Swin-Tiny on Kaggle DR-2015")
    print("=" * 55)

    processed = Path(cfg.PROCESSED_DIR)
    needs_preprocess = (
        not processed.exists()
        or sum(1 for _ in processed.glob("*.jpeg")) < 1000
    )
    if needs_preprocess:
        print("\n  Preprocessing raw images (one-time, ~30-60 min)...")
        preprocess_all()
    else:
        n = sum(1 for _ in processed.glob("*.jpeg"))
        print(f"\n  Found {n} preprocessed images in {cfg.PROCESSED_DIR}")

    history = train()

    full_evaluation(history=history)


if __name__ == "__main__":
    main()