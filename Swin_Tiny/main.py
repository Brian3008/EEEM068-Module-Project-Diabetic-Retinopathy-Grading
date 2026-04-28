# main.py — Run everything from here

from src.train import train
from src.eval import full_evaluation

if __name__ == "__main__":
    print("="*55)
    print("  DR Grading: Swin Transformer Tiny Pipeline  ")
    print("="*55)

    # Step 1: Train
    history = train()

    # Step 2: Evaluate best checkpoint
    full_evaluation(history=history)