# save this as check_run3.py in your D:\dr_beit folder
import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter
from src.dataset import DRFolderDataset, get_transforms
from src.model import build_beit_model
from src.utils import set_seed
from src.metrics import compute_all_metrics, plot_confusion_matrix
from torch.amp import autocast
from tqdm import tqdm
import numpy as np

# must match run 3 settings exactly
IMAGE_DIR  = "data/images"
SEED       = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT  = 0.15
NUM_WORKERS = 4

class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        import cv2
        from PIL import Image
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load dataset
    full_dataset = DRFolderDataset(
        IMAGE_DIR,
        transform=get_transforms(IMAGE_SIZE, "val")
    )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    _, val_ds  = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_ds_clean = SubsetWithTransform(val_ds, get_transforms(IMAGE_SIZE, "val"))
    val_loader   = DataLoader(
        val_ds_clean, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # load best run 3 checkpoint
    model = build_beit_model(num_classes=5).to(device)
    ckpt  = torch.load("outputs_v3/checkpoints/best_beit.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} with QWK {ckpt['qwk']:.4f}")

    # evaluate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            with autocast("cuda"):
                outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_all_metrics(all_labels, all_preds)
    accuracy = (all_preds == all_labels).mean()

    print(f"\nRun 3 Results:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  QWK      : {metrics['qwk']:.4f}")
    print(f"  Macro F1 : {metrics['f1_macro']:.4f}")
    print(f"\n{metrics['report']}")

    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    main()