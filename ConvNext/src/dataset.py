import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


class RetinopathyDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + ".png"
        label = int(self.data.iloc[idx, 1])

        # FIX: search inside class folders
        img_path = None
        for cls in os.listdir(self.root_dir):
            possible = os.path.join(self.root_dir, cls, img_name)
            if os.path.exists(possible):
                img_path = possible
                break

        if img_path is None:
            raise FileNotFoundError(f"{img_name} not found")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def get_data_loaders():
    data_dir = "data/colored_images/colored_images"
    csv_file = "data/trainLabels.csv"

    train_tf, val_tf = get_transforms()

    dataset = RetinopathyDataset(data_dir, csv_file, transform=train_tf)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform = val_tf

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader