import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        self.image_index = {}

        for root, _, files in os.walk(img_dir):
            for f in files:
                self.image_index[f] = os.path.join(root, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0] + ".png"
        label = int(self.df.iloc[idx, 1])

        img_path = self.image_index.get(img_id, None)

        if img_path is None:
            raise FileNotFoundError(f"Missing image: {img_id}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label




def get_data_loaders(
    data_dir="data/colored_images/colored_images",
    csv_file="data/trainLabels.csv",
    batch_size=32
):

    df = pd.read_csv(csv_file)

    # simple stratified split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_tf, val_tf = get_transforms()

    train_dataset = DRDataset(train_df, data_dir, train_tf)
    val_dataset = DRDataset(val_df, data_dir, val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader