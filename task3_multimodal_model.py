import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
CSV_PATH = "houses.csv"
IMAGE_DIR = "house_images"
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class HousingDataset(Dataset):
    def __init__(self, df, tab_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.tab_features = tab_features
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(IMAGE_DIR, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tabular = torch.tensor(
            row[self.tab_features].values.astype(np.float32),
            dtype=torch.float32
        )

        price = torch.tensor([row["price"]], dtype=torch.float32)
        return image, tabular, price


# ---------------- MODEL ----------------
class MultiModalModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.tab_fc = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tab_fc(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(combined)



# ---------------- MAIN ----------------
def main():
    df = pd.read_csv(CSV_PATH)

    # image name create
    df["image_name"] = df["image_id"].astype(str) + ".jpg"

    # remove missing images
    exists_mask = df["image_name"].apply(
        lambda x: os.path.exists(os.path.join(IMAGE_DIR, x))
    )
    df = df[exists_mask]

    print(f"Remaining samples after image check: {len(df)}")

    # tabular features (NO strings)
    tab_features = ["n_citi", "bed", "bath", "sqft"]
    print("Tabular features used:", tab_features)

    # scale tabular
    tab_scaler = StandardScaler()
    df[tab_features] = tab_scaler.fit_transform(df[tab_features])

    # scale price (VERY IMPORTANT)
    price_scaler = StandardScaler()
    df["price"] = price_scaler.fit_transform(df[["price"]])

    # split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = HousingDataset(train_df, tab_features, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiModalModel(tab_dim=len(tab_features)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for img, tab, price in train_loader:
            img, tab, price = img.to(DEVICE), tab.to(DEVICE), price.to(DEVICE)

            optimizer.zero_grad()
            preds = model(img, tab)
            loss = criterion(preds, price)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
