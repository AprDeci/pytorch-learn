import torch
from torch.nn.qat import Linear
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 分 train/val
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"用设备: {device}")

data = {
    "area": np.random.uniform(50, 200, 1000),
    "price": np.random.randint(100, 500, 1000),
    "room": np.random.randint(1, 5, 1000),
}
df = pd.DataFrame(data)
df.to_csv("house_prices.csv", index=False)
print("CSV文件已生成")


class HouseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.features = self.data[["area", "room"]].values  # 输入X
        self.labels = self.data["price"].values.reshape(-1, 1)  # 输出Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y  # 特征,标签


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

train_ds = HouseDataset("train.csv")
val_ds = HouseDataset("val.csv")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))
print(f"批次形状: X {x.shape} [32,2], Y {y.shape} [32,1]")


class HouseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(loader)


model = HouseMLP().to(device)
criterion = nn.MSELoss()  # 回归用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_losses, val_losses = [], []
for epoch in range(5000):
    tl = train_epoch(model, train_loader, criterion, optimizer)
    vl = val_epoch(model, val_loader, criterion)
    train_losses.append(tl)
    val_losses.append(vl)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Train Loss {tl:.4f}, Val Loss {vl:.4f}")

# 画曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("House Price Prediction Loss")
plt.legend()
plt.show()
