import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 分 train/val
import torch.nn as nn
import torch.optim as optim

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
