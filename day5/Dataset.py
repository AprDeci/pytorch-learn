import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class SinDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.05):
        self.x = torch.linspace(-2 * 3.1416, 2 * 3.1416, n_samples).reshape(-1, 1)
        self.y = torch.sin(self.x) + noise * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = SinDataset(n_samples=1000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(200):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Avg Loss: {epoch_loss / len(dataloader):.6f}")