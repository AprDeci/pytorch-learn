import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

# 测试集
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform
)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False  # 测试不用打乱
)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)  # 标准化
train_ds = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


# CNN 模型（Conv + Pool + FC）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )  # 输入1通道，32滤镜，3x3核
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32→64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后 (28/4=7)
        self.fc2 = nn.Linear(128, 10)  # 10类
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(
            self.relu(self.conv1(x))
        )  # Conv1 + ReLU + Pool: [1,28,28] → [32,14,14]
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 + ReLU + Pool: [64,7,7]
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 logits
        return x


# 实例 + 训练（5 epoch 看 acc）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, loader):
    model.train()
    total_correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        total_correct += (predicted == labels).sum().item()
    return 100 * total_correct / total


for epoch in range(5):
    acc = train_epoch(model, train_loader)
    print(f"Epoch {epoch+1}: Train Acc {acc:.2f}%")  # 预期: Epoch1 ~90%, Epoch5 ~99%
