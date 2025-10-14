import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


class MINSTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


model = MINSTNet().to(device)

criterion = nn.CrossEntropyLoss()  # 分类任务首选！
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)  # shape: [batch_size]

        # 前向
        outputs = model(images)  # shape: [batch_size, 10]
        loss = criterion(outputs, labels)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)  # 取最大 logit 的索引
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_acc = evaluate(model, test_loader, device)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print("-" * 40)

# # 取一个 batch
# dataiter = iter(train_loader)
# images, labels = next(dataiter)

# # 显示第一张图
# plt.imshow(images[0].squeeze(), cmap='gray')  # squeeze 去掉通道维
# plt.title(f"Label: {labels[0].item()}")
# plt.show()

torch.save(model.state_dict(), "minst.pth")
print("Model saved to minst.pth")
