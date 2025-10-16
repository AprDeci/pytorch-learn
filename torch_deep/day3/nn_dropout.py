import torch
import torch.nn as nn
from torch.nn.modules import Dropout
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

torch.manual_seed(42)

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


class BaseLineNet(nn.Module):
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
        return self.fc(x.view(-1, 28 * 28))


class DropoutNet(nn.Module):
    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    return 100 * correct / total


baseline = BaseLineNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
train_losses_base, train_accs_base, test_accs_base = [], [], []

epochRange = 10

for epoch in range(epochRange):
    tl, ta = train_epoch(baseline, train_loader, criterion, optimizer)
    te = evaluate(baseline, test_loader)
    train_losses_base.append(tl)
    train_accs_base.append(ta)
    test_accs_base.append(te)
    print(
        f"基线 Epoch {epoch+1}: Train Loss {tl:.4f}, Train Acc {ta:.2f}%, Test Acc {te:.2f}%"
    )

# 添加Dropout
dropout_net = DropoutNet().to(device)
optimizer_drop = torch.optim.Adam(dropout_net.parameters(), lr=0.001)
train_losses_dropout, train_accs_dropout, test_accs_dropout = [], [], []
for epoch in range(epochRange):
    tl, ta = train_epoch(dropout_net, train_loader, criterion, optimizer_drop)
    te = evaluate(dropout_net, test_loader)
    train_losses_dropout.append(tl)
    train_accs_dropout.append(ta)
    test_accs_dropout.append(te)
    print(
        f"Dropout Epoch {epoch+1}: Train Loss {tl:.4f}, Train Acc {ta:.2f}%, Test Acc {te:.2f}%"
    )

# 画对比曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses_base, label="Baseline Train Loss")
plt.plot(train_losses_dropout, label="Dropout Train Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_accs_base, label="Baseline Train Acc")
plt.plot(test_accs_base, label="Baseline Test Acc")
plt.plot(train_accs_dropout, label="Dropout Train Acc")
plt.plot(test_accs_dropout, label="Dropout Test Acc")
plt.legend()
plt.title("Accuracy")
plt.show()
