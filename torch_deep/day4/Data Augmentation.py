import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_loaders(batch_size=128, augmented=False):
    if augmented:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),  # 随机裁剪 输出32*32 填充4
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomRotation(degrees=15),  # 随机旋转
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),  # 随机调整亮度、对比度、饱和度
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform_train = transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class CIFARCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [32,32,32] -> [32,16,16]
        x = self.pool(self.relu(self.conv2(x)))  # [32,16,16] -> [64,8,8]
        x = self.pool(self.relu(self.conv3(x)))  # [64,8,8] -> [128,4,4]
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 训练/评估函数
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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


train_loader_plain, test_loader = get_loaders()
model_plain = CIFARCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_plain.parameters(), lr=0.001)
train_accs_plain, test_accs_plain = [], []
for epoch in range(5):
    _, ta = train_epoch(model_plain, train_loader_plain, criterion, optimizer)
    te = evaluate(model_plain, test_loader)
    train_accs_plain.append(ta)
    test_accs_plain.append(te)
    print(f"原始 Epoch {epoch+1}: Train Acc {ta:.2f}%, Test Acc {te:.2f}%")

train_loader_aug, test_loader = get_loaders(augmented=True)
model_aug = CIFARCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_aug.parameters(), lr=0.001)
train_accs_aug, test_accs_aug = [], []
for epoch in range(5):
    _, ta = train_epoch(model_aug, train_loader_aug, criterion, optimizer)
    te = evaluate(model_aug, test_loader)
    train_accs_aug.append(ta)
    test_accs_aug.append(te)
    print(f"增强 Epoch {epoch+1}: Train Acc {ta:.2f}%, Test Acc {te:.2f}%")

# 曲线对比
plt.rcParams["font.family"] = "Maple Mono NF CN Thin"
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(test_accs_plain, label="原始 Test Acc")
plt.plot(test_accs_aug, label="增强 Test Acc")
plt.title("Test Acc 对比（增强更高！）")
plt.legend()

plt.subplot(1, 2, 2)
# 随机看增强效果
img_aug, _ = next(iter(train_loader_aug))
plt.imshow(transforms.ToPILImage()(img_aug[0]))  # 第一张增强图
plt.title("增强后图片示例")
plt.show()
