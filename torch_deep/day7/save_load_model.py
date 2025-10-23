import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image  # 预测用

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
full_ds = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_ds, val_ds, _ = random_split(full_ds, [48000, 6000, 6000])
test_ds = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


class MNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def quick_train(model, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1} 训完")


model = MNNet().to(device)
quick_train(model, 10)  # 训10 epoch

# 保存参数
torch.save(model.state_dict(), "mnist_model.pth")
print("✅ 模型保存: mnist_model.pth")


# 加载
new_model = MNNet().to(device)
new_model.load_state_dict(
    torch.load("mnist_model.pth", map_location=device)
)  # map_location=CPU/GPU 适配
new_model.eval()  # 推理模式
print("✅ 模型加载成功")


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


test_acc = evaluate(new_model, test_loader)
print(f"加载后 Test Acc: {test_acc:.2f}%")


def predict_digit(model, img_tensor):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        pred = output.argmax(1).item()
        probs = torch.softmax(output, 1)[0].max().item()
    return pred, probs


test_img, test_label = test_ds[0]
test_tensor = test_img.unsqueeze(0)  # [1,1,28,28]
pred, conf = predict_digit(new_model, test_tensor)
plt.imshow(test_img.squeeze(), cmap="gray")
plt.title(f"真标签: {test_label}, 预测: {pred} (置信: {conf:.2%})")
plt.show()
print(f"预测正确？ {pred == test_label}")
