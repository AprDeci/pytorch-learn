import torch
import torch.nn as nn


class MINSTNet(nn.Module):
    def __init__(self) -> None:
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


model = MINSTNet()
model.load_state_dict(torch.load("minst.pth", weights_only=True))

# 评估模式
model.eval()
print("Model loaded")

# 载入图片

from PIL import Image
import torchvision.transforms as transforms


def predict_image(img_path, model):
    # 加载图片 转灰度
    img = Image.open(img_path).convert("L")

    # 预处理
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # 調整為 28x28
            transforms.ToTensor(),  # 轉 Tensor 並歸一化到 [0,1]
        ]
    )
    img_tensor = transform(img).unsqueeze(0)

    # 3. 預測
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)  # 轉為概率
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()

    return pred, confidence


# 先用 MNIST 測試集的一張圖
import torchvision

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)
img, label = test_dataset[0]  # 取第一張

# 保存這張圖（方便你看）
img.save("test_digit.png")
print(f"真實標籤: {label}")

# 預測
pred, conf = predict_image("test_digit.png", model)
print(f"預測結果: {pred}, 置信度: {conf:.2%}")
