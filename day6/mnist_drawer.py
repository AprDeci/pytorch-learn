import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
import numpy as np
from PIL import Image


# 1. 定義模型（和之前一樣）
class MNISTNet(nn.Module):
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


# 2. 載入模型（假設你有 'mnist_model.pth'）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTNet().to(device)
model.load_state_dict(torch.load("minst.pth", weights_only=True))
model.eval()


# 3. 預測函數
def predict_digit(img_tensor):
    with torch.no_grad():
        output = model(img_tensor.to(device))
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()
    return pred, confidence


# 4. 互動畫板類別
class DigitDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(0, 28)
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()  # 讓鼠標畫畫時從上往下
        self.ax.set_title("畫數字 (0-9)，實時預測")

        # 創建空白畫布（28x28 黑色為0，白色為1；MNIST 背景白=0，黑=1）
        self.canvas_data = np.ones((28, 28))  # 初始全白 (1)
        self.im = self.ax.imshow(self.canvas_data, cmap="gray", vmin=0, vmax=1)

        # 連接鼠標事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.drawing = False
        self.last_pos = None

        # 預測文字
        self.pred_text = self.ax.text(
            0.5,
            -0.1,
            "預測: -- (置信度: --%)",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            fontsize=12,
            color="red",
        )

        plt.tight_layout()
        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.last_pos = (int(event.xdata), int(event.ydata))

    def on_release(self, event):
        self.drawing = False
        self.update_prediction()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if self.last_pos:
            self.draw_line(self.last_pos[0], self.last_pos[1], x, y, width=3)
        self.last_pos = (x, y)
        self.im.set_data(self.canvas_data)
        self.fig.canvas.draw_idle()

    def draw_line(self, x0, y0, x1, y1, width=3):
        # 簡單的線條繪製（Bresenham 演算法簡化版）
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < 28 and 0 <= y0 < 28:
                # 畫黑點 (設為0)
                self.canvas_data[y0, x0] = 0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def update_prediction(self):
        # 轉換為 Tensor：白=0，黑=1（MNIST 標準）
        img_array = 1 - self.canvas_data  # 反轉：黑=1，白=0
        img_tensor = (
            torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )  # [1,1,28,28]

        pred, conf = predict_digit(img_tensor)
        self.pred_text.set_text(f"預測: {pred} (置信度: {conf:.1%})")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "c" or event.key == "C":
            self.canvas_data = np.ones((28, 28))  # 清空為白
            self.im.set_data(self.canvas_data)
            self.pred_text.set_text("預測: -- (置信度: --%)")
            self.fig.canvas.draw_idle()
        elif event.key == "s" or event.key == "S":
            # 保存為 PNG
            img = Image.fromarray((self.canvas_data * 255).astype(np.uint8), mode="L")
            img.save("drawn_digit.png")
            print("✅ 已保存為 drawn_digit.png")


# 5. 啟動畫板
drawer = DigitDrawer()
