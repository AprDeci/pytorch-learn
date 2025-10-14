import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------
# 1. 准备数据：y = sin(x) + 噪声
# ----------------------------
x = torch.linspace(-2 * 3.1416, 2 * 3.1416, 200).reshape(-1, 1)  # 200个点，形状 [200, 1]
y = torch.sin(x) + 0.05 * torch.randn_like(x)  # 加点小噪声，更真实


# ----------------------------
# 2. 定义神经网络模型
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 三层全连接网络：1 → 64 → 64 → 1
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),  # 激活函数！让网络能学非线性
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 y
        )

    def forward(self, x):
        return self.layers(x)


model = Net()

# ----------------------------
# 3. 设置训练组件
# ----------------------------
criterion = nn.MSELoss()  # 均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器（比 SGD 更稳）

# ----------------------------
# 4. 训练循环
# ----------------------------
for epoch in range(2000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ----------------------------
# 5. 可视化结果
# ----------------------------
with torch.no_grad():  # 关闭梯度，节省内存
    pred_final = model(x)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=10, label='Noisy data', alpha=0.6)
plt.plot(x, torch.sin(x), 'r--', label='True sin(x)', linewidth=2)
plt.plot(x, pred_final, 'b', label='Neural Net Fit', linewidth=2)
plt.legend()
plt.title('Neural Network Fitting sin(x)')
plt.show()