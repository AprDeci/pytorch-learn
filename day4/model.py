import torch
import torch.nn as nn
from matplotlib import pyplot as plt

x = torch.linspace(-2 * 3.1416, 2 * 3.1416, 200).reshape(-1, 1)
y = x*torch.sin(x) + 0.05 * torch.randn_like(x)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    pred = model(x)

    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():  # 关闭梯度，节省内存
    x_test = torch.linspace(-2 * 3.1416, 2 * 3.1416, 300).reshape(-1, 1)
    y_pred = model(x_test)
    y_true = x_test*torch.sin(x_test)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=10, alpha=0.6, label='Noisy data')
plt.plot(x_test, y_true, 'r--', label='True sin(x)', linewidth=2)
plt.plot(x_test, y_pred, 'b', label='Neural Net', linewidth=2)
plt.legend()
plt.title('Neural Network Fitting sin(x)')
plt.show()