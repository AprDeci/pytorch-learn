import torch
import torch.nn as nn

class TwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)  # 输入2维 → 输出1维
        # self.layer.bias.fill_(1)
        # self.layer.bias.requires_grad = False

    def forward(self, x):
        return self.layer(x)

# 两个样本：(x=1, y=2) 和 (x=2, y=3)
x_data = torch.tensor([[1.0, 2.0],    # 第一个样本
                       [2.0, 3.0],
                       [5.0,7.0]])   # 第二个样本 → shape [2, 2]

z_true = torch.tensor([[2*1.0 + 3*2.0 + 1],   # = 9
                       [2*2.0 + 3*3.0 + 1],
                       [2*5.0 + 3*7.0 + 1]])  # = 14 → shape [2, 1]

model = TwoInputModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    pred = model(x_data)
    loss = criterion(pred, z_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 查看结果
w1, w2 = model.layer.weight[0, 0].item(), model.layer.weight[0, 1].item()
b = model.layer.bias.item()
print(f"学到: z = {w1:.2f}*x + {w2:.2f}*y + {b:.2f}")