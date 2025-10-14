import torch
import torch.nn as nn

class MyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=2, out_features=1)

    def forward(self,x):
        output = self.linear_layer(x)
        return output

model = MyLinearModel()

x_data = torch.tensor([[1.0, 2.0],    # 第一个样本
                       [2.0, 3.0],
                      [3.0,4.0]])   # 第二个样本 → shape [2, 2]

z_true = torch.tensor([[2.0*1 + 3.0*2 + 1],   # = 9
                       [2.0*2 + 3.0*3 + 1],
                       [2.0*3 + 3.0*4 + 1]])  # = 14 → shape [2, 1]


# 损失函数：均方误差
criterion = nn.MSELoss()

# 优化器：自动获取 model 的所有参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100000):
    # 1. 前向：预测
    pred = model(x_data)  # 等价于 model.forward(x)

    # 2. 计算损失
    loss = criterion(pred, z_true)

    # 3. 反向：计算梯度
    optimizer.zero_grad()  # 清空旧梯度
    loss.backward()        # 自动求导


    # 4. 更新参数
    optimizer.step()       # 自动更新所有参数


    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

w_learned = model.linear_layer.weight.detach()  # .item() 转成 Python 数
b_learned = model.linear_layer.bias.item()



print(f"真实公式: z = 2.0 * x1 + 3.0 * x2 + 1.0")
print(f"学到公式: y = {w_learned[0,0]}x1+{w_learned[0,1]}x2 + {b_learned:.2f}")