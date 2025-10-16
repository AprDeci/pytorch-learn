import torch

torch.manual_seed(42)


# 多输出梯度
x = torch.tensor([1.0, 2.0], requires_grad=True)

y = x**2  # [1,4]

grad_outputs = torch.tensor([0.5, 1.0])  # 梯度权重 loss = 0.5*1+1*4 = 4.5
y.backward(gradient=grad_outputs)

print(x.grad)

x.grad.zero_()


# jacobian
# 完整示例：f(x) = x^2，向量版
def f(x):
    return x**2  # x [2] → y [2]


x = torch.tensor([1.0, 2.0], requires_grad=True)
jac = torch.autograd.functional.jacobian(f, x)
print("Jacobian 矩阵 (对角: 2x):", jac)
print("解释: ∂y1/∂x1=2*1=2, ∂y2/∂x2=4，其他0")

# 另一个例子：线性 f(x) = A x + b
A = torch.tensor([[1, 2], [3, 4]], requires_grad=False)
b = torch.tensor([0.5, 0.5], requires_grad=False)


def linear_fn(x):
    return A * x + b


x_lin = torch.tensor([1.0, 1.0], requires_grad=True)
jac_lin = torch.autograd.functional.jacobian(linear_fn, x_lin)
print("\n线性 Jacobian (就是 A):", jac_lin)


from torchview import draw_graph  # 导入

torch.manual_seed(42)

# 先修复你的 Jacobian 示例（正确输出 [[1,2],[3,4]]）
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 去掉 requires_grad=False（A 不需梯度）
b = torch.tensor([0.5, 0.5])


def linear_fn(x):
    return A @ x + b


x_lin = torch.tensor([1.0, 1.0], requires_grad=True)
jac_lin = torch.autograd.functional.jacobian(linear_fn, x_lin)
print("线性 Jacobian (就是 A):", jac_lin)  # 正确: tensor([[1., 2.], [3., 4.]])


# 现在画图：只画模型（tensor 计算图跳过，或用下面 TensorBoard）
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入1维

    def forward(self, x):
        return self.linear(x) ** 2


net = SimpleNet()
# 用 input_size=(1,1)：batch=1, features=1（Linear 输入）
net_graph = draw_graph(
    net, input_size=(1, 1), depth=3, save_graph=True, filename="torchview_net.png"
)
print("网络图已保存: torchview_net.png")  # 生成 PNG，无报错
