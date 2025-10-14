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


from torchviz import make_dot
import torch

# 完整示例：简单计算图
x = torch.tensor(1.0, requires_grad=True)
y = x**2 + 3 * x  # x → PowBackward0 → MulBackward0 → AddBackward0 → y
y.backward()  # 建图

# 可视化
dot = make_dot(y, params={"x": x})
dot.render("compute_graph", format="png")  # 保存 PNG
print("计算图已保存: compute_graph.png (节点: x → y，边: 梯度)")


# 复杂点：小网络
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x) ** 2


net = SimpleNet()
input = torch.tensor([[1.0]], requires_grad=True)
output = net(input)
dot_net = make_dot(output, params=dict(net.named_parameters()))
dot_net.render("net_graph", format="png")
print("网络图已保存: net_graph.png")
