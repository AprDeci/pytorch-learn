import torch

w = torch.tensor(0.5, requires_grad=True)
m = torch.tensor(8.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1.0, 2.0])
y = torch.tensor([2.0, 4.0])
z_true = torch.tensor([33.0, 60.0])
lr = 0.01

for step in range(10000):
    # 前向
    z_pred = w * x+m*y+b
    loss = ((z_pred - z_true)**2).mean()

    # 反向
    loss.backward()

    # 更新（注意 no_grad）
    with torch.no_grad():
        w -= lr * w.grad
        m -= lr * m.grad
        b -= lr * b.grad
        # 清空梯度（重要！否则会累加）
        w.grad.zero_()
        m.grad.zero_()
        b.grad.zero_()

    print(f"Step {step}: loss={loss.item():.4f}, w={w.item():.4f},m ={m.item():.4f} , b={b.item():.4f}")