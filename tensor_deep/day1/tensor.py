import torch

torch.manual_seed(42)

print("---------tensor.zeros_like----------")
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros_like_x = torch.zeros_like(x)
print("x:", x)
print("zeros_like_x:", zeros_like_x)

print("---------tensor.cat----------")
a = torch.rand(2, 3)
b = torch.rand(2, 3)
cat_ab = torch.cat([a, b], dim=1)
print("cat_ab:", cat_ab.shape)

print("---------tensor.unsqueeze----------")
y = torch.randn(3)
print("y.shape:", y.shape)
print("y:", y)
unsq_y = y.unsqueeze(0)
print("unsq_y.shape:", unsq_y.shape)

print("---------tensor.where----------")
z = torch.tensor([1.0, -2.0, 3.0, -4.0])
mask = z > 0
result = torch.where(mask, z, torch.zeros_like(z))
print("result:", result)


noisy = torch.randn(5) + 0.5 * torch.randn(5)
print("noisy:", noisy)
noisy = torch.where(noisy > 0, noisy, torch.zeros_like(noisy)).clamp(0, 1)
print("noisy:", noisy)

print("---------tensor.gather----------")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = torch.tensor([[0, 1], [1, 0], [2, 1]])
gatered = x.gather(1, idx)
print("gatered:", gatered)

print("---------tensor.expand----------")
x = torch.randn(2, 1, 3)
expanded = x.expand(2, 3, 3)
print("expanded:", expanded)
print("expanded.shape:", expanded.shape)


print("---------tensor.einsum----------")
a = torch.full((2, 3), 1)
b = torch.full((3, 2), 2)
# 矩阵乘法
einsum_ab_1 = torch.einsum("ij,jk->ik", a, b)
# 转置
einsum_ab_2 = torch.einsum("ij->ji", a)
print("矩阵乘法 einsum_ab_1:", einsum_ab_1)
print("转置 einsum_ab_2:", einsum_ab_2)
