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

# 用你 noisy 的代码扩展）：
# 生成 3x5 噪声矩阵 noisy_mat = torch.randn(3,5)；用 where 把负变0；
# 用 gather 取每行最大值的位置（index 用 torch.argmax(dim=1)）；
# 最后 expand 到 6x5（重复2次）。
noisy_mat = torch.randn(3, 5)
noisy_mat = torch.where(noisy_mat > 0, noisy_mat, torch.zeros_like(noisy_mat))
# 步骤3: argmax 找每行最大位置 (keepdim=True 让它 [3,1])
max_indices = torch.argmax(noisy_mat, dim=1, keepdim=True)  # [3,1]
print("\n最大索引:", max_indices)  # e.g., tensor([[3], [2], [0]])

# 步骤4: gather 取每行的最大值 (index 现在 [3,1]，匹配2D)
max_values = noisy_mat.gather(1, max_indices)  # [3,1]，每行取1个最大值
print("gather 后 (最大值):", max_values)

noisy_mat = max_values.expand(6, 1)
print("noisy_mat:", noisy_mat.shape)
