import torch

torch.manual_seed(42)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros_like_x = torch.zeros_like(x)
print("x:", x)
print("zeros_like_x:", zeros_like_x)

a = torch.rand(2, 3)
b = torch.rand(2, 3)
cat_ab = torch.cat([a, b], dim=1)
print("cat_ab:", cat_ab.shape)

y = torch.randn(3)
print("y.shape:", y.shape)
print("y:", y)
unsq_y = y.unsqueeze(0)
print("unsq_y.shape:", unsq_y.shape)

z = torch.tensor([1.0, -2.0, 3.0, -4.0])
mask = z > 0
result = torch.where(mask, z, torch.zeros_like(z))
print("result:", result)

noisy = torch.randn(5) + 0.5 * torch.randn(5)
print("noisy:", noisy)
noisy = torch.where(noisy > 0, noisy, torch.zeros_like(noisy)).clamp(0, 1)
print("noisy:", noisy)
