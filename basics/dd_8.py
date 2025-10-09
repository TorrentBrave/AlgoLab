import torch

x = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float64)
y = torch.tensor([[5.5, 6.6], [7.7, 8.8]], dtype=torch.float64)

print(x.dtype)

print(x)
print("Method 1_torch API: ", torch.add(x, y))
print("Method 2_张量类成员: ", x.add(y))



