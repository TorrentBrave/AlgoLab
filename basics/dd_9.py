import torch

# 当两个Tensor的形状一致时, 可以广播
x = torch.ones((10, 1, 5, 2))
y = torch.ones((3, 2, 5))

z = torch.matmul(x, y)

print(z.shape)


