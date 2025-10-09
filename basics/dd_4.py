import torch
ndim2_Tensor = torch.ones([5, 10])

print(ndim2_Tensor.dtype)
new_Tensor= ndim2_Tensor.to(torch.int64)
print(new_Tensor.dtype)
print(new_Tensor)
print()