import torch

ndim_1_Tensor = torch.arange(start=1, end=10, step=1)

print(ndim_1_Tensor)
print(ndim_1_Tensor.dtype)

print(ndim_1_Tensor[0])
print(ndim_1_Tensor[1])
print(ndim_1_Tensor[-1])
print(ndim_1_Tensor[:3])
print(ndim_1_Tensor[::3])
# print(ndim_1_Tensor[::-1])