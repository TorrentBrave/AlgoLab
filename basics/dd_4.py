import torch
ndim2_Tensor = torch.ones([5, 10])

print(ndim2_Tensor.dtype)
print(ndim2_Tensor.device)

print(ndim2_Tensor)

ndim2_Numpy = ndim2_Tensor.numpy()

print(ndim2_Numpy)
print(ndim2_Numpy.dtype)