import torch

ndim2_Tensor = torch.ones([5, 10])
print(ndim2_Tensor)
print(ndim2_Tensor.shape)

new_ndim2_Tensor = torch.unsqueeze(ndim2_Tensor, axis=0)
print(new_ndim2_Tensor)
print(new_ndim2_Tensor.shape)