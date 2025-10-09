import torch

ndim_2_Tensor = torch.tensor([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11]])

print(ndim_2_Tensor)

print(ndim_2_Tensor[0])
print(ndim_2_Tensor[0, :])
print(ndim_2_Tensor[:, 0])
print(ndim_2_Tensor[:, -1])
print(ndim_2_Tensor[:])
print(ndim_2_Tensor[0, 1])