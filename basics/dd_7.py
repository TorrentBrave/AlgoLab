import torch

nim2_Tensor = torch.ones([2, 3], dtype=torch.float32)

print('Origin Tensor: ', nim2_Tensor)

# 修改第 1维为0
# 等价于 nim2_Tensor[0] = 0
# 等价于 nim2_Tensor[0:1] = 0
nim2_Tensor[0, :] = 0
print('Change Tensor: ', nim2_Tensor)

# 修改第 1维为2.1
nim2_Tensor[0:1] = 2.1
print('Change Tensor: ', nim2_Tensor)

# 修改全部Tensor
nim2_Tensor[...] = 3
print('Change Tensor: ', nim2_Tensor)

print(nim2_Tensor.dtype)