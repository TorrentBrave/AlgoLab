import torch

# x 是 tensor
def softmax(X):
    """
    输入:
        - X: shape=[N,C], N 为向量数量, C 为向量维度
    torch.max
        - torch.return_types.max(values=tensor([[3.],
                                        [6.]]),
                        indices=tensor([[1],
                                        [2]]))
    dim=1
        - 在每行中的三个类别概率中选出最大的概率                  
    输出: 
        - 归一化得到 softmax输出
        - [[0.090, 0.665, 0.245],   # 每行的概率和为1
          [0.867, 0.117, 0.016]]
        - 出去后又和 label 比较           
    """
    # 防溢出处理每行取最大值
    x_max = torch.max(X, dim=1, keepdim=True)[0] # N, 1 取出 value
    
    x_exp = torch.exp(X - x_max)
    partition = torch.sum(x_exp, dim=1, keepdim=True) # N, 1

    return x_exp / partition