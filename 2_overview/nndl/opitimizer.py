import torch

def optimizer_lsm(model, X, y, reg_lambda=0):
    """
    输入:
        - model: 模型
        - X: tensor, 特征数据, shape=[N,D]
        - y: tensor, 标签数据, shape=[N]
        - reg_lambda: float, 正则化系数, 默认为0
    输出:
        - model: 优化好的模型

    lsm: Least Squares Method(最小二乘法)的缩写
    reg: 是(正则化) regularization的缩写
    lambda: (λ) 是正则化项前的超参数, 控制正则化的强度
    """
    N, D = X.shape

    # 对输入特征数据所有特征向量求平均
    x_bar_tran = torch.mean(X, dim=0) # shape=(D,) 1D张量的转置是多余的, 和原来的一样

    # 求标签的均值, shape=[1]
    y_bar = torch.mean(y)

    # 广播减法: X - x_bar (Pytorch 自动广播)
    x_sub = X - x_bar_tran # shape=[N,D]

    # 使用 torch.all 判断输入tensor是否全0
    if torch.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = torch.zeros(D, dtype=X.dtype, device=X.device)
        return model
    
    reg_term = reg_lambda * torch.eye(D, dtype=X.dtype, device=X.device)

    tmp = torch.inverse(torch.matmul(x_sub.T, x_sub) + reg_term)

    w = torch.matmul(tmp, torch.matmul(x_sub.T, (y-y_bar)))
    
    b = y_bar - torch.matmul(x_bar_tran, w)

    model.params['b'] = b
    model.params['w'] = w

    return model