import torch

def mean_squared_error(y_true, y_pred):
    """
    输入:
        - y_true: tensor, 样本真实标签
        - y_pred: tensor, 样本预测标签
    输出:
        - error: float, 误差值
    注意:
        - 代码中没有除以2
    """
    assert y_true.shape[0] == y_pred.shape[0]
    error = torch.mean(torch.square(y_true - y_pred))
    return error

if __name__ == "__main__":
    # [N, 1], N = 2
    y_true = torch.tensor([[-0.2], [4.9]], dtype=torch.float32)
    y_pred = torch.tensor([[1.3], [2.5]], dtype=torch.float32)

    error = mean_squared_error(y_true=y_true, y_pred=y_pred).item()

    print("error:", error)
