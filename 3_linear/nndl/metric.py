import torch

def accuracy(preds, labels):
    """
    输入:
        - preds: 预测值, 二分类时, shape=[N, 1], N 为样本数量, 多分类时, shape=[N, C], C 为类别数量
        - labels: 真实标签, shape=[N, 1]
    输出:
        - 准确率: shape=[1]

    二分类用 float 是为了方便布尔平均求准确率，多分类用 int 是因为类别本身是整数索引
    """
    # 确保标签是 1D (避免 [N, 1] 形状问题)
    if labels.ndim > 1:
        labels = labels.squeeze(1) # [N]

    # 判断是二分类任务还是多分类任务,preds.shape[1]=1时为二分类任务，preds.shape[1]>1时为多分类任务
    if preds.shape[1] == 1:
        # 二分类: 概率值 > 0.5 视为类别 1, 否则为 0
        preds = (preds >= 0.5).float().squeeze(1) # [N]
    else:
        # 多分类: 取最大概率对应的类别
        preds = torch.argmax(preds, dim=1) # [N]
    acc = (preds == labels).float().mean()
    return acc


if __name__ == "__main__":
    # -----------------------------------------
    # Accuracy
    # -----------------------------------------
    # 假设模型的预测值为 [[0.], [1.], [1.], [0.]]
    # 真实类别为 [[1.], [1.], [0.], [0.]]
    preds = torch.tensor([[1.], [0.], [0.], [0.]])
    labels = torch.tensor([[1.], [1.], [0.], [0.]])
    """
    如果不压缩维度, 会自动广播
    
    preds = [1, 0, 0, 0]           # shape: (4,)
    labels = [[1],
              [1],
              [0],
              [0]]                 # shape: (4,1)

    [[ True, False, False, False],
    [True, False, False, False],
    [False,  True,  True,  True],
    [False,  True,  True,  True]]
    
    mean = 0.5
    """

    print("accuracy is:", accuracy(preds, labels))