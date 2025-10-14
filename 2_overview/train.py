import torch

from nndl.dataset import X_train, y_train, X_test, y_test   # 数据
from nndl.op import Linear  # 模型
from nndl.squared_error import mean_squared_error   # 损失函数
from nndl.opitimizer import optimizer_lsm   # 参数学习

torch.manual_seed(42)


if __name__ == "__main__":
    model = Linear(input_size=1)
    model = optimizer_lsm(model, X_train.reshape([-1,1]), y_train.reshape([-1,1]))
    # 模型是一维线性回归(input_size=1), 即每个样本只有一个特征, X_train 可能是一个 1D 张量 (shape=(N,))
    # 而 Linear 模型和 optimizer_lsm 函数都假设输入是 2D 矩阵, shape=(N,D)
    print("w_pred:", model.params['w'].item(), "b_pred:", model.params['b'].item())

    y_train_pred = model(X_train.reshape([-1,1])).squeeze()
    train_error = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
    print("train error:", train_error)