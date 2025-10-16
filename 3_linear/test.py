import torch
torch.manual_seed(42)
from nndl.dataset import X_train, y_train, X_dev, y_dev, X_test, y_test, X, y

from nndl.op import model_LR
from nndl.op import BinaryCrossEntropyLoss
from nndl.opitimizer import SimpleBatchGD
from nndl.metric import accuracy
from nndl.runner import Runner
from nndl.tools import plot

import matplotlib.pyplot as plt
def decision_boundary(w, b, x1):
    """
    可视化拟合决策边界 Xw+b=0
    """
    w1, w2 = w
    x2 = (- w1 * x1 - b) / w2
    return x2

if __name__ == '__main__':
    input_dim = 2
    lr = 0.1

    model = model_LR(input_dim=input_dim)
    optimizer = SimpleBatchGD(init_lr=lr, model=model)
    loss_fn = BinaryCrossEntropyLoss()
    metric = accuracy

    runner = Runner(model, optimizer, metric, loss_fn)
    # runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=500, log_epochs=50, save_path="best_model.pth")

    score, loss = runner.evaluate([X_test, y_test])
    print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))

    plt.figure(figsize=(5,5))
    # 绘制原始数据
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())

    w = model.params['w']
    b = model.params['b']
    x1 = torch.linspace(-2, 3, 1000)
    x2 = decision_boundary(w, b, x1)
    # 绘制决策边界
    plt.plot(x1.tolist(), x2.tolist(), color="red")
    plt.savefig('linear_boundary.png')
    plt.show()


