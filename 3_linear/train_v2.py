import torch
torch.manual_seed(42)
from nndl.dataset import X_train, y_train, X_dev, y_dev, X_test, y_test
from nndl.dataset import make_multiclass_classification
from nndl.op import model_SR
from nndl.op import MultiCrossEntropyLoss
from nndl.opitimizer import SimpleBatchGD
from nndl.metric import accuracy
from nndl.runner import Runner
from nndl.tools import plot
import matplotlib.pyplot as plt

mode = 'test'

if __name__ == '__main__':
    if mode == 'train':
        input_dim = 2 # 特征维度
        output_dim = 3 # 类别数
        lr = 0.1 # 学习率

        model = model_SR(input_dim=input_dim, output_dim=output_dim)
        optimizer = SimpleBatchGD(init_lr=lr, model=model)
        loss_fn = MultiCrossEntropyLoss()
        metric = accuracy
        runner = Runner(model, optimizer, metric, loss_fn)
        runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=500, log_epochs=50, save_path="best_multi_model.pth")
        plot(runner, fig_name='multi_linear_acc.png')
    else:
        input_dim = 2 # 特征维度
        output_dim = 3 # 类别数
        lr = 0.1 # 学习率

        model = model_SR(input_dim=input_dim, output_dim=output_dim)
        optimizer = SimpleBatchGD(init_lr=lr, model=model)
        loss_fn = MultiCrossEntropyLoss()
        metric = accuracy
        runner = Runner(model, optimizer, metric, loss_fn)
        score, loss = runner.evaluate([X_test, y_test], load_path="best_multi_model.pth")
        print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))

        x1, x2 = torch.meshgrid(
            torch.linspace(-3.5, 2, 200),
            torch.linspace(-4.5, 3.5, 200),
            indexing="ij"
        )

        x = torch.stack([x1.flatten(), x2.flatten()],dim=1)
        y = runner.predict(x)
        y = torch.argmax(y, dim=1)

        # 绘制类别区域
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(x[:,0].tolist(), x[:,1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)
        n_samples = 1000
        X, y = make_multiclass_classification(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

        plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())
        plt.savefig('multi_linear_dataset-vis.png')
        plt.show()