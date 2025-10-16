import torch
torch.manual_seed(42)
from nndl.dataset import X_train, y_train, X_dev, y_dev

from nndl.op import model_LR
from nndl.op import BinaryCrossEntropyLoss
from nndl.opitimizer import SimpleBatchGD
from nndl.metric import accuracy
from nndl.runner import Runner
from nndl.tools import plot

if __name__ == "__main__":


    input_dim = 2
    lr = 0.1

    model = model_LR(input_dim=input_dim)
    optimizer = SimpleBatchGD(init_lr=lr, model=model)
    loss_fn = BinaryCrossEntropyLoss()
    metric = accuracy

    runner = Runner(model, optimizer, metric, loss_fn)
    runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=500, log_epochs=50, save_path="best_model.pth")

    plot(runner,fig_name='linear-acc-fixed_6.png')