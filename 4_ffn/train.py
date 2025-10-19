import torch
torch.manual_seed(42)
# import matplotlib.pyplot as plt
from nndl.dataset import X_train, y_train, X_dev, y_dev

from nndl.op import Model_MLP_L2
from nndl.op import BinaryCrossEntropyLoss
from nndl.opitimizer import BatchGD
from nndl.metric import accuracy
from nndl.runner import RunnerV2_1
from nndl.tools import plot

model_saved_dir = "model"

epoch_num = 1000

# 输入层维度为2
input_dim = 2
# 隐藏层维度为5
hidden_dim = 5
# 输出层维度为1
output_dim = 1

# 定义网络
model = Model_MLP_L2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 损失函数
loss_fn = BinaryCrossEntropyLoss(model)

# 优化器
learning_rate = 0.2
optimizer = BatchGD(learning_rate, model)

# 评价方法
metric = accuracy

# 实例化RunnerV2_1类，并传入训练配置
runner = RunnerV2_1(model, optimizer, metric, loss_fn)

runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=epoch_num, log_epochs=50, save_dir=model_saved_dir)

plot(runner,fig_name='fw-loss2.png')

# 打印训练集和验证集的损失
# plt.figure()
# plt.plot(range(epoch_num), runner.train_loss, color="#e4007f", label="Train loss")
# plt.plot(range(epoch_num), runner.dev_loss, color="#f19ec2", linestyle='--', label="Dev loss")
# plt.xlabel("epoch", fontsize='large')
# plt.ylabel("loss", fontsize='large')
# plt.legend(fontsize='x-large')
# plt.savefig('fw-loss2.png')
# plt.show()
