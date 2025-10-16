import torch
import torch.nn.functional as F
from activation import softmax

torch.manual_seed(10) # 设置随机种子

class Op:
    """
    框架式设计: 无法预先知道子类的 forward() 需要多少参数
    """
    def __init__(self):
        pass
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
    def forward(self, *input, **kwargs):
        raise NotImplementedError
    def backward(self, *input, **kwargs):
        raise NotImplementedError

def logistic(x):
    """
    定义: Logistic函数
    """
    return 1 / (1 + torch.exp(-x))



class model_LR(Op):
    def __init__(self, input_dim):
        super(model_LR, self).__init__()

        self.params = {}
        # self.params['w'] = torch.zeros((input_dim, 1), dtype=torch.float32) # 线性层的权重参数初始化为0
        self.params['w'] = torch.normal(mean=0, std=0.01, size=(input_dim, 1), dtype=torch.float32) * 0.01 # 线性层的权重参数全部随机高斯分布
        # 乘以 0.01 是为了 让权重初始化更小、训练更稳定, 防止梯度爆炸或消失，是一种简易但有效的初始化策略
        self.params['b'] = torch.zeros(1, dtype=torch.float32)

        self.grads = {}
        self.X = None
        self.outputs = None

    def __call__(self, inputs):
        return self.forward(inputs)
    def forward(self, inputs):
        """
        输入:
            - inputs: shape=[N,D], N 样本数量, D 特征维度
        输出:
            - outputs: 预测标签为1的概率, shape=[N,1]
        """
        self.X = inputs
        # 线性计算
        score = torch.matmul(inputs, self.params['w']) + self.params['b']
        # Logistic 函数
        self.outputs = logistic(score)
        return self.outputs

    def backward(self, labels):
        """
        输入:
            - labels: 真实标签, shape=[N, 1]
        """
        N = labels.shape[0]
        # 计算偏导数
        self.grads['w'] = -1 / N * torch.matmul(self.X.T, (labels - self.outputs))
        self.grads['b'] = -1 / N * torch.sum(labels - self.outputs)

class BinaryCrossEntropyLoss(Op):
    def __init__(self):
        self.predicts = None
        self.labels = None
        self.num = None
    
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        输入:
            - predicts: 预测值, shape=[N, 1], N为样本数量
            - labels: 真实标签, shape=[N, 1]
        输出:
            - 损失值: shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = -1. / self.num * (torch.matmul(self.labels.T, torch.log(self.predicts)) + torch.matmul((1 - self.labels.T), torch.log(1 - self.predicts)))
        loss = torch.squeeze(loss, axis=1)
        return loss


if __name__ == "__main__":
    # ------------------------------------------------
    # Logistic function - plot
    # ------------------------------------------------
    """
    import matplotlib.pyplot as plt
    x = torch.linspace(-10, 10, 1000)
    plt.figure()
    plt.plot(x.tolist(), logistic(x).tolist(), color="#e4007f", label="Logistic Function")
    # 设置坐标轴
    ax = plt.gca()
    # 取消右侧和上侧坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # 设置默认的x轴和y轴方向
    ax.xaxis.set_ticks_position('bottom') 
    ax.yaxis.set_ticks_position('left')
    # 设置坐标原点为(0,0)
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    # 添加图例
    plt.legend()
    plt.savefig('linear-logistic.png')
    # plt.show()
    """
    # -------------------------------------------------
    # Logistic function - generate data
    # -------------------------------------------------
    torch.manual_seed(42)

    # 随机生成 3条长度为 4的数据
    inputs = torch.randn(3,4)
    print('Input is:', inputs)

    # 实例化模型
    model = model_LR(4)
    outputs = model(inputs)
    print('Output is:', outputs)
    # --------------------------------------------------
    # 交叉熵损失
    # --------------------------------------------------
    labels = torch.ones(3,1) # 生成长度为3, 值为1的标签数据
    bce_loss = BinaryCrossEntropyLoss()
    print("bce_loss:", bce_loss(outputs, labels))
