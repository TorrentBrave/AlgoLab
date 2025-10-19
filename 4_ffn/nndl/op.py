import torch
import torch.nn.functional as F
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

class Linear(Op):
    def __init__(self, input_dim, output_dim, name, weight_init=torch.randn, bias_init=torch.zeros):
        """
        输入:
            - input_dim: 输入向量的维度
            - output_dim: 输出向量的维度
            - name: 算子名字
            - weight_init: 权重初始化方法, 默认使用'torch.randn' 进行标准正态分布初始化
            - bias_init: 偏置初始化方法, 默认使用全 0 初始化
        """
        super(Linear, self).__init__()
        self.params = {}
        self.params['W'] = weight_init(input_dim, output_dim)
        self.params['b'] = bias_init(1, output_dim)
        
        self.inputs = None
        self.grads = {}

        self.name = name
    def forward(self, inputs):
        """
        前向传播
        输入：
           - inputs: [N, input_dim], N 是样本数量
        输出：
           - outputs: 预测值, shape=[N, output_dim]
        """
        self.inputs = inputs
        outputs = torch.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs
    def backward(self, grads):
        """
        反向传播
        grads: 上游梯度, shape=[N, output_size]
        返回: 当前层输入的梯度, shape=[N, input_size]
        """
        # 对权重和偏置求梯度
        self.grads['W'] = torch.matmul(self.inputs.T, grads)   # shape: [input_size, output_size]
        self.grads['b'] = torch.sum(grads, dim=0, keepdim=True)  # shape: [1, output_size]

        # 返回的是对输入 X 的梯度
        return torch.matmul(grads, self.params['W'].T)

class Logistic(Op):
    """
    把数值映射到 [0,1] 区间
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None # Logistic 层无参数

    def forward(self, inputs):
        """
        前向传播
        输入：
           - inputs: [N, C], N 是样本数量
        输出：
           - outputs: 预测值, shape=[N, C]
        """
        self.inputs = inputs
        outputs = 1.0 / (1.0 + torch.exp(-inputs))
        self.outputs = outputs
        return outputs
    def backward(self, grads):
        """
        使用 Logistic 作为激活函数, 所以需要 backward, 如果
        反向传播
        输入：
           - grads: 上游梯度, shape=[N, C], shape 与 inputs 相同
           - 最终输出对于outputs的梯度
        返回： 输出对输入的梯度
           - 最终输出对于inputs的梯度
        """
        grad_inputs = grads * self.outputs * (1.0 - self.outputs)
        return grad_inputs

class Model_MLP_L2(Op):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        整个网络, 实现完整的两层神经网络的前向和反向计算
        输入:
            - input_dim: 输入维度
            - hidden_dim: 隐藏层神经元数量
            - output_dim: 输出维度
        fc: Fully Connected
        fn: Function
        """
        self.fc1 = Linear(input_dim, hidden_dim, name="fc1")
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_dim, output_dim, name="fc2")
        self.act_fn2 = Logistic()

        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2]
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        """
        输入:
            - X: tensor, 输入数据, shape=[N, input_dim]
        输出:
            - a2: 预测值, [N, output_dim]
        """
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        return a2
    def backward(self, loss_grad_a2):
        """
        loss_grad_a2 就是 损失函数对网络最终输出 a2 的梯度
        """
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)
    
class BinaryCrossEntropyLoss(Op):
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None

        self.model = model
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
        # 防止 log(0) 出现数值溢出
        # eps = 1e-8 加在 log() 中
        loss = -1. / self.num * (
            torch.matmul(self.labels.T, torch.log(self.predicts)) +
            torch.matmul ((1 - self.labels).T, torch.log(1 - self.predicts))
        )

        loss = torch.squeeze(loss, dim=1)
        return loss
    def backward(self):
        # eps = 1e-8
        grad = -1. / self.num * (
            self.labels / self.predicts -
            (1 - self.labels) / (1 - self.predicts)
        )
        # 梯度反向传播
        self.model.backward(grad)

if __name__ == '__main__':
    torch.manual_seed(42)
    model = Model_MLP_L2(input_dim=5, hidden_dim=10, output_dim=1)
    # 随机生成1条长度为5的数据
    X = torch.randn(1,5)
    result = model(X) 
    print(result)