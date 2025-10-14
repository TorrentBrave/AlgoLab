import torch

torch.manual_seed(42)

class Op(object):
    def __init__(self):
        pass
    def __call__(self, *inputs):
        return self.forward(*inputs)
    def forward(self, *inputs):
        raise NotImplementedError
    def backward(self, *outputs_grads):
        raise NotImplementedError
    
class Linear(Op):
    def __init__(self, input_size):
        """
        输入：
           - input_size:模型要处理的数据特征向量长度
           
        torch.rand 是均匀采样
        torch.randn 是标准正态分布
        """
        super(Linear, self).__init__()
        
        self.input_size = input_size
        
        # 模型参数
        self.params = {}
        self.params['w'] = torch.randn((self.input_size, 1), dtype=torch.float32)
        self.params['b'] = torch.zeros(1, dtype=torch.float32)

    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        """
        输入：
           - X: tensor, shape=[N,D]
           注意这里的X矩阵是由N个x向量的转置拼接成的,与原教材行向量表示方式不一致
        输出：
           - y_pred: tensor, shape=[N]
        """
        N, D = X.shape

        if self.input_size==0:
            return torch.full((N, 1), fill_value=self.params['b'])
        
        # 输入数据维度合法性验证
        assert D==self.input_size

        y_pred = torch.matmul(X, self.params['w']) + self.params['b']

        return y_pred

if __name__ == "__main__":
    input_size = 3
    N = 2
    X = torch.randn((N, input_size), dtype=torch.float32) # 生成 2个维度为3 的数据
    model = Linear(input_size)
    y_pred = model(X)
    print("y_pred:", y_pred) # 输出结果的个数也是 2个