class Runner(object):
    """
    应用机器学习方法的流程:
        - 数据
        - 模型
        - 损失函数
        - 优化器(参数学习)
        
        - 模型训练
        - 模型评价
        - 模型预测
    """
    def __init__(self, model, optimizer, loss_fn, metric):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
    
    def train(self, train_dataset, dev_dataset=None, **kwargs):
        pass
    def evaluate(self, data_set, **kwargs):
        pass
    def predict(self, x, **kwargs):
        pass
    def save_model(self, save_path):
        pass
    def load_model(self, model_path):
        pass