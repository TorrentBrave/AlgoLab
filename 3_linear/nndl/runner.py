# 相比RunnerV1, RunnerV2类在训练中使用梯度下降进行网络优化, 模型训练过程中计算在训练和验证集的损失及评估指标
import torch
torch.manual_seed(42)

class Runner(object):
    def __init__(self, model, optimizer, metric, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        # 记录训练过程中的评价指标变化情况
        self.train_scores = []
        self.dev_scores = []
        # 记录训练过程中的损失函数变化情况
        self.train_loss = []
        self.dev_loss = []
    
    def train(self, train_set, dev_set, **kwargs):
        # 传入训练轮数, 如果没有传入值则默认 0
        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 100)
        save_path = kwargs.get("save_path", "best_model.pth")
        # 梯度打印函数
        print_grads = kwargs.get("print_grads", None)
        # 记录全局最优指标
        best_score = 0
        # 记录全局最优指标
        for epoch in range(num_epochs):
            X, y = train_set
            # 获取模型预测
            logits = self.model(X) 
            # 计算交叉损失
            trn_loss = self.loss_fn(logits, y).item() # .item() 是把里面的数值提取出来
            self.train_loss.append(trn_loss)
            # 计算评价指标
            trn_score = self.metric(logits, y).item()
            self.train_scores.append(trn_score)
            # 计算参数梯度
            self.model.backward(y)
            if print_grads is not None:
                # 打印每一层的梯度
                print_grads(self.model)
            # 更新模型参数
            self.optimizer.step()
            dev_score, dev_loss = self.evaluate(dev_set)
            if dev_score > best_score:
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
            if epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}, loss: {trn_loss}, score: {trn_score}")
                print(f"[Dev] epoch: {epoch}, loss: {dev_loss}, score: {dev_score}")
    def evaluate(self, data_set):
        X, y = data_set
        # 计算模型输出
        logits = self.model(X)
        # 计算损失函数
        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        # 计算评价指标
        score = self.metric(logits, y).item()
        self.dev_scores.append(score)
        return score, loss
    def predict(self, X):
        return self.model(X)
    def save_model(self, save_path):
        torch.save(self.model.params, save_path)
    def load_model(self, model_path):
        self.model.params = torch.load(model_path)