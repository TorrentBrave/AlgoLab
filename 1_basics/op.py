class Op(object):
    """
    __call__ 让操作对象 Add() 可以像函数一样被调用,并自动触发前向计算(forward),使代码更简洁、符合直觉

    self是类方法的第一个参数, 是类实例(对象) 自身的引用. 是类方法中的第一个参数, 用于访问该实例的属性(对象)
    """
    def __init__(self):
        pass
    def __call__(self, *inputs):
        return self.forward(*inputs)
    def forward(self, *inputs):
        raise NotImplementedError
    def backward(self, *outputs_grads):
        raise NotImplementedError