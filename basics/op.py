class Op(object):
    def __init__(self):
        pass
    def __call__(self, *inputs):
        return self.forward(*inputs)
    def forward(self, *inputs):
        raise NotImplementedError
    def backward(self, *outputs_grads):
        raise NotImplementedError