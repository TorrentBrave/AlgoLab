from basics.op import Op
import math

class Exponential(Op):
    def __init__(self):
        super(Exponential, self).__init__()
    def forward(self, x):
        self.x = x
        outputs = math.exp(x)
        return outputs
    def backward(self, grads):
        grads = grads * math.exp(self.x)
        return grads