from basics.op import Op

class Multiply(Op):
    def __init__(self):
        super(Multiply, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)
    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x * y
        return outputs
    def backward(self, grads):
        grads_x = grads * self.y
        grads_y = grads * self.x
        return grads_x, grads_y

if __name__ == "__main__":
    x = 1
    y = 4
    mul_op = Multiply()
    z = mul_op(x, y)
    grads_x, grads_y = mul_op.backward(grads=2)
    print("x's grad is: ", grads_x)
    print("y's grad is: ", grads_y)