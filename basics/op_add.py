from op import Op

class Add(Op):
    def __init__(self):
        super(Add, self).__init__()
    def __call__(self, x, y):
        return self.forward(x, y)
    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x + y
        return outputs
    def backward(self, grads):
        grads_x = grads * 1
        grads_y = grads * 1
        return grads_x, grads_y

if __name__ == "__main__":
    x = 1
    y = 4
    add_op = Add()
    z = add_op(x, y)
    grads_x, grads_y = add_op.backward(grads=1)
    print("x's grad is: ", grads_x)
    print("y's grad is: ", grads_y)