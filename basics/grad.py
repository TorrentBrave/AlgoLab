import torch

if __name__ == "__main__":
    a = torch.tensor(2.0)
    b = torch.tensor(5.0, requires_grad=True)
    c = a * b
    c.retain_grad()
    c.backward()
    print("Tensor a's grad is: {}".format(a.grad))
    print("Tensor b's grad is: {}".format(b.grad))
    print("Tensor c's grad is: {}".format(c.grad))