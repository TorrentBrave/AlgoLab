import torch

def main():
    ones_tensor  = torch.ones([2, 3, 4, 5])
    print(ones_tensor.ndim)
    print(ones_tensor.shape)
    print(ones_tensor.shape[-1])
    print(ones_tensor.size())
    print(ones_tensor.numel())
    print(ones_tensor)
    
if __name__ == "__main__":
    main()
