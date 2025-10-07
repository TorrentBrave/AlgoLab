import torch

def main():
    zeros_tensor = torch.zeros([3, 2])
    print(zeros_tensor)
    
    ones_tensor = torch.ones([3, 2])
    print(ones_tensor)

    full_tensor = torch.full([3, 2], 10)
    print(full_tensor)
if __name__ == "__main__":
    main()