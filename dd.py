import torch

def main():
    arange_tensor  = torch.arange(start=1, end=5, step=1)
    print(arange_tensor)

    # linspace_tensor = torch.linspace(start=1, end=5, step=5)
    # print(linspace_tensor)
if __name__ == "__main__":
    main()