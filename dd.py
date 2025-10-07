import torch

def main():
    ndim_1_tensor = torch.tensor([2.0, 3.0, 4.0])
    print(ndim_1_tensor)
    
    ndim_2_tensor = torch.tensor([[2.0, 3.0, 4.0],
                                  [1.0, 6.0, 7.0]])
    print(ndim_2_tensor)

    ndim_3_tensor = torch.tensor([[[2.0, 3.0, 4.0],
                                   [1.0, 6.0, 7.0]],
                                  [[2.0, 3.0, 4.0],
                                   [1.0, 6.0, 7.0]]])
    print(ndim_3_tensor)

if __name__ == "__main__":
    main()