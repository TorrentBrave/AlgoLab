import torch

def main():
    ndim_3_Tensor = torch.tensor([[[1, 2, 3, 4, 5],
                                [6, 7, 8, 9, 10]],

                                [[11, 12, 13, 14, 15],
                                [16, 17, 18, 19, 20]],
                                
                                [[21, 22, 23, 24, 25],
                                [26, 27, 28, 29, 30]]])
    
    ndim_1_Tensor = ndim_3_Tensor.reshape([-1])
    print(ndim_1_Tensor.shape)
    print(ndim_1_Tensor)

    ndim_new_3_Tensor = ndim_3_Tensor.reshape([0, 5, 2])
    print(ndim_new_3_Tensor.shape)
    print(ndim_new_3_Tensor)

if __name__ == "__main__":
    main()
