# dataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import  methods.modeling.config
import pickle

from torch.utils.data import Dataset, DataLoader

class VectorDataset(Dataset):

    def __init__(
            self, 
            path_inp_data, 
            path_coef_data, 
            path_out_data, 
            type_vector
            ):
        super(VectorDataset, self).__init__()

        with open(path_inp_data, 'rb') as handle:
            self.inp_data = pickle.load(handle)

        with open(path_out_data, 'rb') as handle:
            self.out_data = pickle.load(handle)

        with open(path_coef_data, 'rb') as handle:
            self.coef_data = pickle.load(handle)

        self.type_vector = type_vector

    def __len__(self):
        return self.inp_data[self.type_vector].shape[0]
    
    def __getitem__(self, index):
        input_vector = self.inp_data[self.type_vector][index]
        output_vector = self.out_data[self.type_vector][index]
        coef_vector = self.coef_data[index]
        #output_vector = torch.unsqueeze(output_vector, 0)

        return input_vector, coef_vector, output_vector

def draw_plot(vector):
    x = np.array([i for i in range(len(vector))]) 
    plt.plot(x, vector, color ="red")
    plt.show()

def test():
    dataset = VectorDataset(
        config.PATH_INP_DATA, 
        config.PATH_COEF_DATA,
        config.PATH_OUTPUT_DATA, 
        'train'
        )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for inp, out, coef in loader:
        print(inp.shape)
        print(out.shape)
        print(coef.shape)
        break


if __name__ == "__main__":
    test()
