# romnet - ffnet
import math
import torch
from torch import nn

class MinmaxNormalizer(nn.Module):
    def __init__(self, mins, dif):
        super().__init__()

        self.register_buffer('mins', torch.tensor(mins)),
        self.register_buffer('dif', torch.tensor(dif))

    def forward(self, x):
        return (x - self.mins) / self.dif 

class StddevNormalizer(nn.Module):
    def __init__(self, means, deviations):
        super().__init__()

        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('deviations', torch.tensor(deviations))

    def forward(self, x):
        return (x - self.means) / self.deviations

class FfNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = {**config}

        self.build()


    def make_fc_layer(self, f_in, f_out):
        fc = nn.Linear(f_in, f_out)
        nn.init.kaiming_uniform_(fc.weight)
        fc_act = nn.ELU()
        return nn.Sequential(
            fc,
            fc_act
        )


    def build(self):
        fc_layers = [
            self.make_fc_layer(self.config['n_params'], self.config['n_neurons']),
        ]

        for i in range(self.config['n_layers']):
            fc_layers.append(
                self.make_fc_layer(self.config['n_neurons'], self.config['n_neurons'])
            )

        fc_layers.append(
            self.make_fc_layer(self.config['n_neurons'], self.config['n'])
        )

        self.u_n = nn.Sequential(*fc_layers)
        
        #print('\n\nFully connected layers: ')
        #print(self.u_n)

        if self.config['scaling'] == 'minmax':
            print(f'\nUsing minmax scaling of input data with parameters:\n\tMin: {self.config["mins"]},\n\tMax-min: {self.config["dif"]}')
            self.normalizer = MinmaxNormalizer(self.config['mins'], self.config['dif'])

        elif self.config['scaling'] == 'standard' or self.config['scaling'] == 'standart': 
            print(f'\nUsing standard deviation normalization of input data with parameters:\n\tMean: {self.config["means"]}\n\tDeviation: {self.config["deviations"]}')
            self.normalizer = StddevNormalizer(self.config['means'], self.config['deviations'])

        elif self.config['scaling'] == 'batchnorm':
            print('\nUsing nn.BatchNorm1d for normalization of input data')
            self.normalizer = nn.BatchNorm1d(self.config['n_params'])

        else:
            print('\nUsing no normalization for input data.')
            self.normalizer = lambda x: x


    def forward(self, x):            
        x = self.normalizer(x)
        
        return self.u_n(x)

