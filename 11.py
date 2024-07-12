# autoencoder
import torch.nn as nn

from torch import Tensor 
from torchinfo import summary

class Block(nn.Module):

    def __init__(
            self, 
            d_in: int, 
            d_out: int, 
            bias: bool = True, 
            activation = 'ReLU',
            dropout: float = 0, 
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias)
        self.activation = nn.LeakyReLU() if activation == 'ReLU' else nn.Sigmoid()
        self.dropout = nn.Dropout(0) if dropout == 0 else nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class Autoencoder(nn.Module):

    def __init__(
            self, 
            size: list, 
            dropout: float
            ) -> None:
        
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            Block(d_in=size[0], d_out=size[1]), 
            Block(d_in=size[1], d_out=size[2]) 
        )

        self.decoder = nn.Sequential(
            Block(d_in=size[2], d_out=size[3]),
            Block(d_in=size[3], d_out=size[4]), 
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

def example():   
    model = Autoencoder([1901, 500, 25, 500, 1901], dropout=0)
    summary(model, input_size=(200, 1901), depth=4, col_names=["input_size", "output_size", "num_params", "kernel_size"])

if __name__ == "__main__":
    example()
