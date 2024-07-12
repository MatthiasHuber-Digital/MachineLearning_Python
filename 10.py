# resnet
import torch
import torch.nn as nn

from torch import Tensor 
from torchinfo import summary

class Block(nn.Module):

    def __init__(
            self, 
            d_main: int, 
            d_hidden: int, 
            bias_first: bool = True, 
            bias_second: bool = True, 
            dropout_first: float = 0.5, 
            dropout_second: float = 0.5,
            skip_connection: bool = True,
                 ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_main)
        self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
        self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
        self.activation = nn.ReLU()
        self.dropout_first = nn.Dropout(dropout_first)
        self.dropout_second = nn.Dropout(dropout_second)
        self.skip_connection = skip_connection

    def forward(self, x: Tensor) -> Tensor:
        x_input = x
        x = self.normalization(x)
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.dropout_first(x)
        x = self.linear_second(x)
        x = self.dropout_second(x)
        if self.skip_connection:
            x = x_input + x
        
        return x

class Head(nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            bias: bool = True,
                ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_in)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(d_in, d_out, bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x

class ResNetModel(nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            n_blocks: int,
            d_main: int,
            d_hidden: int,
            dropout_first: float,
            dropout_second: float
            )-> None:
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
            Block(
            d_main=d_main,
            d_hidden=d_hidden,
            bias_first=True,
            bias_second=True,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            skip_connection=True,
            )
            for _ in range(n_blocks)
            ]
        )
        self.head = Head(
            d_in=d_main,
            d_out=d_out,
            bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        mins = [0.0050277801105811, 0.0050256579608133, -9.972816424840127, -9.96617244994268, -19.98712209821679, -2.976937633221969, -9.485251730305608, 2.08030640002573] 
        dif = [0.014959358676726701, 0.0149007663927914, 19.935366458934727, 19.886152766249147, 39.896472378680485, 5.968450060912874, 18.938547224934446, 37.804011961340436]           

        #x = (x - torch.tensor(mins, device='cuda'))/torch.tensor(dif, device='cuda')
  
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def example():   
    model = ResNetModel(
        d_in=5,
        d_out=25,
        n_blocks=4,
        d_main=128,
        d_hidden=256,
        dropout_first=0.2,
        dropout_second=0,
    )
    summary(model, input_size=(10, 5), depth=4, col_names=["input_size", "output_size", "num_params", "kernel_size"])

if __name__ == "__main__":
    example()
