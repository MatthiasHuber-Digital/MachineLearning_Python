# autoencover_conv
import torch.nn as nn

from torchinfo import summary

class Autoencoder_conv(nn.Module):

    def __init__(
            self, 
            size: list, 
            kernel_size: int,
            dropout: float
            ) -> None:
        
        super(Autoencoder_conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=size[0], out_channels=size[1], kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=size[1], out_channels=size[2], kernel_size=9, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=size[2], out_channels=size[3], kernel_size=9, stride=2),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=size[3], out_channels=size[4], kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=4),
            nn.ConvTranspose1d(in_channels=size[4], out_channels=size[5], kernel_size=3, stride=2, padding=5),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=6),
            nn.ConvTranspose1d(in_channels=size[5], out_channels=size[6], kernel_size=3, stride=2, padding=4),
            nn.LeakyReLU(),
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=size[0], out_channels=size[1], kernel_size=3, stride=8, padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(in_channels=size[1], out_channels=size[2], kernel_size=3, stride=8, padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(in_channels=size[2], out_channels=size[3], kernel_size=3, stride=8),
        #     nn.LeakyReLU(),
        # )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=size[3], out_channels=size[4], kernel_size=3, stride=8, output_padding=1),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(in_channels=size[4], out_channels=size[5], kernel_size=3, stride=8, padding=3),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(in_channels=size[5], out_channels=size[6], kernel_size=3, stride=9, padding=5),
        #     nn.LeakyReLU(),
        # )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

def example():   
    model = Autoencoder_conv(size=[1, 32, 16, 10, 16, 32, 1], kernel_size=9, dropout=0)
    summary(model, input_size=(200, 1, 1901), depth=4, col_names=["input_size", "output_size", "num_params", "kernel_size"])

if __name__ == "__main__":
    example()
