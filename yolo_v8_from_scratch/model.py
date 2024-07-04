# https://www.youtube.com/watch?v=Grir6TZbc1M
import torch
import torch.nn as nn

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        batch_norm: bool = True,
        **kwargs
    ) -> None:
        super.__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            bias=not batch_norm,
            **kwargs
        )  # when batch norm is activated, we shouldnt use the bias!
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_batch_norm = batch_norm

    def forward(self, x):
        if self.use_batch_norm:
            return self.leaky_relu(
                self.batch_norm(self.conv(x))
            )  # for the scaled prediction, we DO NOT want to use batch norm and leaky relu - ONLY conv
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, input_channels: int, use_residual: bool = True, num_repetitions: int = 1
    ):
        super.__init__()
        self.layers = nn.ModuleList()
        for rep in num_repetitions:
            self.layers += [
                CNNBlock(
                    input_channels=input_channels,
                    output_channels=input_channels // 2,
                    kernel_size=1,
                ),
                CNNBlock(
                    input_channels=input_channels // 2,
                    output_channels=input_channels,
                    kernel_size=3,
                    padding=1,
                ),
            ]

        self.use_residual = use_residual
        self.num_repetitions = num_repetitions

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                # THE RESIDUAL IS ADDED HERE AS X - this means a skip-conn which passes from input to input
                x = layer(x) + x
            else:
                x = layer(x)


class ScalePrediction(nn.Module):
    def __init__(self, input_channels: int, amount_of_classes: int) -> None:
        super().__init__()
        self.prediction = nn.Sequential(
            CNNBlock(
                input_channels=input_channels,
                output_channels=2 * input_channels,  # DOUBLE the channels
                kernel_size=3,
                padding=1,
            ),
            CNNBlock(
                input_channels=2 * input_channels,  # going from the doubled channels...
                output_channels=(amount_of_classes + 5)
                * 3,  # convert channels to 3 anchor boxes = all classes, p0, d_x, d_y, d_w, d_h,
                batch_norm=False,
                kernel_size=1,
            ),
        )
        self.amount_of_classes = amount_of_classes

        def forward(self, x):
            return (
                self.prediction(x)
                .reshape(
                    x.shape[0], 3, self.amount_of_classes + 5, x.shape[2], x.shape[3]
                )
                .permute(0, 1, 3, 4, 2)
            )  # number samples in batch x 3 anchors x 13 px width x 13 height px x 5+amount_of_classes (13px: scale=1:1)


class YOLOv3(nn.Module):
    # This class puts all the block types together
    def __init__(self, input_channels: int = 3, amount_of_classes: int = 20) -> None:
        super().__init__()
        self.amount_of_classes = amount_of_classes
        self.input_channels = input_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        pass

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        input_channels = self.input_channels

        for module in config:
            if isinstance(module, tuple):
                output_channels, kernel_size, stride = (
                    module  # this is how the data is arranged in the confg file
                )
                layers.append(
                    CNNBlock(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                input_channels = output_channels  # this one changes the next input_channels to the current output_channel count
            elif isinstance(module, list):
                num_repetitions = module[1]
                layers += [
                    ResidualBlock(
                        input_channels=input_channels,
                        num_repetitions=num_repetitions,
                    ))
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(
                            input_channels=input_channels,
                            use_residual=False, num_repetitions=1,
                        ),
                        CNNBlock(
                            input_channels=input_channels, output_channels=input_channels//2, kernel_size=1,
                        ),
                        ScalePrediction(input_channels=input_channels//2, amount_of_classes=self.amount_of_classes)
                    ]
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    input_channels = input_channels * 3
                    # upsampling layers are ALSO for concatenation, which is why we use 3 here - 
                    # concatenation comes RIGHT AFTER upsampling
                    