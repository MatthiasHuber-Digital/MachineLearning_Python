# https://www.youtube.com/watch?v=Grir6TZbc1M
import torch
import torch.nn as nn
from model_config import config


class CNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        batch_norm: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
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
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repetitions):
            self.layers += [
                nn.Sequential(
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
                )
            ]

        self.use_residual = use_residual
        self.num_repetitions = num_repetitions

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                # THE RESIDUAL IS ADDED HERE AS X - this means a skip-conn which passes from input to input
                x = x + layer(x)
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
            .reshape(x.shape[0], 3, self.amount_of_classes + 5, x.shape[2], x.shape[3])
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
        predicted_outputs = []
        route_connections = (
            []
        )  # are sort of the skip connections used for concatenation

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                """This means that in predict_outputs there will be 3 entries,
                each of them being able to make predictions on its own scale.
                This again means that they all can process the data to their necessary scale.
                first entry: coarse resolution
                second entry: medium resolution
                third entry: fine resolution
                """
                predicted_outputs.append(layer(x))
                continue  # skip all the remaining code for this loop, as we only want to predict with this layer!

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repetitions == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat(
                    [x, route_connections[-1]], dim=1
                )  # concatenate the skip connections with the current data
                route_connections.pop()

        return predicted_outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        input_channels = self.input_channels

        for module in config:
            print("Line in config: \t" + str(module))
            if isinstance(module, tuple):
                output_channels, kernel_size, stride = module  # this is how the data is arranged in the confg file
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
                layers.append(ResidualBlock(
                        input_channels=input_channels,
                        num_repetitions=num_repetitions))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(
                            input_channels=input_channels,
                            use_residual=False,
                            num_repetitions=1,
                        ),
                        CNNBlock(
                            input_channels=input_channels,
                            output_channels=input_channels // 2,
                            kernel_size=1,
                        ),
                        ScalePrediction(
                            input_channels=input_channels // 2,
                            amount_of_classes=self.amount_of_classes,
                        ),
                    ]
                    input_channels = input_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    input_channels = input_channels * 3
                    # upsampling layers are ALSO for concatenation, which is why we use 3 here -
                    # concatenation comes RIGHT AFTER upsampling

        return layers


if __name__ == "__main__":
    amount_of_classes = 20
    image_length_px = 416  # v1: 448, v3: 416
    model = YOLOv3(amount_of_classes=amount_of_classes)
    x = torch.randn(
        (2, 3, image_length_px, image_length_px)
    )  # using two random tensors with 3 channels and the quadratic image size 416x416
    prediction_outputs = model(x)

    assert model(x)[0].shape == (
        2,
        3,
        image_length_px // 32,
        image_length_px // 32,
        amount_of_classes + 5,
    )
    assert model(x)[1].shape == (
        2,
        3,
        image_length_px // 16,
        image_length_px // 16,
        amount_of_classes + 5,
    )
    assert model(x)[2].shape == (
        2,
        3,
        image_length_px // 8,
        image_length_px // 8,
        amount_of_classes + 5,
    )
    print("Successfully trained the model!")
