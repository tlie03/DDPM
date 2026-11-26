from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from .base import NoiseEstimate
from .utils import sinusoidal_encoding


class Unet(NoiseEstimate):

    def __init__(
            self,
            x_shape: torch.Size,
            y_shape: torch.Size,
            num_blocks: int,
            num_channels_init: int,
            time_encoding_channels: int,
            activation_function: str = "ReLU",
            *args,
            **kwargs
    ):
        """
        :param x_shape: shape of x (channels, height, width)
        :param y_shape: shape of y
        :param num_blocks: number of unet blocks used, must be at least 1
        :param num_channels_init: number of inner channels of the first block. Will double with every block
        :param time_encoding_channels: dimension of the sinusoidal time encoding
        :param activation_function: activation function used, can be any string that
        is a valid activation function class in torch.nn. Defaults to ReLU
        """
        super().__init__(x_shape, y_shape, *args, **kwargs)
        assert num_blocks >= 1

        y_dim = 1
        for dim in y_shape:
            y_dim *= dim

        x_channels, _, _ = x_shape
        self.encoder = ChainedConv2d(
            channels=[x_channels, num_channels_init],
            activation_function="Identity"
        )
        # no activation function, to enable negative values
        self.decoder = ChainedConv2d(
            channels=[2 * num_channels_init, num_channels_init, x_channels],
            activation_function="Identity"
        )

        current_block = None
        for i in range(num_blocks-1, -1, -1):
            block = UNetBlock(
                channels_in=(2**i) * num_channels_init,
                activation_function=activation_function,
                sub_block=current_block,
                t_dim=time_encoding_channels,
                y_dim=y_dim,
            )
            current_block = block

        self.entry_block: UNetBlock = current_block
        self.time_encoding_channels = time_encoding_channels

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch Tensor of shape (batch_size, channels, height, width)
        :param y: torch Tensor of shape (batch_size, *self.y_shape)
        :param t: torch Tensor of shape (batch_size)
        :return: torch Tensor of shape (batch_size, channels, height, width)
        """
        x = self.encoder(x)
        y = torch.flatten(input=y, start_dim=1)
        t = sinusoidal_encoding(t, self.time_encoding_channels)

        x_pred = self.entry_block(x=x, y=y, t=t)
        return self.decoder(x_pred)


class UNetBlock(torch.nn.Module):

    def __init__(
            self,
            channels_in: int,
            sub_block: UNetBlock | None,
            t_dim: int,
            y_dim: int,
            activation_function: str = "ReLU",
            *args,
            **kwargs
    ):
        """
        Creates an Unet Block.
        :param channels_in: number of channels that are passed to this block
        :param sub_block: the sub U-net block. None if the block is the base layer
        :param t_dim: dimension of time encoding
        :param y_dim: dimension of conditional encoding
        :param activation_function: activation function used, can be any string that
        is a valid activation function class in torch.nn. Defaults to ReLU
        """
        super().__init__(*args, **kwargs)

        self.embed_t = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.ReLU(),
            nn.Linear(t_dim, t_dim),
            nn.ReLU(),
            nn.Linear(t_dim, channels_in)
        )
        self.embed_y = nn.Sequential(
            nn.Linear(y_dim, y_dim),
            nn.ReLU(),
            nn.Linear(y_dim, y_dim),
            nn.ReLU(),
            nn.Linear(y_dim, channels_in)
        )

        self.encoder = ChainedConv2d(
            channels=[channels_in, 2*channels_in, 2*channels_in],
            activation_function=activation_function
        )
        if sub_block is not None:
            self.decoder = ChainedConv2d(
                channels=[4*channels_in, 2*channels_in, 2*channels_in],
                activation_function=activation_function
            )
            self.down_scale = nn.Conv2d(
                in_channels=2*channels_in,
                out_channels=2*channels_in,
                kernel_size=2,
                stride=2,
                padding=1
            )
            # needed to reduce channel size after interpolation up-scaling
            self.upscale_conv = ChainedConv2d(
                channels=[4*channels_in, 2*channels_in],
                activation_function=activation_function
            )
        self.sub_block: UNetBlock = sub_block

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        input shape and output shape are exactly the same.
        :param x: torch tensor of shape (batch_size, channels_in, height, width)
        :param t: torch tensor of shape (batch_size, t_dim)
        :param y: torch tensor of shape (batch_size, y_dim)
        :return: torch tensor of shape (batch_size, 2*channels_in, height, width)
        """
        t_embed = self.embed_t(t)[:, :, None, None]
        y_embed = self.embed_y(y)[:, :, None, None]
        x = x + t_embed + y_embed

        x = self.encoder(x)
        if self.sub_block is None:
            return x
        else:
            _, _, height, width = x.size()
            x_sub = self.down_scale(x)
            x_sub = self.sub_block(x=x_sub, y=y, t=t)
            x_sub = nn.functional.interpolate(x_sub, size=(height, width), mode='nearest')
            x_sub = self.upscale_conv(x_sub)
            x = torch.cat([x, x_sub], dim=1)  # concatenate along channel dimension
            return self.decoder(x)


class ChainedConv2d(torch.nn.Module):
    """
    Implements chained 2d convolutions with different channel dimensions,
    that keeps the data size. Kernel size is 3x3, stride is 1 and padding is 1.
    """
    KERNEL_SIZE = 3
    PADDING = 1
    STRIDE = 1

    def __init__(self, channels: List[int], activation_function: str = "ReLU", *args, **kwargs):
        """
        :param channels: dimensions channels, length of list-1 is
        the number of convolutions that will be applied. First element must be input dimension and last
        element determines the output dimension of this module
        :param activation_function: the activation function used, can be any string that
        corresponds to a pytorch activation function class. Defaults to ReLU
        """
        super().__init__(*args, **kwargs)
        assert len(channels) >= 2
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.KERNEL_SIZE,
                        stride=self.STRIDE,
                        padding=self.PADDING
                    ),
                    getattr(torch.nn, activation_function)()
                ) for in_c, out_c in zip(channels, channels[1:])
            ]
        )

    def forward(self, x) -> torch.Tensor:
        """
        :param x: torch tensor of shape (batch_size, channels[0], height, width)
        :return: torch tensor of shape (batch_size, channels[-1], height, width)
        """
        for layer in self.layers:
            x = layer(x)
        return x
