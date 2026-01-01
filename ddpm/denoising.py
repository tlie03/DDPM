from typing import List
from abc import ABC
import torch
from torch import nn

from .util import sinusoidal_encoding


class Denoise(torch.nn.Module, ABC):

    def __init__(
            self,
            x_shape: torch.Size,
            y_shape: torch.Size,
            *args, **kwargs
    ):
        """
        abstract class of a denoiser that predicts the noise that was applied to an input.
        :param x_shape: shape of the noise input
        :param y_shape: shape of conditional y
        :param device: device on which the model will run
        """
        super().__init__(*args, **kwargs)
        self.x_shape = x_shape
        self.y_shape = y_shape

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of x, y and t values and predicts the total noise that was applied to x.
        :param x: tensor of shape(BATCH_SIZE, self.x_shape)
        :param y: tensor of shape(BATCH_SIZE, self.y_shape)
        :param t: tensor of shape(BATCH_SIZE) where all values must be integers in [0, T),
        where T is the length of the beta schedule.
        :return: tensor of shape (BATCH_SIZE, self.x_shape)
        """
        pass


class DnnDenoise(Denoise):

    def __init__(
            self,
            x_shape: torch.Size,
            y_shape: torch.Size,
            time_encoding_channels: int = 32,
            hidden_dims: List[int] = None,
            hidden_activations: List[str] = None,
            *args, **kwargs
    ):
        """
        Implements an MLP based denoiser. The conditional y and time embedding are concatenated
        to the input layer and to each hidden layer. The time variable is encoded using
        sinusoidal positional encoding and is then embedded through a small MLP of size
        [time_encoding_channels] * 3 with no activation in the last layer.
        :param x_shape: shape of input x
        :param y_shape: shape of conditional y
        :param time_encoding_channels: number of channels used for the sinusoidal time encoding
        and also the size of the time embedding vector
        :param hidden_dims: List that defines the number and sizes of the hidden dimensions.
        :param hidden_activations: List that defines the activation functions used in each hidden layer.
        Can be any torch.nn activation function as a string ("ReLU", "SiLU", etc.)
        """
        super().__init__(
            x_shape,
            y_shape,
            *args,
            **kwargs
        )

        # check and prepare hyperparams
        if hidden_dims is None:
            hidden_dims = []
        if hidden_activations is None:
            hidden_activations = []
        assert len(hidden_dims) == len(hidden_activations)
        self.dims = [x_shape.numel()] + hidden_dims + [x_shape.numel()]
        self.activations = hidden_activations + ["Identity"]

        self.time_encoding_channels = time_encoding_channels
        # maps positional time encoding to an embedding vector of the same length
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(time_encoding_channels, time_encoding_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(time_encoding_channels, time_encoding_channels)
        )

        self.layers = nn.ModuleList()
        for in_dim, out_dim, activation in zip(self.dims, self.dims[1::], self.activations):
            layer = DnnDenoiseLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                y_dim=y_shape.numel(),
                t_dim=time_encoding_channels,
                activation_function=activation,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        For docs look in upper class
        """
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        t = sinusoidal_encoding(t=t, dim=self.time_encoding_channels)
        t = self.time_embedding(t)

        for layer in self.layers:
            x = layer(x, y, t)

        x = x.unflatten(dim=1, sizes=self.x_shape[1:])
        return x


class DnnDenoiseLayer(torch.nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            y_dim: int,
            t_dim: int,
            activation_function: str,
            *args, **kwargs
    ):
        """
        A layer in the denoising DNN. Maps a vector of size in_dim + y_dim + t_dim to a vector of size
        out_dim using a linear transformation and an activation function.
        :param in_dim: length of output vector of previous layer
        :param out_dim: length of output vector of this layer
        :param y_dim: length of conditional vector
        :param t_dim: length of time embedding vector
        :param activation_function: can be any torch.nn activation function as a string ("ReLU", "SiLU", etc.)
        """
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(t_dim + y_dim + in_dim, out_dim)
        try:
            self.activation = getattr(nn, activation_function)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation_function}' not supported by torch")

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        concatenates t, y, x along dim=1 and feeds it into linear layer, then applies activation function
        :param x: tensor of shape (BATCH_SIZE, self.in_dim)
        :param y: tensor of shape (BATCH_SIZE, self.y_dim)
        :param t: tensor of shape (BATCH_SIZE, self.t_dim)
        :return: activation(linear(concat([t, y, x], dim=1))) tensor of shape (BATCH_SIZE, self.out_dim)
        """
        input = torch.cat((t, y, x), dim=1)
        input = self.linear(input)
        return self.activation(input)
