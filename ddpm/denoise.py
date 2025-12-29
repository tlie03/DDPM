from typing import List
import torch
from torch import nn


class Denoise(torch.nn.Module):

    def __init__(
            self,
            beta_schedule: torch.Tensor,
            x_dim: int,
            y_dim: int,
            time_encoding: str = "scalar",
            time_encoding_channels: int = 10,
            *args, **kwargs
    ):
        """
        Class from which all DDPMs will inherit.
        A DDPM predicts noise added to x_0 based on x_t, t and perhaps a conditional vector y.
        :param beta_schedule: the beta schedule used
        :param x_dim: dimension of the input data x
        :param y_dim: dimension of conditional vector y
        :param time_encoding: what type of encoding is used for t, either "scalar" or "positional"
        :param time_encoding_channels: dimension of time encoding if time_encoding is "positional"
        "scalar" interpolates t between 0 and 1
        "positional" uses positional encoding with channels time_encoding_dim
        """
        super().__init__(*args, **kwargs)
        assert time_encoding in [
            "scalar",
            "positional",
        ]
        self.beta_schedule = beta_schedule
        self.T = beta_schedule.shape[0]
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.t_dim = 1 if time_encoding == "scalar" else time_encoding_channels
        self.time_encoding = time_encoding

    def get_beta_schedule(self) -> torch.Tensor:
        return self.beta_schedule.clone()

    def encode_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        calculates encoding for t.
        :param t: tensor of shape (BATCH_SIZE) with values between 0 and T-1
        :return: tensor whose shape depends on the encoding method used
        """
        if self.time_encoding == "scalar":
            return self.encode_t_scalar(t)
        elif self.time_encoding == "positional":
            return self.encode_t_positional(t)
        else:
            raise NotImplementedError(f"implementation for time_encoding {self.time_encoding} does not jet exist."
                                      f"Supported encodings are \"scalar\" and \"positional\"")

    def encode_t_scalar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encodes t into a scalar that is interpolated between 0 and 1.
        :param t: vector of shape (BATCH_SIZE) with values between 0 and T-1
        :return: scalar tensor of shape (BATCH_SIZE) with values between 0 and 1
        """
        return t / self.T

    def encode_t_positional(self, t: torch.Tensor) -> torch.Tensor:
        """
        Creates a positional encoding for t, with channel size self.t_dim
        :param t: vector of shape (BATCH_SIZE) with values between 0 and T-1
        :return: PE tensor of shape (BATCH_SIZE, self.t_dim)
        """
        idx = torch.arange(0, self.t_dim).unsqueeze(0).repeat(t.size(0), 1)
        pos = t.unsqueeze(0).T.repeat(1, self.t_dim)
        return torch.where(
            0 == idx % 2,
            torch.sin(
                pos / (torch.pow(10000, idx / self.t_dim))
            ),
            torch.cos(
                pos / (torch.pow(10000, (idx - 1) / self.t_dim))
            ),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predicts the total noise that was used to generate x at timestep t in the diffusion process.
        y is used as a conditional vector.
        :param x: tensor of shape (BATCH_SIZE, height, width) and height * width = self.x_dim
        :param y: tensor of shape (BATCH_SIZE, self.y_dim), with any values
        :param t: tensor of shape (BATCH_SIZE) with values between 0 and T-1
        :return:
        """
        pass


class DnnDenoise(Denoise):

    def __init__(
            self,
            beta_schedule: torch.Tensor,
            x_dim: int,
            y_dim: int,
            time_encoding: str = "scalar",
            time_encoding_channels: int = 10,
            hidden_dims: List[int] = None,
            hidden_activations: List[str] = None,
            *args, **kwargs
    ):
        """
        A deep neural network denoiser. with following architecture.
        - Input dim will be x_dim + y_dim + time_encoding_dim
          x and y will be normalized at input
        - Output dim is x_dim and sigmoid is applied to output
        - hidden_dims will be concatenated with time_encoding and conditional variable c,
          resulting in y_dim + time_encoding_dim + hidden_dim for the hidden dims.
          [y_dim, time_encoding_dim, hidden_dims[i]] -> hidden_dims[i+1]
        - hidden_activations is the activation function to be used in the hidden layers
        For the docs of the other params see upper class.
        """
        super().__init__(
            beta_schedule,
            x_dim,
            y_dim,
            time_encoding,
            time_encoding_channels,
            *args, **kwargs)

        # check and prepare hyperparams
        if hidden_dims is None:
            hidden_dims = []
        if hidden_activations is None:
            hidden_activations = []
        assert len(hidden_dims) == len(hidden_activations)
        self.dims = [x_dim] + hidden_dims + [x_dim]
        self.activations = hidden_activations + ["Identity"]

        self.layers = nn.ModuleList()
        for in_dim, out_dim, activation in zip(self.dims, self.dims[1::], self.activations):
            layer = DnnDenoiseLayer(
                in_dim=in_dim,
                y_dim=self.y_dim,
                t_dim=self.t_dim,
                out_dim=out_dim,
                activation_function=activation,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        For docs look in upper class
        """
        x_shape = x.shape
        x = x.flatten(start_dim=1)
        t = self.encode_t(t)

        for layer in self.layers:
            x = layer(x, y, t)

        x = x.unflatten(dim=1, sizes=x_shape[1:])
        return x


class DnnDenoiseLayer(torch.nn.Module):

    def __init__(
            self,
            in_dim: int,
            y_dim: int,
            t_dim: int,
            out_dim: int,
            activation_function: str,
            *args, **kwargs
    ):
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
