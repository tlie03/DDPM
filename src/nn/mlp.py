import torch

from .base import NoiseEstimate
from .utils import sinusoidal_encoding


class MLP(NoiseEstimate):

    def __init__(
            self,
            x_shape: torch.Size,
            y_shape: torch.Size,
            num_classes: int,
            hidden_dim: int,
            num_layers: int,
            time_encoding_channels: int,
            *args, **kwargs
    ):
        """

        :param x_shape: shape of input x
        :param y_shape: shape of conditional y must be (BATCH_SIZE)
        :param hidden_dim: dimension of hidden state
        :param num_layers: number of layers
        :param time_encoding_channels: dimension of PE for time variable
        """
        super().__init__(
            x_shape,
            y_shape,
            *args,
            **kwargs
        )
        self.hidden_dim = hidden_dim
        self.time_encoding_channel = time_encoding_channels

        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(time_encoding_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.label_embedding = torch.nn.Embedding(num_classes, hidden_dim)

        self.input_embedding = torch.nn.Sequential(
            torch.nn.Linear(x_shape.numel(), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                # final linear layer lets the residual update be negative as well
                torch.nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, x_shape.numel()),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        first embeds all variables, then sums them to get a single hidden state. The hidden
        state is then passed through the hidden layers using residual connections and the last
        hidden state is then linearly projected to the output vector.
        returns: Tensor of shape (batch, *x.shape)
        """
        t_encoding = sinusoidal_encoding(t=t, dim=self.time_encoding_channel)
        t_embedding = self.time_embedding(t_encoding)

        y_embedding = self.label_embedding(y)

        x = x.flatten(start_dim=1)
        x_embedding = self.input_embedding(x)

        h = x_embedding + y_embedding + t_embedding
        for layer in self.layers:
            h = h + layer(h)

        out = self.output_layer(h)
        return out.unflatten(dim=1, sizes=self.x_shape)
