from abc import ABC, abstractmethod
import torch


class NoiseEstimate(torch.nn.Module, ABC):

    def __init__(
            self,
            x_shape: torch.Size,
            y_shape: torch.Size,
            *args, **kwargs
    ):
        """
        abstract class of a denoiser that predicts the noise that was applied to an input.
        :param x_shape: shape of the noise input, without batch dimension
        :param y_shape: shape of conditional y, without batch dimension
        """
        super().__init__(*args, **kwargs)
        self.x_shape = x_shape
        self.y_shape = y_shape

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of x, y and t values and predicts the total noise that was applied to x.
        Expects that all x, y, t are on self.device
        :param x: tensor of shape(BATCH_SIZE, self.x_shape)
        :param y: tensor of shape(BATCH_SIZE, self.y_shape)
        :param t: tensor of shape(BATCH_SIZE) where all values must be integers in [0, T),
        where T is the length of the beta schedule.
        :return: tensor of shape (BATCH_SIZE, self.x_shape)
        """
        pass


