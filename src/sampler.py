from typing import Iterator, Tuple

import torch

from src.ddpm import BetaSchedule, reverse, reverse_step
from src.nn import NoiseEstimate


class Sampler:
    """
    Generates samples with a trained noise estimate by running the reverse process
    (algorithm 2 of https://arxiv.org/abs/2006.11239).
    """

    def __init__(
            self,
            model: NoiseEstimate,
            beta_schedule: BetaSchedule,
            device: torch.device = torch.device("cpu"),
    ):
        """
        :param model: trained noise estimate, is put into evaluation mode
        :param beta_schedule: beta schedule, should be the one the model was trained with
        :param device: device used for sampling
        """
        self.device = device
        self.model = model.to(device)
        self.model.train(False)
        self.beta_schedule = beta_schedule.to(device)

    def noise(self, num_samples: int) -> torch.Tensor:
        """
        :return: standard normal noise x_T of shape (num_samples, model.x_shape)
        """
        return torch.randn(size=(num_samples,) + tuple(self.model.x_shape), device=self.device)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, x_T: torch.Tensor | None = None) -> torch.Tensor:
        """
        Runs the entire reverse process.
        :param y: conditionals of shape (BATCH_SIZE, model.y_shape), one sample is generated per entry
        :param x_T: optional starting noise of shape (BATCH_SIZE, model.x_shape)
        :return: samples x_0 of shape (BATCH_SIZE, model.x_shape)
        """
        y, x_T = self._prepare(y, x_T)
        return reverse(x_T, y, self.model, self.beta_schedule)

    def trajectory(self, y: torch.Tensor, x_T: torch.Tensor | None = None) -> Iterator[Tuple[int, torch.Tensor]]:
        """
        Runs the reverse process step by step, e.g. for animations.
        :param y: conditionals of shape (BATCH_SIZE, model.y_shape)
        :param x_T: optional starting noise of shape (BATCH_SIZE, model.x_shape)
        :return: iterator over (t, x) for t = T-1, ..., 0, where x is the result of the reverse step at t
        """
        y, x = self._prepare(y, x_T)
        batch_size = x.size(0)
        for t in range(len(self.beta_schedule) - 1, -1, -1):
            t_batch = torch.full((batch_size,), fill_value=t, device=self.device)
            with torch.no_grad():
                x = reverse_step(x, y, t_batch, self.model, self.beta_schedule)
            yield t, x

    def _prepare(self, y: torch.Tensor, x_T: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.to(self.device)
        x_T = self.noise(y.size(0)) if x_T is None else x_T.to(self.device)
        return y, x_T
