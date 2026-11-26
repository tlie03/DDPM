import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.ddpm import BetaSchedule, forward
from src.nn import NoiseEstimate


class Trainer:
    """
    Implements a ddpm trainer. Trains a noise estimate by sampling a random t for every sample,
    noising the sample with the forward process and minimizing the MSE between the predicted
    and the actual noise (algorithm 1 of https://arxiv.org/abs/2006.11239).
    """

    def __init__(
            self,
            model: NoiseEstimate,
            data: torch.utils.data.Dataset,
            beta_schedule: BetaSchedule,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            path: str | Path,
            device: torch.device = torch.device("cpu"),
    ):
        """
        :param model: noise estimate that is trained
        :param data: dataset that returns (x, y) pairs
        :param beta_schedule: beta schedule used for the forward process
        :param epochs: number of epochs
        :param batch_size: batch size
        :param learning_rate: learning rate of the Adam optimizer
        :param path: path the model state dict is saved to after every epoch
        :param device: device used for training
        """
        self.device = device
        self.model = model.to(device)
        self.data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
        self.beta_schedule = beta_schedule.to(device)
        self.epochs = epochs
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self) -> List[float]:
        """
        Trains the model for self.epochs epochs and saves it to self.path after every epoch.
        :return: list containing the mean loss of every epoch
        """
        self.model.train(True)
        losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                epoch_loss += self._train_step(x, y)

            epoch_loss /= len(self.data_loader)
            losses.append(epoch_loss)
            torch.save(self.model.state_dict(), self.path)
            print(f"loss for epoch {epoch}/{self.epochs}: {epoch_loss}")

        self.model.train(False)
        return losses

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Performs a single optimization step on a batch.
        :return: loss of the batch
        """
        self.optimizer.zero_grad()

        t = torch.randint(low=0, high=len(self.beta_schedule), size=(x.size(0),), device=self.device)
        x_t, noise = forward(x, t, self.beta_schedule)

        noise_pred = self.model(x=x_t, y=y, t=t)
        loss = self.loss_function(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        return loss.item()
