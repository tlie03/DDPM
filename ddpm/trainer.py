import torch
from torch.utils.data import DataLoader, Dataset

from denoise import Denoise


class Trainer:

    def __init__(
            self,
            model: Denoise,
            model_path: str,
            train_dataset: Dataset,
            test_dataset: Dataset,
            beta_schedule: torch.Tensor,
            batch_size: int,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            loss_function=torch.nn.MSELoss,
    ):

        self.model = model
        self.model_path = model_path
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.beta_schedule = beta_schedule
        self.alphas = 1 - beta_schedule
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self):
        for i, epoch in enumerate(range(self.epochs)):
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                batch_size, channel, height, width = x.size()
                time_steps = torch.randint(0, self.beta_schedule.shape[0], (batch_size,))
                noise = torch.randn(batch_size, channel, height, width)

                alpha_bars = self.alpha_bars[time_steps]
                x_factor = torch.sqrt(alpha_bars)[:, None, None, None]
                noise_factor = torch.sqrt(1 - alpha_bars)[:, None, None, None]

                x = x_factor * x + noise_factor * noise

                noise_model = self.model.forward(x=x, y=y, t=time_steps)
                loss = self.loss_function(noise_model, noise)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                total_loss = 0
                for x, y in self.test_loader:
                    batch_size, channel, height, width = x.size()
                    time_steps = torch.randint(1, len(self.beta_schedule) + 1, (batch_size,))
                    noise = torch.randn(batch_size, channel, height, width)

                    alpha_bars = time_steps.clone().apply_(self.alpha_bar_t)

                    x_factor = torch.sqrt(alpha_bars)
                    x_factor = x_factor[:, None, None, None]

                    noise_factor = 1 - alpha_bars
                    noise_factor = torch.sqrt(noise_factor)
                    noise_factor = noise_factor[:, None, None, None]

                    x = x_factor * x + noise_factor * noise

                    noise_model = self.model.forward(x=x, y=y, t=time_steps)
                    loss = self.loss_function(noise_model, noise)
                    total_loss += loss.item()

                total_loss /= len(self.test_loader)
                print(f'Epoch {i + 1}, Loss: {total_loss:.4f}')

        torch.save(self.model.state_dict(), self.model_path)


