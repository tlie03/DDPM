from typing import List

import torch
from torch import nn


NUM_DIGITS = 10


class Encoder(torch.nn.Module):

    def __init__(self, hidden_dim: List[int], latent_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = torch.nn.ModuleList([nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(784, hidden_dim[0]), nn.ReLU()])
        for h1, h2 in zip(hidden_dim, hidden_dim[1:]):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], latent_dim))
        layers.append(nn.Tanh())

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_dim: List[int], latent_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        layers = torch.nn.ModuleList([nn.Linear(latent_dim + NUM_DIGITS, hidden_dim[0]), nn.ReLU()])
        for h1, h2 in zip(hidden_dim, hidden_dim[1:]):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], 784))
        layers.append(nn.Sigmoid())
        layers.append(nn.Unflatten(dim=1, unflattened_size=[1, 28, 28]))

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_latent_dim(self):
        return self.latent_dim


class Autoencoder(torch.nn.Module):

    def __init__(self, enc_hidden: List[int], dec_hidden: List[int], latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(hidden_dim=enc_hidden, latent_dim=latent_dim)
        self.decoder = Decoder(hidden_dim=dec_hidden, latent_dim=latent_dim)

    def get_decoder(self):
        return self.decoder

    def forward(self, x, y):
        latents = self.encoder.forward(x)
        x = torch.concat([y, latents], dim=1)
        return self.decoder.forward(x)
