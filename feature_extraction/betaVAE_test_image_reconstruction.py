import torch
from torch.utils.data import Subset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

from betaVAE import VAutoencoder
from betaVAE_train import (LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN)

matplotlib.use("TkAgg")

NUM_DIGITS = 10
NUM_IMAGES = 20


def one_hot_target(label):
    return F.one_hot(torch.tensor(label), num_classes=NUM_DIGITS).float()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        target_transform=one_hot_target,
    )

    model = VAutoencoder(
        latent_dim=LATENT_DIM,
        enc_hidden=ENCODER_HIDDEN,
        dec_hidden=DECODER_HIDDEN,
    )
    state_dict = torch.load("betaVAE.pth")
    model.load_state_dict(state_dict)

    subset_indices = list(range(NUM_IMAGES))  # Get indices for the first NUM_DP datapoints
    subset = Subset(test_dataset, subset_indices)

    data, targets = [], []

    for i in range(len(subset)):
        x, y = subset[i]
        data.append(x)
        targets.append(y)

    data_tensor = torch.stack(data)
    targets_tensor = torch.stack(targets)

    reconstructions, mu, logvar = model.forward(data_tensor, targets_tensor)

    images = torch.cat([data_tensor, reconstructions], dim=0)

    grid = make_grid(images, nrow=NUM_IMAGES)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
