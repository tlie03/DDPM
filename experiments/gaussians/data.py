import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor


def create_gaussian_ring(
        n_samples: int,
        num_modes: int = 8,
        radius: float = 4.0,
        std: float = 0.2,
        random_seed: int = 42,
) -> Tensor:
    """
    Generates a mixture of num_modes equally weighted gaussians whose centers are evenly spaced
    on a circle around the origin (0, 0).
    :param n_samples: total number of points
    :param num_modes: number of gaussians
    :param radius: distance of the centers from the origin
    :param std: standard deviation of every gaussian
    :param random_seed: seed for the random number generator for reproducibility
    :return: tensor of shape (n_samples, 2), dtype=torch.float32
    """
    rng = np.random.default_rng(random_seed)
    angles = 2 * np.pi * np.arange(num_modes) / num_modes
    centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # assign the points to the modes as evenly as possible
    modes = np.arange(n_samples) % num_modes
    x = centers[modes] + rng.normal(scale=std, size=(n_samples, 2))
    return torch.from_numpy(x).float()


def points(n_samples: int, cfg: DictConfig) -> Tensor:
    """
    :param n_samples: number of points
    :param cfg: data section of the experiment config
    :return: points of shape (n_samples, 2)
    """
    return create_gaussian_ring(
        n_samples=n_samples,
        num_modes=cfg.num_modes,
        radius=cfg.radius,
        std=cfg.std,
        random_seed=cfg.random_seed,
    )
