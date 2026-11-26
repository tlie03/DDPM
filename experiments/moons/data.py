import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor


def create_moons(
        n_samples: int,
        noise_level: float = 0.05,
        scale: float = 2.0,
        random_seed: int = 42,
) -> Tensor:
    """
    Generates two interleaving half circles ("moons") centered around the origin (0, 0).
    :param n_samples: total number of points, split evenly between both moons
    :param noise_level: standard deviation of the gaussian noise added to the points (before scaling)
    :param scale: factor the centered points are multiplied with
    :param random_seed: seed for the random number generator for reproducibility
    :return: tensor of shape (n_samples, 2), dtype=torch.float32
    """
    rng = np.random.default_rng(random_seed)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    angles_upper = np.linspace(0, np.pi, n_upper)
    angles_lower = np.linspace(0, np.pi, n_lower)
    upper = np.stack([np.cos(angles_upper), np.sin(angles_upper)], axis=1)
    lower = np.stack([1 - np.cos(angles_lower), 0.5 - np.sin(angles_lower)], axis=1)

    x = np.concatenate([upper, lower], axis=0)
    x += rng.normal(scale=noise_level, size=x.shape)
    # the moons span [-1, 2] x [-0.5, 1], move their center to the origin
    x = (x - np.array([0.5, 0.25])) * scale
    return torch.from_numpy(x).float()


def points(n_samples: int, cfg: DictConfig) -> Tensor:
    """
    :param n_samples: number of points
    :param cfg: data section of the experiment config
    :return: moon points of shape (n_samples, 2)
    """
    return create_moons(
        n_samples=n_samples,
        noise_level=cfg.noise_level,
        scale=cfg.scale,
        random_seed=cfg.random_seed,
    )
