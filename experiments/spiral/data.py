from typing import Tuple
from torch import Tensor
import torch
import numpy as np
from omegaconf import DictConfig


def create_spiral_dataset_torch(
        n_samples: int,
        start_angle: float = 0,
        n_turns: float = 1.5,
        noise_level: float = 0.00,
        random_seed: int = 42,
        max_radius: float = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Generates a 2D dataset forming two interwoven spirals around the origin (0, 0)
    and returns them as PyTorch Tensors.
    :param n_samples: Total number of points to generate (must be even).
    :param start_angle: Initial angle for the spiral in radians.
    :param n_turns: Number of full rotations for the spiral.
    :param noise_level: The amount of random noise to add to the coordinates.
    :param random_seed: Seed for the random number generator for reproducibility.
    :param max_radius: Maximum radius of the spirals
    :return tuple: (X, Y)
        X (torch.Tensor): The feature tensor of shape (n_samples, 2), dtype=torch.float32.
        Y (torch.Tensor): The label tensor of shape (n_samples,), dtype=torch.long.
    """
    np.random.seed(random_seed)
    n_points_per_spiral = n_samples // 2

    # 1. Generate radius and angle for the first spiral (Class 0)
    r0 = np.linspace(0.1, max_radius, n_points_per_spiral)
    t0 = np.linspace(start_angle, n_turns * 2 * np.pi, n_points_per_spiral)
    x0_np = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)
    y0_np = np.zeros(n_points_per_spiral, dtype=int)

    # 2. Generate the second spiral (Class 1) - offset by pi
    r1 = np.linspace(0.1, max_radius, n_points_per_spiral)
    t1 = np.linspace(start_angle + np.pi, n_turns * 2 * np.pi + np.pi, n_points_per_spiral)
    x1_np = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)
    y1_np = np.ones(n_points_per_spiral, dtype=int)

    x_np = np.concatenate([x0_np, x1_np], axis=0)
    y_np = np.concatenate([y0_np, y1_np], axis=0)
    x_np += np.random.normal(scale=noise_level, size=x_np.shape)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).long()

    return x_torch, y_torch


def points(n_samples: int, cfg: DictConfig) -> Tensor:
    """
    :param n_samples: number of points
    :param cfg: data section of the experiment config
    :return: spiral points of shape (n_samples, 2)
    """
    x, _ = create_spiral_dataset_torch(
        n_samples=n_samples,
        start_angle=cfg.start_angle,
        n_turns=cfg.n_turns,
        noise_level=cfg.noise_level,
        random_seed=cfg.random_seed,
        max_radius=cfg.max_radius,
    )
    return x
