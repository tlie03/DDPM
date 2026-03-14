from typing import Tuple
from torch import Tensor
import torch
import numpy as np


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

    Args:
        n_samples (int): Total number of points to generate (must be even).
        start_angle (float): Initial angle for the spiral in radians.
        n_turns (float): Number of full rotations for the spiral.
        noise_level (float): The amount of random noise to add to the coordinates.
        random_seed (int): Seed for the random number generator for reproducibility.
        max_radius: (float): Maximum radius of the spirals
    Returns:
        tuple: (X, y)
               X (torch.Tensor): The feature tensor of shape (n_samples, 2), dtype=torch.float32.
               y (torch.Tensor): The label tensor of shape (n_samples,), dtype=torch.long.
    """
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be an even number for two balanced spirals.")

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

    # 3. Combine and add noise (still in NumPy)
    x_np = np.concatenate([x0_np, x1_np], axis=0)
    y_np = np.concatenate([y0_np, y1_np], axis=0)
    x_np += np.random.normal(scale=noise_level, size=x_np.shape)

    # 4. Convert NumPy arrays to PyTorch Tensors
    # X is features (float32) and y is labels (long/int64)
    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).long()

    return x_torch, y_torch
