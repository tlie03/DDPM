import torch

from .beta_schedule import BetaSchedule
from src.nn import NoiseEstimate


EPS = 1e-8


def reverse(x_T, y, model: NoiseEstimate, beta_schedule: BetaSchedule):
    """
    Computes the entire reverse process.
    x_T, y, model, beta_schedule must all be on the same device !!!

    x_T: shape (BATCH_SIZE, model.x_shape). Is actually x_T-1 when using the notation that aligns
    with the notation used throughout the other modules.
    y: shape (BATCH_SIZE, model.y_shape)
    model: Denoise model
    beta_schedule: BetaSchedule that is used for denoising. Should ideally be the same schedule that was used
    to train the model
    """
    T = len(beta_schedule)
    batch_size = x_T.size(0)
    for t in range(T - 1, -1, -1):
        x_T = reverse_step(x_T, y, torch.full((batch_size,), fill_value=t, device=x_T.device), model, beta_schedule)
    return x_T


def reverse_step(x_t, y, t, model: NoiseEstimate, beta_schedule: BetaSchedule) -> torch.Tensor:
    """
    Computes x_t-1 based on x_t, y, and t using a denoiser and the corresponding beta_schedule.
    x_t, y, t, model and beta_schedule must be on the same device!!!

    x_t: shape (BATCH_SIZE, model.x_shape)
    y: shape (BATCH_SIZE, model.y_shape)
    t: shape (BATCH_SIZE), can be different values within the range of [0, T-1]
    """
    drift = model(x=x_t, y=y, t=t)

    # broadcast t to match the shape of x_prev, so that beta schedule returns correct shape
    batch_size = x_t.size(0)
    x_ndim = x_t.ndim - 1
    new_shape = (batch_size,) + (1,) * x_ndim
    t = t.view(new_shape)

    noise = beta_schedule.sigma(t) * torch.randn_like(x_t, device=x_t.device)
    drift = (
                    (1 - beta_schedule.alpha(t)) / (beta_schedule.sqrt_one_minus_alpha_bar(t) + EPS)
            ) * drift

    scaling = 1 / (beta_schedule.sqrt_alpha(t) + EPS)
    return scaling * (x_t - drift) + noise
