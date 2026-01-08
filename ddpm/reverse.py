import torch

from ddpm import Denoise, BetaSchedule

EPS = 1e-8


def reverse_step(x_t, y, t, model: Denoise, beta_schedule: BetaSchedule) -> torch.Tensor:
    """
    Computes x_t-1 based on x_t, y, and t using a denoiser and the corresponding beta_schedule.
    x_t, y, t, model and beta_schedule must all be on the same device!!!

    x_t shape: (BATCH_SIZE, model.x_shape)
    y shape: (BATCH_SIZE, model.y_shape)
    t shape: (BATCH_SIZE)
    """
    noise = beta_schedule.sigma(t) * torch.randn_like(x_t, device=x_t.device)
    drift = model(x=x_t, y=y, t=t)
    drift = (
                    (1 - beta_schedule.alpha(t)) / (beta_schedule.sqrt_one_minus_alpha_bar(t) + EPS)
            ) * drift

    scaling = 1 / (beta_schedule.sqrt_alpha(t) + EPS)
    return scaling * (x_t - drift) * noise
