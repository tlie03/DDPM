import torch

from .beta_schedule import BetaSchedule


def forward(x, t, beta_schedule: BetaSchedule):
    """
    Computes x_t based on x (sample without noise), t and a beta schedule.
    x, t and beta_schedule must be on the same device!!!

    x: shape (BATCH_SIZE, *)
    t: shape (BATCH_SIZE), can be different values within the range of [0, T-1]
    return: x_t, epsilon
    """
    noise = torch.randn_like(x, device=x.device)
    x_t = beta_schedule.sqrt_alpha_bar(t) * x + beta_schedule.sqrt_one_minus_alpha_bar(t) * noise
    return x_t, noise


def forward_step(x_prev, t, beta_schedule: BetaSchedule):
    """
        Computes x_t based on x_t-1 (x_prev), t and a beta schedule.
        x_prev, t and beta_schedule must be on the same device!!!

        x_prev: shape (BATCH_SIZE, *)
        t: shape (BATCH_SIZE), can be different values within the range of [0, T-1]
        return: x_t+1, epsilon
    """
    noise = torch.randn_like(x_prev, device=x_prev.device)
    x_t = torch.sqrt(1 - beta_schedule.beta(t)) * x_prev + torch.sqrt(beta_schedule.beta(t)) * noise
    return x_t, noise
