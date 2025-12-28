import torch
import math


def linear_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alpha_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-4, 0.999)
