from __future__ import annotations

import torch


class BetaSchedule:
    """
    Class used to handle the create and handle the beta schedule used by a ddpm.
    On initialization, it prepares the following constant tensors that are all needed either during
    training or sampling:
    - betas
    - alphas
    - alphas_bar
    - sqrt_alphas
    - sqrt_alphas_bar
    - sqrt_one_minus_alphas_bar
    - sigmas
    All of these tensors have shape (T).
    Note that indices go from 0 to T-1 instead of 1 to T,
    as in the notation of the landmark DDPM paper https://arxiv.org/abs/2006.11239.
    """
    EPS = 1e-8

    def __init__(self, betas: torch.Tensor):
        self._T = betas.shape[0]
        self._betas = betas
        self._alphas = 1 - betas
        self._alphas_bar = torch.cumprod(self._alphas, dim=0)
        self._sqrt_alphas = torch.sqrt(self._alphas)
        self._sqrt_alphas_bar = torch.sqrt(self._alphas_bar)
        self._sqrt_one_minus_alphas_bar = torch.sqrt(1 - self._alphas_bar)

        # set value at new index 0 to 1 so that self.sigmas[0] = 0
        alphas_bar_shifted = torch.cat([torch.ones(1), self._alphas_bar[:-1]], dim=0)
        # add EPS in denominator for numerical stability
        self._sigmas = torch.sqrt(
            ((1 - alphas_bar_shifted) / (1 - self._alphas_bar + self.EPS)) * self._betas
        )

    def __len__(self) -> int:
        return self._T

    def beta(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._betas[t]

    def alpha(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._alphas[t]

    def alpha_bar(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._alphas_bar[t]

    def sqrt_alpha(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._sqrt_alphas[t]

    def sqrt_alpha_bar(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._sqrt_alphas_bar[t]

    def sqrt_one_minus_alpha_bar(self, t: int | torch.Tensor) -> torch.Tensor:
        return self._sqrt_one_minus_alphas_bar[t]

    def sigma(self, t: int | torch.Tensor) -> torch.Tensor:
        """sigma(0)=0"""
        return self._sigmas[t]

    @staticmethod
    def linear(length: int, start: float = 1e-4, end: float = 0.02) -> BetaSchedule:
        betas = torch.linspace(start, end, length, dtype=torch.float64).float()
        return BetaSchedule(betas=betas)

    @staticmethod
    def cosine(length: int, s: float = 0.008, max_beta=0.999) -> BetaSchedule:
        x = torch.linspace(0, length, length + 1, dtype=torch.float64)
        alphas_bar = torch.cos((((x / length) + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clip(betas, 0, max_beta).float()
        return BetaSchedule(betas=betas)



