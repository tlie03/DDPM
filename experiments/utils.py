"""
Helpers shared by the experiments to turn a hydra config into objects.
"""
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.ddpm import BetaSchedule


def device(cfg: DictConfig) -> torch.device:
    """:return: the device given by cfg.device, auto picks cuda if available"""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def beta_schedule(cfg: DictConfig) -> BetaSchedule:
    """
    :return: the beta schedule given by cfg.diffusion: schedule name, length T and optionally further
    arguments of the schedule, e.g. max_beta for the cosine schedule
    """
    kwargs = {key: value for key, value in cfg.diffusion.items() if key not in ("schedule", "T")}
    return getattr(BetaSchedule, cfg.diffusion.schedule)(cfg.diffusion.T, **kwargs)


def path(p: str) -> Path:
    """:return: p as absolute path, relative paths are resolved against the directory the script was started in"""
    return Path(to_absolute_path(p))
