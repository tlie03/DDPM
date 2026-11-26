"""
Animates how a fixed set of gaussians data points moves from data to noise (forward process)
and back to data (reverse process with the trained MLP). Saves one gif per process to the output folder.
Run from the project root: python -m experiments.gaussians.animate [hydra overrides]
"""
import hydra
from omegaconf import DictConfig

from experiments import toy2d
from experiments.gaussians.data import points


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    toy2d.animate(cfg, data=points(cfg.animate.num_samples, cfg.data))


if __name__ == "__main__":
    main()
