"""
Trains the MLP noise estimate on the moons dataset.
Run from the project root: python -m experiments.moons.train [hydra overrides]
"""
import hydra
from omegaconf import DictConfig

from experiments import toy2d
from experiments.moons.data import points


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    toy2d.train(cfg, points(cfg.data.num_samples, cfg.data))


if __name__ == "__main__":
    main()
