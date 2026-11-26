"""
Training and sampling shared by all 2D toy experiments (spiral, moons, gaussians).
Every toy experiment learns its point cloud unconditionally with an MLP noise estimate:
all points get the label 0. An experiment only has to provide its config and its data points.
"""
import numpy as np
import torch
import matplotlib
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ddpm import forward_step
from src.nn import MLP
from src.sampler import Sampler
from src.trainer import Trainer
from experiments import utils


def build_model(cfg: DictConfig) -> MLP:
    return MLP(
        x_shape=torch.Size([2]),
        y_shape=torch.Size([]),
        num_classes=1,  # unconditional: every point has label 0
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        time_encoding_channels=cfg.model.time_encoding_channels,
    )


def train(cfg: DictConfig, points: torch.Tensor):
    """
    Trains the MLP on the given points and saves it to cfg.model_path.
    :param points: tensor of shape (N, 2)
    """
    torch.manual_seed(cfg.seed)
    model_path = utils.path(cfg.model_path)
    trainer = Trainer(
        model=build_model(cfg),
        data=TensorDataset(points, torch.zeros(points.size(0), dtype=torch.long)),
        beta_schedule=utils.beta_schedule(cfg),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        path=model_path,
        device=utils.device(cfg),
    )
    trainer.train()
    print(f"model saved to {model_path}")


def load_sampler(cfg: DictConfig) -> Sampler:
    """:return: sampler using the trained MLP stored at cfg.model_path"""
    model_path = utils.path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"no trained model at {model_path}, run the train script of '{cfg.name}' first")
    dev = utils.device(cfg)
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    return Sampler(model, utils.beta_schedule(cfg), device=dev)


def sample(cfg: DictConfig, reference: torch.Tensor):
    """
    Samples cfg.sample.num_samples points with the trained MLP, prints how close they are to the data
    and saves a plot of data and samples side by side to cfg.sample.plot_path.
    :param reference: real data points of shape (N, 2) the samples are compared to
    """
    torch.manual_seed(cfg.seed)
    sampler = load_sampler(cfg)

    samples = sampler.sample(torch.zeros(cfg.sample.num_samples, dtype=torch.long)).cpu()

    nearest = torch.cdist(samples, reference).min(dim=1).values
    print(f"mean distance of a sample to the nearest data point: {nearest.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    limit = 1.5 * reference.abs().max().item()
    for ax, points, title in zip(axes, (reference, samples), ("data", "samples (MLP)")):
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.2)
        ax.set_title(f"{cfg.name}: {title}")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
    fig.tight_layout()

    plot_path = utils.path(cfg.sample.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    print(f"plot saved to {plot_path}")


def animate(cfg: DictConfig, data: torch.Tensor):
    """
    Animates how a fixed set of data points is noised by the forward process (t = 0 -> T) and how the
    trained MLP denoises exactly these noised points again with the reverse process (t = T -> 0).
    Saves one gif per process to cfg.animate.forward_gif_path and cfg.animate.reverse_gif_path.
    :param data: data points of shape (N, 2) that are animated
    """
    torch.manual_seed(cfg.seed)
    sampler = load_sampler(cfg)
    beta_schedule = utils.beta_schedule(cfg)
    T = len(beta_schedule)
    n = data.size(0)

    # forward process: x_0 -> x_T, one forward step per frame
    forward_frames = [(f"t = 0 / {T}", data)]
    x = data
    for t in range(T):
        x, _ = forward_step(x, torch.full((n,), t), beta_schedule)
        forward_frames.append((f"t = {t + 1} / {T}", x))

    # reverse process starting from exactly the noised points: x_T -> x_0
    reverse_frames = [(f"t = {T} / {T}", x)]
    y = torch.zeros(n, dtype=torch.long)
    for t, x in sampler.trajectory(y, x_T=x):
        reverse_frames.append((f"t = {t} / {T}", x.cpu()))

    _save_gif(cfg, data, forward_frames, "data to noise", cfg.animate.forward_gif_path)
    _save_gif(cfg, data, reverse_frames, "noise to data", cfg.animate.reverse_gif_path)


def _save_gif(cfg: DictConfig, data: torch.Tensor, frames: list, title: str, path: str):
    """
    Saves the frames as gif with cfg.animate.fps frames per second. The first and last frame are held
    for a second. Every frame is reduced to cfg.animate.colors colors, which keeps the gif small,
    the plots only use a few colors anyway.
    :param data: real data points, drawn in light gray as reference
    :param frames: list of (label, points of shape (N, 2))
    """
    fig, ax = plt.subplots(figsize=(5, 5), dpi=cfg.animate.dpi)
    limit = 1.5 * data.abs().max().item()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.scatter(data[:, 0], data[:, 1], s=2, color="lightgray", alpha=0.3, zorder=0)
    scatter = ax.scatter(data[:, 0], data[:, 1], s=4, color="tab:blue", alpha=0.5, zorder=1)
    text = ax.text(0.02, 0.96, "", transform=ax.transAxes, family="monospace")
    fig.tight_layout()

    images = []
    for label, x in frames:
        scatter.set_offsets(x.numpy())
        text.set_text(label)
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        images.append(image.quantize(colors=cfg.animate.colors))
    plt.close(fig)

    frame_ms = round(1000 / cfg.animate.fps)
    durations = [frame_ms] * len(images)
    durations[0] += 1000
    durations[-1] += 1000

    gif_path = utils.path(path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"animation saved to {gif_path}")
