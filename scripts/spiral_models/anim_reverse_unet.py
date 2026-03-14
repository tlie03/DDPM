import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib

from src.ddpm.beta_schedule import BetaSchedule

matplotlib.use("TkAgg")

from src.ddpm import Unet, reverse_step

from scripts.spiral_models.params_data import (
    MAXIMUM_RADIUS,
)
from scripts.spiral_models.params_unet import (
    T,
    BETA_SCHEDULE,
    TIME_ENCODING_CHANNELS,
    MODEL_PATH,
    X_SHAPE,
    Y_SHAPE,
    NUM_BLOCKS,
    NUM_CHANNELS_INIT,
    ACTIVATION_FUNCTION,
)

ALPHA = 0.1


def init_reverse_plot(num_samples):
    x_T = torch.randn(num_samples, 2)
    y = torch.zeros(num_samples)

    fig, ax = plt.subplots(figsize=(6, 6))

    limit = MAXIMUM_RADIUS * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("DDPM Reverse Process: Noise to Data")

    scatter = ax.scatter(x_T[:, 0], x_T[:, 1], s=5, alpha=ALPHA, color='blue')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')

    return x_T, y, fig, ax, scatter, time_text


def update_reverse_frame(t_index, x_t_tensor, y_tensor, scatter, time_text, beta_schedule, model):
    t_index = torch.full((x_t_tensor.shape[0],), t_index).int()
    x_t_minus_1 = reverse_step(x_t_tensor[:, None, None, :], y_tensor[:, None], t_index, model, beta_schedule)
    x_t_minus_1 = x_t_minus_1.squeeze()

    x_t_tensor.copy_(x_t_minus_1)

    scatter.set_offsets(x_t_tensor.detach().numpy())
    time_text.set_text(f'Time Step t={t_index[0]} / {T}')

    return scatter, time_text


if __name__ == "__main__":
    x_T_state, y, fig, ax, scatter_artist, time_text_artist = init_reverse_plot(2000)

    print(f"Creating reverse animation for T={T} steps...")

    t_indices_to_denoise = range(T - 1, -1, -1)

    model = Unet(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        time_encoding_channels=TIME_ENCODING_CHANNELS,
        num_blocks=NUM_BLOCKS,
        num_channels_init=NUM_CHANNELS_INIT,
        activation_function=ACTIVATION_FUNCTION
    )

    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)

    ani_reverse = anim.FuncAnimation(
        fig,
        update_reverse_frame,
        frames=t_indices_to_denoise,
        fargs=(x_T_state, y, scatter_artist, time_text_artist, BETA_SCHEDULE, model),
        interval=50,
        blit=True,
        repeat=False,
    )

    plt.show()
