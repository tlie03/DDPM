import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib

matplotlib.use("TkAgg")

from ddpm import MLPDenoise, BetaSchedule, reverse_step

from visualization.spiral_params import (
    DATA_SEED,
    NUM_SAMPLES,
    START_ANGLE,
    N_TURNS,
    NOISE_LEVEL,
    MAXIMUM_RADIUS,
    X_SHAPE,
    Y_SHAPE,
    T,
    BETA_SCHEDULE,
    NUM_LAYERS,
    HIDDEN_DIM,
    TIME_ENCODING_CHANNELS,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
)


def init_reverse_plot(num_samples):
    x_T = torch.randn(num_samples, 2)
    y = torch.zeros(num_samples).int()

    fig, ax = plt.subplots(figsize=(6, 6))

    limit = 3.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("DDPM Reverse Process: Noise to Data")

    scatter = ax.scatter(x_T[:, 0], x_T[:, 1], s=5, alpha=0.5, color='blue')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')

    return x_T, y, fig, ax, scatter, time_text


def update_reverse_frame(t_index, x_t_tensor, y_tensor, scatter, time_text, beta_schedule, model):
    t_index = torch.full((x_t_tensor.shape[0],), t_index).int()
    x_t_minus_1 = reverse_step(x_t_tensor, y_tensor, t_index, model, beta_schedule)

    x_t_tensor.copy_(x_t_minus_1)

    scatter.set_offsets(x_t_tensor.detach().numpy())
    time_text.set_text(f'Time Step t={t_index[0]} / {T}')

    return scatter, time_text


if __name__ == "__main__":
    x_T_state, y, fig, ax, scatter_artist, time_text_artist = init_reverse_plot(NUM_SAMPLES)

    print(f"Creating reverse animation for T={T} steps...")

    t_indices_to_denoise = range(T - 1, -1, -1)

    model = MLPDenoise(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        time_encoding_channels=TIME_ENCODING_CHANNELS,
        num_classes=NUM_CLASSES
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
