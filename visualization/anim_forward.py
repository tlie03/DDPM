import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib

from ddpm import BetaSchedule, forward_step
from data.spiral_data import create_spiral_dataset_torch


matplotlib.use("TkAgg")


DATA_SEED = 42
NUM_SAMPLES = 2000
START_ANGLE = 0
N_TURNS = 1.5
NOISE_LEVEL = 0.00
MAXIMUM_RADIUS = 1.5

T = 1000
BETA_SCHEDULE = BetaSchedule.cosine(T)


current_x = None


def init_plot(x_data):
    """Initializes the plot artists and sets up the axes."""
    global current_x

    # 1. Initialize current_x to the starting data (x_0)
    # Detach and move to CPU for plotting/NumPy conversion if necessary
    current_x = x_data.clone().cpu()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Set axis limits based on the maximum expected displacement (e.g., +/- 2*max_radius)
    limit = MAXIMUM_RADIUS * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("DDPM Forward Process: Data to Noise")

    # Create the mutable scatter plot artist
    # x_0 is an (N, 2) tensor/array.
    scatter = ax.scatter(current_x[:, 0], current_x[:, 1], s=5, alpha=0.5, color='blue')

    # Create the mutable text artist for the timestep display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')

    return fig, ax, scatter, time_text


def update_frame(t_index, scatter, time_text, beta_schedule):
    """
    Update function called by FuncAnimation.
    Performs one forward step (x_t -> x_t+1) and updates the plot.

    t_index is the target time step index for the current frame.
    """
    global current_x

    t_batch = torch.full((current_x.shape[0],), t_index, dtype=torch.long)
    x_next, _ = forward_step(current_x, t_batch, beta_schedule)

    current_x = x_next

    # 4. Update the Matplotlib artists
    # Convert the torch tensor back to a numpy array for plotting
    scatter.set_offsets(current_x.numpy())
    time_text.set_text(f'Time Step t={t_index} / {T}')

    return scatter, time_text


if __name__ == "__main__":
    x, _ = create_spiral_dataset_torch(
        n_samples=NUM_SAMPLES,
        start_angle=START_ANGLE,
        n_turns=N_TURNS,
        noise_level=NOISE_LEVEL,
        random_seed=DATA_SEED,
        max_radius=MAXIMUM_RADIUS,
    )

    # 1. Initialize the plot and get the artists
    fig, ax, scatter_artist, time_text_artist = init_plot(x)

    # 2. Create the FuncAnimation object
    print(f"Creating animation for T={T} steps...")

    ani = anim.FuncAnimation(
        fig,
        update_frame,
        frames=T,
        fargs=(scatter_artist, time_text_artist, BETA_SCHEDULE),
        interval=10,
        blit=True,
        repeat=True,
    )
    plt.show()


