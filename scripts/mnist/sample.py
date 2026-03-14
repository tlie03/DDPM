import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from src.ddpm import Unet, reverse

from scripts.mnist.params import (
    X_SHAPE,
    Y_SHAPE,
    BETA_SCHEDULE,
    MODEL_PATH,
    TIME_ENCODING_CHANNELS,
    NUM_CHANNELS_INIT,
    NUM_BLOCKS,
    ACTIVATION_FUNCTION
)

digit = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

if __name__ == "__main__":

    model = Unet(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        time_encoding_channels=TIME_ENCODING_CHANNELS,
        num_channels_init=NUM_CHANNELS_INIT,
        num_blocks=NUM_BLOCKS,
        activation_function=ACTIVATION_FUNCTION
    )

    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)

    noise = torch.randn(size=torch.Size((1, )) + X_SHAPE)
    sample = reverse(noise, digit, model, BETA_SCHEDULE)

    plt.imshow(sample.squeeze().detach().numpy(), cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')  # Hide the X and Y axes
    plt.show()
