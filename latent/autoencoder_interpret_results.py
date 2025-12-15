import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib

from autoencoder import Autoencoder
from autoencoder_train import (LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN)

matplotlib.use("TkAgg")

DIGIT_VECTOR = torch.Tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
IMAGES_INBETWEEN = 1


if __name__ == "__main__":

    model = Autoencoder(
        latent_dim=LATENT_DIM,
        enc_hidden=ENCODER_HIDDEN,
        dec_hidden=DECODER_HIDDEN,
    )
    state_dict = torch.load("autoencoder.pth")
    model.load_state_dict(state_dict)
    decoder = model.get_decoder()

    # each latent dim has value between -1 and 1
    latent_dim = decoder.get_latent_dim()

    images = []
    for dim in range(latent_dim):
        step_size = 2/(IMAGES_INBETWEEN+1)
        interpolation = -1
        for _ in range(IMAGES_INBETWEEN + 2):
            latent = torch.zeros((1, latent_dim))
            latent[0][dim] = interpolation
            input = torch.concat([DIGIT_VECTOR, latent], dim=1)
            output = decoder(input)

            interpolation += step_size
            images.append(output)

    images = torch.cat(images, dim=0)
    print(images.shape)
    grid = make_grid(images, nrow=IMAGES_INBETWEEN+2)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
