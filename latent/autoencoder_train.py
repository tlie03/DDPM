import torch.nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from autoencoder import Autoencoder
from autoencoder import NUM_DIGITS

ENCODER_HIDDEN = [512, 256, 128]
DECODER_HIDDEN = [128, 256, 512]
LATENT_DIM = 8

EPOCHS = 80
LEARNING_RATE = 0.0001


def one_hot_target(label):
    return F.one_hot(torch.tensor(label), num_classes=NUM_DIGITS).float()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=one_hot_target
    )
    print("Loaded training dataset")

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        target_transform=one_hot_target
    )
    print("Loaded test dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    autoencoder = Autoencoder(
        enc_hidden=ENCODER_HIDDEN,
        dec_hidden=DECODER_HIDDEN,
        latent_dim=LATENT_DIM
    )

    params = autoencoder.parameters()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_loader):
            x = batch[0]
            y = batch[1]

            reconstruction = autoencoder.forward(x, y)
            loss = loss_function(x, reconstruction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = 0
        for i, batch in enumerate(test_loader):
            x = batch[0]
            y = batch[1]

            reconstruction = autoencoder.forward(x, y)
            loss = loss_function(x, reconstruction)
            final_loss += loss
        print(f"test loss of epoch {epoch+1} is {final_loss}")

    torch.save(autoencoder.state_dict(), "autoencoder.pth")
