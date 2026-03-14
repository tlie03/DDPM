import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from src.ddpm import forward, Unet


from scripts.mnist.params import (
    X_SHAPE,
    Y_SHAPE,
    T,
    BETA_SCHEDULE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    TIME_ENCODING_CHANNELS,
    NUM_CHANNELS_INIT,
    NUM_BLOCKS,
    ACTIVATION_FUNCTION
)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x * 2) - 1)])
target_transform = transforms.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)

# 2. Load the dataset
mnist_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
    target_transform=target_transform
)


if __name__ == "__main__":
    data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

    model = Unet(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        num_blocks=NUM_BLOCKS,
        num_channels_init=NUM_CHANNELS_INIT,
        activation_function=ACTIVATION_FUNCTION,
        time_encoding_channels=TIME_ENCODING_CHANNELS
    )

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()

            t = torch.randint(low=0, high=T, size=(x.size(0),))
            x_t, noise = forward(x, t, BETA_SCHEDULE)

            noise_pred = model(x=x_t, y=y, t=t)
            loss = loss_function(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"loss for epoch {epoch}: {epoch_loss / len(data_loader)}")

    torch.save(model.state_dict(), MODEL_PATH)
