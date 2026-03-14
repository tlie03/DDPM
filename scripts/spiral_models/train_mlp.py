import torch
from torch.utils.data import Dataset, DataLoader

from src.ddpm import forward, MLP
from src.util import create_spiral_dataset_torch

from scripts.spiral_models.params_data import (
    DATA_SEED,
    NUM_SAMPLES,
    START_ANGLE,
    N_TURNS,
    NOISE_LEVEL,
    MAXIMUM_RADIUS,
)

from scripts.spiral_models.params_mlp import (
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


class MyCustomDataset(Dataset):
    def __init__(self, x, y):
        """
        x: Tensor of shape [num_samples, num_features]
        y: Tensor of shape [num_samples, num_classes] (one-hot)
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    x, _ = create_spiral_dataset_torch(
        n_samples=NUM_SAMPLES,
        start_angle=START_ANGLE,
        n_turns=N_TURNS,
        noise_level=NOISE_LEVEL,
        random_seed=DATA_SEED,
        max_radius=MAXIMUM_RADIUS
    )

    dataset = MyCustomDataset(x, torch.zeros(size=(x.size(0),)).int())
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE
    )

    model = MLP(
        x_shape=X_SHAPE,
        y_shape=Y_SHAPE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        time_encoding_channels=TIME_ENCODING_CHANNELS,
        num_classes=NUM_CLASSES
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

        print(f"loss for epoch {epoch}: {epoch_loss / len(data_loader)}")

    torch.save(model.state_dict(), MODEL_PATH)
