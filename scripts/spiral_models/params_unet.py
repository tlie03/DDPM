import torch

from src.ddpm import BetaSchedule


T = 100
BETA_SCHEDULE = BetaSchedule.cosine(T)
print(BETA_SCHEDULE.expected_alpha_bar())

BATCH_SIZE = 128
EPOCHS = 4000
LEARNING_RATE = 0.001
MODEL_PATH = "trained_models/unet.pth"
X_SHAPE = torch.Size((1, 1, 2))
Y_SHAPE = torch.Size((1,))


# unet specific configurations
NUM_BLOCKS = 2
NUM_CHANNELS_INIT = 10
ACTIVATION_FUNCTION = "SiLU"
TIME_ENCODING_CHANNELS = 16
