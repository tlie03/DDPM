import torch

from src.ddpm import BetaSchedule


X_SHAPE = torch.Size((1, 28, 28))
Y_SHAPE = torch.Size((10,))

T = 100
BETA_SCHEDULE = BetaSchedule.cosine(T)

BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 0.002
MODEL_PATH = "mnist_unet"

TIME_ENCODING_CHANNELS = 16
NUM_CHANNELS_INIT = 8
NUM_BLOCKS = 3
ACTIVATION_FUNCTION = "SiLU"
