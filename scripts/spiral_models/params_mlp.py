import torch

from src.ddpm import BetaSchedule


T = 100
BETA_SCHEDULE = BetaSchedule.cosine(T)

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.001
MODEL_PATH = "trained_models/mlp.pth"
X_SHAPE = torch.Size((2,))
Y_SHAPE = torch.Size()

# mlp specific configurations
NUM_LAYERS = 4
HIDDEN_DIM = 16
TIME_ENCODING_CHANNELS = 8
NUM_CLASSES = 1
