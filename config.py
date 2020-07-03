import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 100
LEARNING_RATE = 1e-3
TOTAL_EPOCH = 30