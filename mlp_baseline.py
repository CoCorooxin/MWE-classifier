import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_baseline(nn.Module):
    """
    A vanilla MLP with 2 hidden layers
    """
    def __init__(self, emb_size, hidden_size, drop_out = 0.):

        super(MLP_baseline, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_size * self.input_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Dropout(drop_out),
            nn.ReLU(),

        )
