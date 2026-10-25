import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        weights = 1 / 10 ** np.linspace(0, 9, dimension)
        self.w.weight = torch.nn.Parameter(
            torch.from_numpy(weights).float().reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output