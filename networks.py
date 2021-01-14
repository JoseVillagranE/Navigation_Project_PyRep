import torch
import torch.nn as nn
import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(state_dim, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 2),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)

class CriticNetwork(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.preprocess = nn.Linear(state_dim, 256)
        self.model = nn.Sequential(nn.Linear(256 + 2, 256), nn.LeakyReLU(),
                                   nn.Linear(256, 256), nn.LeakyReLU(),
                                   nn.Linear(256, 1))

    def forward(self, state, action):
        state = self.preprocess(state)
        return self.model(torch.cat((state, action), dim=1))
