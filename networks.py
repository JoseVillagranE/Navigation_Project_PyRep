import torch
import torch.nn as nn
import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, state_dim):

        self.model = nn.Sequential(nn.Linear(state_dim, 34),
                                   nn.ReLU(),
                                   nn.Linear(34, 55),
                                   nn.ReLU(),
                                   nn.Linear(55, 2))

    def forward(self, x):
        return self.model(x)

class CriticNetwork(nn.Module):

    def __init__(self, state_dim):

        self.model = nn.Sequential(nn.Linear(state_dim+2, 261), nn.ReLU(),
                                   nn.Linear(261, 163), nn.ReLU(),
                                   nn.Linear(163, 33), nn.ReLU(),
                                   nn.Linear(33, 1))

    def forward(self, state, action):
        return self.model(torch.cat((state, action), dim=1)
