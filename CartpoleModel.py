# Assignment 2, Reinforcement Learning
#
# Authors:      ...
# University:   University of Leiden
# Semester:     Spring 2024 
#
# Description:  The CartpoleModel used to train and test the DQN agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CartpoleModel(nn.Module):
    """
        Cartpole Model implementation
        The architecture consists of:
            - Input     (4 neurons)
            - H1        (32 neurons)
            - H2        (16 neurons)
            - Output    (2 neurons)
    """
    def __init__(self):
        super(CartpoleModel, self).__init__()
        self.layers = [
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 

    