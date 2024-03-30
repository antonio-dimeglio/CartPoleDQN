import torch.nn as nn 
import torch.nn.functional as F

class DQNModel(nn.Module):
    """
        Simple DQN model with 3 fully connected layers and ReLU activation.
        The n_hidden parameter specifies the number of hidden units in the hidden layers.
    """
    def __init__(self, 
                 n_state:int = 4, 
                 n_action:int = 2,
                 n_hidden:int = 128):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DuelingDQNModel(nn.Module):
    """
        Dueling DQN model with 3 fully connected layers and ReLU activation.
        The n_hidden parameter specifies the number of hidden units in the hidden layers.

        The model has two output layers: one for the value function and one for the advantage function.
        The final output is the sum of the value and advantage functions minus the mean of the advantage function.
    """
    def __init__(self, 
                 n_state:int = 4, 
                 n_action:int = 2,
                 n_hidden:int = 128):
        super(DuelingDQNModel, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        
        self.advantage = nn.Linear(n_hidden, n_action)
        self.value = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage - advantage.mean()