import torch.nn as nn 
import torch.nn.functional as F

class DQNModel(nn.Module):
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