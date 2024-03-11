import torch 
from torch import nn 
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_observation, n_actions):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

def load_model(path:str, n_observation, n_actions) -> Model:
    """
        Loads the model from the specified path

        Args:
        - path: The path to the model file
        - n_observation: The number of observations
        - n_actions: The number of actions

        Returns:
        - The model
    """
    model = Model(n_observation, n_actions)
    model.load_state_dict(torch.load(path))
    return model

def save_model(path:str, model: Model) -> None:
    """
        Saves the model to the specified path

        Args:
        - path: The path to the model file
        - model: The model to be saved

        Returns: None
    """
    torch.save(model.state_dict(), path)