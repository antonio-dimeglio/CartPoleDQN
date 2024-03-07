# Assignment 2, Reinforcement Learning
#
# Authors:      ...
# University:   University of Leiden
# Semester:     Spring 2024 
#
# Description:  The Agent module to be used to 
#               train and test the DQN agent
#               on the CartPole environment.

import numpy as np 
import torch
from CartpoleModel import CartpoleModel
import gymnasium as gym 

class Agent:
    """
        TODO: Docstrings and class implementation
    """

    def __init__(self, 
                 policy:str, 
                 model:CartpoleModel = None, 
                 lr:float=0.001,
                 gamma:float=0.99,
                 epsilon:float=1.0):
        
        self.model:CartpoleModel = CartpoleModel() if model == None else model 
        self.env:gym.Env = gym.make("CartPole-v1")
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon


        # Model optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # Loss function
        self.loss = torch.nn.MSELoss()

        # Model device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def _perform_action(self, 
                        state:torch.FloatTensor) -> int:
        """
            TODO: Finish docstring and do implementation 
            Performs an action based on the current state
            and the policy used.

            Args:
                state (FloatTensor): The current state of the agent in the environment

            Returns:
                int: The action to be performed
        """
        # action = None
        # match self.policy:
        #     case "e-greedy":
        #         if np.random.uniform() < self.epsilon:
        #             action = self.env.action_space.sample()
        #         else:
        #             action = torch.argmax(q_values).item()

        #     case "softmax":
        #         probabilities = torch.softmax(q_values, dim=0)
        #         action = np.random.choice(
        #             range(self.env.action_space.n),
        #             p=probabilities.detach().numpy() # The tensor is detached as it is loaded onto the GPU first
        #         )

        #     case "novelty":
        #         # TODO
        #         pass 
        #     case _:
        #         raise ValueError("Invalid policy")
        pass 
    

    def _update_model(self):
        """
            TODO: Docstring and implementation
        """
        pass 

            
    def train(self, epochs:int=100):
        """
            TODO: Docstring and finishing implementation 
        """
        for epoch in range(epochs):
            done = False
            state, _ = self.env.reset()

            while not done:
                action = self._perform_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state 

    def save(self, path:str="cartpole_model.pt"):
        torch.save(
            self.model.state_dict(),
            path
        )

    def load(self, path:str="cartpole_model.pt"):
        self.model.load_state_dict(
            path
        )