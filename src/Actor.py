import gymnasium as gym 
from Model import Model 
from ReplayMemory import *
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import random
from math import exp 
from itertools import count 


class Actor:
    def __init__(self,
                 env:gym.Env,
                 policy_model:Model = None,
                 target_model:Model = None,
                 device:str="cpu",
                 batch_size:int=128,
                 gamma:float=0.999,
                 tau:float=0.005,
                 learning_rate:float=1e-3,
                 memory_size:int=10000,
                 policy:str="eps_greedy",

                 epsilon_start:float=0.9,
                 epsilon_end:float=0.05,
                 epsilon_decay:int=1000,

                 temp_start:float=0.9,
                 temp_end:float=0.05,
                 temp_decay:int=1000):
        """
            Initializes the Actor class with the following parameters:
            - env: The environment to be used for training (cartpole)
            - policy_model: The model to be used for training (if not loaded, a new model will be created)
            - target_model: The model to be used for target (if not loaded, a new model will be created)
            - device: The device to be used for training (cpu or cuda)
            - batch_size: The batch size to be used for training
            - gamma: The discount factor to be used for training
            - learning_rate: The learning rate to be used for training
            - memory_size: The memory size to be used for training
            - policy: The policy to be used for training (eps_greedy or boltzmann)
            
            - epsilon_start: The starting value for epsilon
            - epsilon_end: The ending value for epsilon
            - epsilon_decay: The decay value for epsilon
            
            - tau_start: The starting value for tau
            - tau_end: The ending value for tau
            - tau_decay: The decay value for tau
        """
        
        self.env = env 
        self.device = device 
        self.policy_model = policy_model if policy_model is not None else Model(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_model = target_model if target_model is not None else Model(env.observation_space.shape[0], env.action_space.n).to(device) 
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy = policy


        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        self.loss = nn.SmoothL1Loss()

        # Epsilon-greedy policy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Boltzmann policy parameters
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay = temp_decay

    def train(self,
              num_episodes:int=500) -> list[float]:
        """
            Trains the model for a given number of episodes
            Args:
            - num_episodes: The number of episodes to train the model

            Returns:
            - episode_durations: The duration of each episode
        """

        self.steps_done = 0 
        episode_durations = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

            for t in count():
                action = self.__select_action(state)

                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.__backprop()

                target_model_state_dict = self.target_model.state_dict()
                policy_model_state_dict = self.policy_model.state_dict()

                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key] * self.tau + target_model_state_dict[key]*(1-self.tau)
                
                self.target_model.load_state_dict(target_model_state_dict)

                if done:
                    episode_durations.append(t+1)
                    break

        return episode_durations


    def __select_action(self, state:torch.Tensor) -> torch.Tensor:
        """
            Selects an action based on the policy and the current state

            Args:
            - state: The current state of the environment

            Returns:
            - action: The action to be taken (as a tensor)
        """
        action = None
        match self.policy:
            case "eps_greedy":
                # Epsilon-greedy policy with exponential decay
                x = random.random()
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * exp(-1. * self.steps_done / self.epsilon_decay)

                if x < epsilon:
                    action = torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_model(state).max(1)[1].view(1, 1)
            case "boltzmann":
                # Boltzmann policy with exponential decay
                x = random.random()
                tau = self.temp_end + (self.temp_start - self.temp_end) * exp(-1. * self.steps_done / self.temp_decay)

                with torch.no_grad():
                    action_values = self.policy_model(state)
                    action_probs = F.softmax(action_values / tau, dim=1)
                    action = action_probs.multinomial(1)
            case _:
                raise ValueError(f"Policy {self.policy} not recognized")
            
        self.steps_done += 1
        return action
            
    def __backprop(self):
        """
            Backpropagates the loss to the policy model
            Args: None
            Returns: None
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, 
                batch.next_state)),
            device=self.device,
            dtype=torch.bool 
            )
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()