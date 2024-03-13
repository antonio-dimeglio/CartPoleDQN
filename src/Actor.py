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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.policy_model = policy_model if policy_model is not None else Model(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = target_model if target_model is not None else Model(env.observation_space.shape[0], env.action_space.n).to(self.device) 
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
                num_episodes:int=500,
                training_type:str="DQN") -> list[float]:

        """
            Trains the model for a given number of episodes
            Args:
                - num_episodes: The number of episodes to train the model
                - training_type: The type of training to be used (DQN, DQN_ER, DQN_TN, DQN_TNER)
        """

        match training_type:
            case "DQN":
                return self.__train_no_memory_no_target(num_episodes)
            case "DQN_ER":
                return self.__train_memory_no_target(num_episodes)
            case "DQN_TN":
                return self.__train_no_memory_target(num_episodes)
            case "DQN_TNER":
                return self.__train_memory_target(num_episodes)
            case _:
                raise ValueError(f"Training type {training_type} not recognized")
            

    def __train_memory_target(self,
              num_episodes:int=500) -> list[float]:
        """
            Trains the model for a given number of episodes using both memory and target models
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

                self.__backprop_memory_target()

                target_model_state_dict = self.target_model.state_dict()
                policy_model_state_dict = self.policy_model.state_dict()

                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key] * self.tau + target_model_state_dict[key]*(1-self.tau)
                
                self.target_model.load_state_dict(target_model_state_dict)

                if done:
                    episode_durations.append(t+1)
                    break

        return episode_durations
    
    def __train_no_memory_no_target(self,
                num_episodes:int=500) -> list[float]:
        """

            Trains the model for a given number of episodes, without using memory and target models
            Args:
            - num_episodes: The number of episodes to train the model

            Returns:
            - episode_durations: The duration of each episode
        """

        self.steps_done = 0
        episode_durations = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0) # Convert to tensor of shape (1, 4)

            for t in count(): # Infinite loop that breaks when the episode ends
                action = self.__select_action(state) # Select an action based on the policy

                # Take the action
                # Item gets the value of the tensor as a python number
                obs, reward, terminated, truncated, _ = self.env.step(action.item()) 

                
                reward = torch.tensor([reward], device=self.device) # Convert to tensor of shape (1, 1)
                

                # Convert the observation to a tensor of shape (1, 4)
                next_state = None if terminated else torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.__backprop(state, action, reward, next_state)

                if terminated or truncated:
                    episode_durations.append(t+1)
                    break

                state = next_state

        return episode_durations
                
    def __train_memory_no_target(self,
                               num_episodes:int=500) -> list[float]:
        """
            Trains the model for a given number of episodes using memory but not target models
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

                self.__backprop_memory_no_target()

                if done:
                    episode_durations.append(t+1)
                    break

        return episode_durations
    
    def __train_no_memory_target(self,
                num_episodes:int=500) -> list[float]:
        """
            Trains the model for a given number of episodes using target but not memory models
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

                self.__backprop_no_memory_target(state, action, reward, next_state)

                target_model_state_dict = self.target_model.state_dict()
                policy_model_state_dict = self.policy_model.state_dict()

                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key] * self.tau + target_model_state_dict[key]*(1-self.tau)
                
                self.target_model.load_state_dict(target_model_state_dict)

                if done:
                    episode_durations.append(t+1)
                    break

                state = next_state

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
            
    def __backprop_memory_target(self):
        """
            Backpropagates the loss to the policy model when using both memory and target models
            Args: None
            Returns: None
        """

        # If the memory is not full, return
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)
        
        # Transpose the batch 
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the non-final states
        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, 
                batch.next_state)),
            device=self.device,
            dtype=torch.bool 
            )
        
        # Concatenate the non-final states
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # Concatenate the states, actions, and rewards
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute the state-action values for the current state and the next state
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Compute the next state values for the non-final states
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected state-action values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        
        # Backpropagate the loss
        loss.backward()

        # Clip the gradients
        nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)

        # Update the model
        self.optimizer.step()

    def __backprop_no_memory_target(self,
                state:torch.Tensor,
                action:torch.Tensor,
                reward:torch.Tensor,
                next_state:torch.Tensor) -> None:
        """
            Backpropagates the loss to the policy model when memory is not used but target is used
            Args:
                state: The current state
                action: The action taken
                reward: The reward received
                next_state: The state transitioned to

            Returns: None
        """

        # To update the model in the case of no memory but target
        # we need to compute the state-action values for the current state
        # and the next state, and then compute the expected state-action values
        # and the loss

        state_action_values = self.policy_model(state).gather(1, action)
        next_state_values = torch.zeros(1, device=self.device)

        if next_state is not None:
            with torch.no_grad():
                next_state_values = self.target_model(next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def __backprop_memory_no_target(self):
        """
            Backpropagates the loss to the policy model when using memory but not target models
            Args: None
            Returns: None
        """

        # If the memory is not full, return
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)
        
        # Transpose the batch 
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the non-final states
        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, 
                batch.next_state)),
            device=self.device,
            dtype=torch.bool 
            )
        
        # Concatenate the non-final states
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # Concatenate the states, actions, and rewards
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute the state-action values for the current state and the next state
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Compute the next state values for the non-final states
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected state-action values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        
        # Backpropagate the loss
        loss.backward()

        # Clip the gradients
        nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)

        # Update the model
        self.optimizer.step()

    def __backprop(self,
                state:torch.Tensor, 
                action:torch.Tensor, 
                reward:torch.Tensor, 
                next_state:torch.Tensor) -> None:
        """
            Backpropagates the loss to the policy model when memory and target are not used
            Args:
                state: The current state
                action: The action taken
                reward: The reward received
                next_state: The state transitioned to
        """
        # To update the model in the case of no memory and no target 
        # we need to compute the state-action values for the current state
        # and the next state, and then compute the expected state-action values
        # and the loss

        state_action_values = self.policy_model(state).gather(1, action)
        next_state_values = torch.zeros(1, device=self.device)

        if next_state is not None:
            with torch.no_grad():
                next_state_values = self.policy_model(next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

