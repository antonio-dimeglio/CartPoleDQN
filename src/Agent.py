from DQNModel import DQNModel as Model
import gymnasium as gym
from collections import deque
from time import time
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Agent:
    def __init__(
            self,
            lr: float = 0.001,
            gamma: float = 0.99,
            epsilon: float = 0.1,
            temperature: float = 1.0,
            batch_size: int = 64,
            memory_size: int = 10000,
            update_frequency: int = 100,
            policy:str = "e_greedy",
            requires_memory: bool = False,
            requires_target: bool = False,
            hidden_size: int = 128

    ) -> None:
        """
        Args:
            lr: learning rate
            gamma: discount factor
            epsilon: exploration rate
            temperature: temperature for softmax policy
            batch_size: batch size for training
            memory_size: size of memory buffer
            update_frequency: frequency of updating target network
            policy: policy for selecting action
            requires_memory: whether the agent requires memory buffer
            requires_target: whether the agent requires target network
        """
        self.env = gym.make('CartPole-v1')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.temperature = temperature
        self.batch_size = batch_size
        self.memory_size = memory_size if requires_memory else 1
        self.update_frequency = update_frequency
        self.policy = policy
        self.memory = deque(maxlen=self.memory_size)
        self.requires_target = requires_target
        self.policy_model = Model(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size).to(self.device)
        
        if self.requires_target:
            self.target_model = Model(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size).to(self.device)
            self.target_model.load_state_dict(self.policy_model.state_dict())
        else:
            self.target_model = self.policy_model

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

    def __backpropagate(self) -> float:
        """
            Args:
                state: current state
                action: action taken
                reward: reward received
                next_state: next state
                done: whether the episode is done

            Returns:
                loss: loss of the model
        """
        
        sample_size = min(len(self.memory), self.batch_size)
        samples = random.sample(self.memory, sample_size)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        # Compute the Q values for the current state-action pairs in the sample
        # and the Q values for the next state-action pairs
        # then compute the target Q values
        q_values = self.policy_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1).values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



    def __select_action(self, state) -> int:
        action = None
        match self.policy:
            case "e_greedy":
                x = random.random()

                if x < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    q_values = self.policy_model(state)
                    action = torch.argmax(q_values).item() 
        
            case "boltzmann" | "softmax":
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                q_values = self.policy_model(state)

                probabilities = F.softmax(q_values / self.temperature, dim=0).cpu().detach().numpy()
                action = np.random.choice(self.env.action_space.n, p=probabilities)

            case _:
                raise NotImplementedError(f"Policy {self.policy} is not implemented.")
            
        return action
    

    def train(self, n_episodes: int = 100) -> np.array:
        """
        Args:
            n_episodes: number of episodes to train
        """
        start_time = time()
        curr_iter = 1
        rewards = np.zeros(n_episodes)
        
        for episode in tqdm(range(n_episodes)):
            
            state, _ = self.env.reset()
            done = False
            curr_reward = 0
            while not done:
                action = self.__select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated 

                self.memory.append((
                        torch.tensor(state, dtype=torch.float32, device=self.device),
                        torch.tensor(action, dtype=torch.int64, device=self.device),
                        torch.tensor(reward, dtype=torch.float32, device=self.device),
                        torch.tensor(next_state, dtype=torch.float32, device=self.device),
                        torch.tensor(done, dtype=torch.float32, device=self.device)))
                

                self.__backpropagate()

                curr_reward += reward 
                state = next_state

                if self.requires_target and curr_iter % self.update_frequency == 0:
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                
                curr_iter += 1
            
            rewards[episode] = curr_reward

        print(f"Training took {round(time() - start_time)} seconds.")
        return rewards
    
    def test(self, n_episodes: int = 100) -> np.array:
        """
        Args:
            n_episodes: number of episodes to test

        Returns:
            rewards: rewards received for each episode
        """
        
        rewards = np.zeros(n_episodes)

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            curr_reward = 0

            while not done:
                action = self.__select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated 

                curr_reward += reward 
                state = next_state

            rewards[episode] = curr_reward

        return rewards