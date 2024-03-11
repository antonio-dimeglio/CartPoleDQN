# Assignment 2, Reinforcement Learning
# Authors: ...
# University of Leiden
# Semester: Spring 2024 
#
# Description: The entry point for the DQN agent.

import argparse as ap
import gymnasium as gym
import torch
from Model import Model
from ReplayMemory import *
import random 
import math
import matplotlib.pyplot as plt 
import torch.nn as nn 
from itertools import count 

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000

def select_action(state, 
                  steps_done,
                  policy_model,
                  env,
                  device):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_duration(
        episode_durations:list,
        show_results:bool=False):

    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_results:
        plt.title("Results")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())


    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model(
        memory:ReplayMemory,
        device:str,
        policy_model: Model,
        target_model: Model,
        optimizer
):
    if len(memory) < BATCH_SIZE:
        return 
    
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(
        map(lambda s: s is not None, 
            batch.next_state)),
        device=device,
        dtype=torch.bool 
        )

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_model(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_observation = env.observation_space.shape[0]
    
    policy_model = Model(n_observation, n_actions)
    target_model = Model(n_observation, n_actions)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    num_episodes = 500 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps_done = 0 

    episode_durations = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state,
                                   steps_done,
                                   policy_model,
                                   env,
                                   device)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated 

            if terminated:
                next_state = None 
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)

            state = next_state 

            optimize_model(memory, 
                           device, 
                           policy_model, 
                           target_model,
                           optimizer)
            
            target_model_state_dict = target_model.state_dict()
            policy_model_state_dict = target_model.state_dict()

            for key in policy_model_state_dict:
                target_model_state_dict[key] = policy_model_state_dict[key] * TAU + target_model_state_dict[key]*(1-TAU)
            
            target_model.load_state_dict(target_model_state_dict)


            if done:
                episode_durations.append(t+1)
                plot_duration(episode_durations)
                break 
    
    print("Done")
    plot_duration(episode_durations, True)
    plt.ioff()
    plt.show()
        

    


if __name__ == "__main__":
    main()