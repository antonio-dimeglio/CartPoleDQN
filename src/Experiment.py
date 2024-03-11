from Actor import *
from Model import * 
from ReplayMemory import * 

import numpy as np 
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from time import time 


def experiment_different_exploration_strategies():
    """
        This function performs the experiment for different exploration strategies
    """
    # Create the environment
    env_epsilon = gym.make("CartPole-v1")
    env_boltzmann = gym.make("CartPole-v1")

    # Create the policy model
    policy_model_epsilon = Model(env_epsilon.observation_space.shape[0], env_epsilon.action_space.n)
    policy_model_boltzmann = Model(env_boltzmann.observation_space.shape[0], env_boltzmann.action_space.n)

    # Create the target model
    target_model_epsilon = Model(env_epsilon.observation_space.shape[0], env_epsilon.action_space.n)
    target_model_boltzmann = Model(env_boltzmann.observation_space.shape[0], env_boltzmann.action_space.n)

    # Create different start and end values for epsilon and tau
    epsion_starts = [0.9, 0.5, 0.1]
    epsilon_ends = [0.05, 0.05, 0.05]

    tau_starts = [0.9, 0.5, 0.1]
    tau_ends = [0.05, 0.05, 0.05]

    # Create actors
    policies = ["eps_greedy", "boltzmann"] 

    results = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Train the actors
    for policy in policies:
        for i in range(3):
            start_time = time()
            actor = Actor(env=env_epsilon,
                          policy_model=policy_model_epsilon if policy == "eps_greedy" else policy_model_boltzmann,
                          target_model=target_model_epsilon if policy == "eps_greedy" else target_model_boltzmann,
                          device=device,
                          batch_size=128,
                          gamma=0.999,
                          tau=0.005,
                          learning_rate=1e-3,
                          memory_size=10000,
                          policy=policy,
                          epsilon_start=epsion_starts[i],
                          epsilon_end=epsilon_ends[i],
                          epsilon_decay=1000,
                          temp_start=tau_starts[i],
                          temp_end=tau_ends[i],
                          temp_decay=1000)

            results.append(actor.train())
            print(f"Training policy: {policy}, time elapsed: {round(time() - start_time)} seconds.")


    # Plot the results

    plt.figure(1)
    plt.clf()

    for i in range(3):
        plt.plot(results[i]["episode_durations"], label=f"epsilon_start={epsion_starts[i]}")
        plt.plot(results[i+3]["episode_durations"], label=f"tau_start={tau_starts[i]}")

    plt.title("Episode durations for different exploration strategies")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.legend()
    plt.show()

    # Save the plots 

    plt.savefig("exploration_strategies.png")


def main():
    print("Running the experiments...")
    print("Experiment 1: Different exploration strategies.")
    experiment_different_exploration_strategies()
    print("Experiment 1 completed.")
    print("Experiments completed.")


if __name__ == "__main__":
    main()