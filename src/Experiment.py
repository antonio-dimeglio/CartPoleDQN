from Actor import *
from Model import * 
from ReplayMemory import * 

import numpy as np 
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from time import time 

def moving_average(data, window_size):
    """
        This function calculates the moving average of the given data
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def experiment_different_exploration_strategies():
    """
        This function performs the experiment for different exploration strategies
    """
    # Create the environment
    env_epsilon = gym.make("CartPole-v1")
    env_boltzmann = gym.make("CartPole-v1")

    # Create the policy model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy_model_epsilon = Model(env_epsilon.observation_space.shape[0], env_epsilon.action_space.n).to(device)
    policy_model_boltzmann = Model(env_boltzmann.observation_space.shape[0], env_boltzmann.action_space.n).to(device)

    # Create the target model
    target_model_epsilon = Model(env_epsilon.observation_space.shape[0], env_epsilon.action_space.n).to(device)
    target_model_boltzmann = Model(env_boltzmann.observation_space.shape[0], env_boltzmann.action_space.n).to(device)

    # Create different start and end values for epsilon and tau
    epsilon_starts = [0.9, 0.5, 0.1]
    epsilon_ends = [0.05, 0.05, 0.05]

    tau_starts = [0.9, 0.5, 0.1]
    tau_ends = [0.05, 0.05, 0.05]

    # Create actors
    policies = ["eps_greedy", "boltzmann"] 

    results = []

    # Train the actors
    for policy in policies:
        for i in range(3):
            start_time = time()
            actor = Actor(env=env_epsilon if policy == "eps_greedy" else env_boltzmann,
                          policy_model=policy_model_epsilon if policy == "eps_greedy" else policy_model_boltzmann,
                          target_model=target_model_epsilon if policy == "eps_greedy" else target_model_boltzmann,
                          batch_size=128,
                          gamma=0.999,
                          tau=0.005,
                          learning_rate=1e-3,
                          memory_size=10000,
                          policy=policy,
                          epsilon_start=epsilon_starts[i],
                          epsilon_end=epsilon_ends[i],
                          epsilon_decay=1000,
                          temp_start=tau_starts[i],
                          temp_end=tau_ends[i],
                          temp_decay=1000)

            results.append(actor.train_memory_target())
            print(f"Training policy: {policy}, time elapsed: {round(time() - start_time)} seconds.")


    # Plot the results

    plt.figure(1)
    plt.clf()

    for i in range(3):
        plt.plot(moving_average(results[i], 100), label=fr"eps_greedy, $\Epsilon$-start ={epsilon_starts[i]}, $\Epsilon$-end={epsilon_ends[i]}")
        plt.plot(moving_average(results[i + 3], 100), label=f"boltzmann, $\Tau$-start={tau_starts[i]}, $\Tau$-end={tau_ends[i]}")

    plt.title("Episode durations for different exploration strategies")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.legend()
    plt.show()

    # Save the plots 

    plt.savefig("exploration_strategies.png")

def experiment_dqn_vs_dqner_vs_dqntn_vs_dqntner():
    """
        This function performs the experiment for DQN, DQNER, DQNTN and DQNTNER
    """
    # Create the environment
    env_dqn = gym.make("CartPole-v1")
    env_dqner = gym.make("CartPole-v1")
    env_dqntn = gym.make("CartPole-v1")
    env_dqntner = gym.make("CartPole-v1")

    # Create the policy model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy_model_dqn = Model(env_dqn.observation_space.shape[0], env_dqn.action_space.n).to(device)
    policy_model_dqner = Model(env_dqner.observation_space.shape[0], env_dqner.action_space.n).to(device)
    policy_model_dqntn = Model(env_dqntn.observation_space.shape[0], env_dqntn.action_space.n).to(device)
    policy_model_dqntner = Model(env_dqntner.observation_space.shape[0], env_dqntner.action_space.n).to(device)

    # Create the target model
    target_model_dqn = Model(env_dqn.observation_space.shape[0], env_dqn.action_space.n).to(device)
    target_model_dqner = Model(env_dqner.observation_space.shape[0], env_dqner.action_space.n).to(device)
    target_model_dqntn = Model(env_dqntn.observation_space.shape[0], env_dqntn.action_space.n).to(device)
    target_model_dqntner = Model(env_dqntner.observation_space.shape[0], env_dqntner.action_space.n).to(device)

    # Create the actors
    actors = [Actor(env=env_dqn,
                    policy_model=policy_model_dqn,
                    target_model=target_model_dqn,
                    batch_size=128,
                    gamma=0.999,
                    tau=0.005,
                    learning_rate=1e-3,
                    memory_size=10000,
                    policy="eps_greedy",
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay=1000,
                    temp_start=0.9,
                    temp_end=0.05,
                    temp_decay=1000),

                Actor(env=env_dqner,
                    policy_model=policy_model_dqner,
                    target_model=target_model_dqner,
                    batch_size=128,
                    gamma=0.999,
                    tau=0.005,
                    learning_rate=1e-3,
                    memory_size=10000,
                    policy="eps_greedy",
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay=1000,
                    temp_start=0.9,
                    temp_end=0.05,
                    temp_decay=1000),

                Actor(env=env_dqntn,
                    policy_model=policy_model_dqntn,
                    target_model=target_model_dqntn,
                    batch_size=128,
                    gamma=0.999,
                    tau=0.005,
                    learning_rate=1e-3,
                    memory_size=10000,
                    policy="eps_greedy",
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay=1000,
                    temp_start=0.9,
                    temp_end=0.05,
                    temp_decay=1000),

                Actor(env=env_dqntner,
                    policy_model=policy_model_dqntner,
                    target_model=target_model_dqntner,
                    batch_size=128,
                    gamma=0.999,
                    tau=0.005,
                    learning_rate=1e-3,
                    memory_size=10000,
                    policy="eps_greedy",
                    epsilon_start=0.9,
                    epsilon_end=0.05,
                    epsilon_decay=1000,
                    temp_start=0.9,
                    temp_end=0.05,
                    temp_decay=1000)]

    results = []
    training_type = ["DQN", "DQN_ER", "DQN_TN", "DQN_TNER"]
    # Train the actors
    for i in range(4):
        start_time = time()
        results.append(actors[i].train(training_type=training_type[i]))
        print(f"Training {training_type[i]}, time elapsed: {round(time() - start_time)} seconds.")


    # Plot the results
    for i in range(4):
        plt.plot(moving_average(results[i], 100), label=f"Actor {i + 1}")

    plt.title("Episode durations for DQN, DQNER, DQNTN and DQNTNER")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.legend()
    
    # Save the plot
    plt.savefig("dqn_dqner_dqntn_dqntner.png")





def main():
    print("Running the experiments...")
    print("Running the experiment for different DQN variants...")
    experiment_dqn_vs_dqner_vs_dqntn_vs_dqntner()

if __name__ == "__main__":
    main()