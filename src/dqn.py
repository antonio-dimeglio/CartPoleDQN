from Agent import *
import numpy as np 
import matplotlib.pyplot as plt
import argparse as ap
from tabulate import tabulate
from time import time 

TRAIN_EPISODES = 500
NUM_TRIALS = 5
WINDOW_LENGTH = 5

def moving_average(data, window_length):
    return np.convolve(data, np.ones(window_length), 'valid') / window_length

def measure_runtimes():
    print("Measuring runtimes...")

    agents = ["DQN", "DQN with target network", "DQN with experience replay", "DQN with target network and experience replay"]
    uses_memory = [False, False, True, True]
    uses_target = [False, True, False, True]

    runtimes = np.zeros((4, num_trials))

    for i, curr_agent in enumerate(agents):
        print(f"Measuring runtime for {curr_agent}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            start_time = time()
            agent = Agent(requires_memory=uses_memory[i], requires_target=uses_target[i])
            agent.train(TRAIN_EPISODES)
            runtimes[i, j] = time() - start_time
    
    runtimes_avg = np.mean(runtimes, axis=1)
    runtimes_std = np.std(runtimes, axis=1)

    table = []
    for i in range(4):
        table.append([agents[i], runtimes_avg[i], runtimes_std[i]])
    
    print(tabulate(table, headers=["Agent", "Average Runtime (s)", "Standard Deviation"]))
    print("Runtime measurement complete.")



def ablation_experiment():
    print("Running ablation study...")
    results = np.zeros((4, TRAIN_EPISODES, NUM_TRIALS))
    agents = ["DQN", "DQN with target network", "DQN with experience replay", "DQN with target network and experience replay"]
    uses_memory = [False, False, True, True]
    uses_target = [False, True, False, True]

    for i, curr_agent in enumerate(agents):
        print(f"Traning {curr_agent}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(requires_memory=uses_memory[i], requires_target=uses_target[i])
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(4):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=agents[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Ablation Study")
    plt.savefig("ablation_study.png")
    plt.clf()
    print("Ablation study complete.")

def learning_rate_experiment():
    print("Running learning rate experiment...")
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    results = np.zeros((len(learning_rates), TRAIN_EPISODES, NUM_TRIALS))


    for i, lr in enumerate(learning_rates):
        print(f"Training with learning rate {lr}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(lr=lr, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)
        
    for i in range(len(learning_rates)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"lr = {learning_rates[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Rate Experiment")
    plt.savefig("learning_rate_experiment.png")
    plt.clf()
    print("Learning rate experiment complete.")

def policy_experiment():
    print("Running policy experiment...")
    policies = ["e_greedy", "softmax"]
    results = np.zeros((len(policies), TRAIN_EPISODES, NUM_TRIALS))

    for i, policy in enumerate(policies):
        print(f"Training with policy {policy}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(policy=policy, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(policies)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=policies[i])

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Policy Experiment")
    plt.savefig("policy_experiment.png")
    plt.clf()
    print("Policy experiment complete.")

def batch_size_experiment():
    print("Running batch size experiment...")
    batch_sizes = [32, 64, 128, 256]
    results = np.zeros((len(batch_sizes), TRAIN_EPISODES, NUM_TRIALS))

    for i, batch_size in enumerate(batch_sizes):
        print(f"Training with batch size {batch_size}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(batch_size=batch_size, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(batch_sizes)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"batch size = {batch_sizes[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Batch Size Experiment")
    plt.savefig("batch_size_experiment.png")
    plt.clf()
    print("Batch size experiment complete.")

def memory_size_experiment():
    print("Running memory size experiment...")
    memory_sizes = [1000, 5000, 10000, 50000]
    results = np.zeros((len(memory_sizes), TRAIN_EPISODES, NUM_TRIALS))

    for i, memory_size in enumerate(memory_sizes):
        print(f"Training with memory size {memory_size}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(memory_size=memory_size, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(memory_sizes)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"memory size = {memory_sizes[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Memory Size Experiment")
    plt.savefig("memory_size_experiment.png")
    plt.clf()
    print("Memory size experiment complete.")

def update_frequency_experiment():
    print("Running update frequency experiment...")
    update_frequencies = [10, 50, 100, 200]
    results = np.zeros((len(update_frequencies), TRAIN_EPISODES, NUM_TRIALS))

    for i, update_frequency in enumerate(update_frequencies):
        print(f"Training with update frequency {update_frequency}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(update_frequency=update_frequency, requires_target=True, requires_memory=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(update_frequencies)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"update frequency = {update_frequencies[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Update Frequency Experiment")
    plt.savefig("update_frequency_experiment.png")
    plt.clf()
    print("Update frequency experiment complete.")

def exploration_factor_experiment():
    print("Running exploration factor experiment...")
    epsilons = [0.01, 0.1, 0.2, 0.5]
    results = np.zeros((len(epsilons), TRAIN_EPISODES, NUM_TRIALS))

    for i, epsilon in enumerate(epsilons):
        print(f"Training with epsilon {epsilon}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(epsilon=epsilon, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(epsilons)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"epsilon = {epsilons[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Exploration Factor Experiment")
    plt.savefig("exploration_factor_experiment.png")
    plt.clf()
    print("Exploration factor experiment complete.")

def temperature_experiment():
    print("Running temperature experiment...")
    temperatures = [0.1, 0.5, 1.0, 2.0]
    results = np.zeros((len(temperatures), TRAIN_EPISODES, NUM_TRIALS))

    for i, temperature in enumerate(temperatures):
        print(f"Training with temperature {temperature}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(temperature=temperature, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(temperatures)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"temperature = {temperatures[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Temperature Experiment")
    plt.savefig("temperature_experiment.png")
    plt.clf()
    print("Temperature experiment complete.")

def architecture_experiment():
    print("Running architecture experiment...")
    hidden_sizes = [32, 64, 128, 256]
    results = np.zeros((len(hidden_sizes), TRAIN_EPISODES, NUM_TRIALS))

    for i, hidden_size in enumerate(hidden_sizes):
        print(f"Training with hidden size {hidden_size}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(hidden_size=hidden_size, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(hidden_sizes)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=f"hidden size = {hidden_sizes[i]}")

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Architecture Experiment")
    plt.savefig("architecture_experiment.png")
    plt.clf()
    print("Architecture experiment complete.")


def hyperparameters_experiment():
    learning_rate_experiment()
    policy_experiment()
    batch_size_experiment()
    memory_size_experiment()
    update_frequency_experiment()
    exploration_factor_experiment()
    temperature_experiment()
    architecture_experiment()

def main():
    parser = ap.ArgumentParser(
        "Deep Q-Learning for the CartPole environment.",
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument(
        "--run-ablation",
        type=bool,
        default=False,
        help="Run ablation study.",
        action=ap.BooleanOptionalAction
    )

    parser.add_argument(
        "--run-hyperparameters",
        type=bool,
        default=False,
        help="Run hyperparameters experiments (different learning rate, gamma, epsilon/temperature, architecture and policy.).",
        action=ap.BooleanOptionalAction
    )

    parser.add_argument(
        "--measure-runtimes",
        type=bool,
        default=False,
        help="Measure runtimes for different agents.",
        action=ap.BooleanOptionalAction
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
        help="Number of episodes to train for in the experiments.",
        nargs="?",
        action="store"
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Number of trials to run for each experiment.",
        nargs="?",
        action="store"
    )

    args = parser.parse_args()

    global TRAIN_EPISODES
    TRAIN_EPISODES = args.num_episodes

    global NUM_TRIALS
    NUM_TRIALS = args.num_trials
    

    if args.run_ablation:
        ablation_experiment()

    if args.run_hyperparameters:
        hyperparameters_experiment()

    if args.measure_runtimes:
        measure_runtimes()
    

if __name__ == "__main__":
    main()