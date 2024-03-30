from Agent import *
import numpy as np 
import matplotlib.pyplot as plt
import argparse as ap
from tabulate import tabulate

TRAIN_EPISODES = 500
NUM_TRIALS = 5
WINDOW_LENGTH = 20

def moving_average(data, window_length):
    return np.convolve(data, np.ones(window_length), 'valid') / window_length

def measure_runtimes():
    print("Measuring runtimes...")

    agents = ["DQN-EN-TN", "DQN-EN", "DQN-TN", "DQN"]
    uses_memory = [False, False, True, True]
    uses_target = [False, True, False, True]

    runtimes_avg = []
    runtimes_std = []

    for i, curr_agent in enumerate(agents):
        print(f"Measuring runtime for {curr_agent}...")
        agent = Agent(requires_memory=uses_memory[i], requires_target=uses_target[i])
        runtime_curr_agent = agent.benchmark_training()

        runtimes_avg.append(np.mean(runtime_curr_agent))
        runtimes_std.append(np.std(runtime_curr_agent))


    table = []
    for i in range(len(agents)):
        table.append([agents[i], runtimes_avg[i], runtimes_std[i]])

    print(tabulate(table, headers=["Agent", "Average Runtime (s)", "Standard Deviation Runtime (s)"]))
    print("Runtime measurement complete.")

def ablation_study():
    print("Running ablation study...")
    results = np.zeros((4, TRAIN_EPISODES, NUM_TRIALS))
    agents = ["DQN-ER-TN", "DQN-ER", "DQN-TN", "DQN"]
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
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=r"$\alpha$ = {}".format(learning_rates[i]))

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Rate Experiment")
    plt.savefig("learning_rate_experiment.png")
    plt.clf()
    print("Learning rate experiment complete.")

def policy_experiment():
    print("Running policy experiment...")
    epsilons = [0.1, 0.2, 0.5]
    temperatures = [0.1, 1.0, 2.0]

    results_epsilons = np.zeros((len(epsilons), TRAIN_EPISODES, NUM_TRIALS))
    results_temperatures = np.zeros((len(temperatures), TRAIN_EPISODES, NUM_TRIALS))
    
    for i, epsilon in enumerate(epsilons):
        print(f"Training with epsilon {epsilon}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(epsilon=epsilon, requires_memory=True, requires_target=True)
            results_epsilons[i, :, j] = agent.train(TRAIN_EPISODES)

    print("Epsilon experiment complete.")

    for i, temperature in enumerate(temperatures):
        print(f"Training with temperature {temperature}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(temperature=temperature, requires_memory=True, requires_target=True)
            results_temperatures[i, :, j] = agent.train(TRAIN_EPISODES)
    
    print("Temperature experiment complete.")

    for i in range(len(epsilons)):
        plt.plot(moving_average(np.mean(results_epsilons[i], axis=1), WINDOW_LENGTH), label=r"$\epsilon$ = {}".format(epsilons[i]))

    for i in range(len(temperatures)):
        plt.plot(moving_average(np.mean(results_temperatures[i], axis=1), WINDOW_LENGTH), label=r"$\tau$ = {}".format(temperatures[i]))

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
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=r"$b$ = {}".format(batch_sizes[i]))

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
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=r"$m$ = {}".format(memory_sizes[i]))

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Memory Size Experiment")
    plt.savefig("memory_size_experiment.png")
    plt.clf()
    print("Memory size experiment complete.")

def update_frequency_experiment():
    print("Running update frequency experiment...")
    update_frequencies = [100, 200, 500]
    results = np.zeros((len(update_frequencies), TRAIN_EPISODES, NUM_TRIALS))

    for i, update_frequency in enumerate(update_frequencies):
        print(f"Training with update frequency {update_frequency}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(update_frequency=update_frequency, requires_target=True, requires_memory=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(len(update_frequencies)):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=r"$u$ = {}".format(update_frequencies[i]))

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Update Frequency Experiment")
    plt.savefig("update_frequency_experiment.png")
    plt.clf()
    print("Update frequency experiment complete.")

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
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=r"$h$ = {}".format(hidden_sizes[i]))

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Architecture Experiment")
    plt.savefig("architecture_experiment.png")
    plt.clf()
    print("Architecture experiment complete.")

def dqn_vs_dueling_dqn():
    print("Running DQN vs Dueling DQN experiment...")
    agents = ["DQN", "Dueling DQN"]
    results = np.zeros((2, TRAIN_EPISODES, NUM_TRIALS))

    for i, curr_agent in enumerate(agents):
        print(f"Training {curr_agent}...")
        for j in range(NUM_TRIALS):
            print(f"Trial {j + 1}/{NUM_TRIALS}...")
            agent = Agent(dueling = True if i == 1 else False, requires_memory=True, requires_target=True)
            results[i, :, j] = agent.train(TRAIN_EPISODES)

    for i in range(2):
        plt.plot(moving_average(np.mean(results[i], axis=1), WINDOW_LENGTH), label=agents[i])

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN vs Dueling DQN Experiment")
    plt.savefig("dqn_vs_dueling_dqn_experiment.png")
    plt.clf()
    print("DQN vs Dueling DQN experiment complete.")

def hyperparameters_experiment():
    learning_rate_experiment()
    batch_size_experiment()
    memory_size_experiment()
    architecture_experiment()
    update_frequency_experiment()
    policy_experiment()
    dqn_vs_dueling_dqn()


    print("Hyperparameters experiment complete.")

    
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
        "--num-episodes",
        type=int,
        default=500,
        help="Number of episodes to train for in the experiments.",
        nargs="?",
        action="store"
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials to run for each experiment.",
        nargs="?",
        action="store"
    )

    parser.add_argument(
        "--window-length",
        type=int,
        default=20,
        help="Window length for moving average.",
        nargs="?",
        action="store"
    )

    args = parser.parse_args()
    
    global TRAIN_EPISODES
    TRAIN_EPISODES = args.num_episodes

    global NUM_TRIALS
    NUM_TRIALS = args.num_trials
    
    global WINDOW_LENGTH
    WINDOW_LENGTH = args.window_length


    if args.run_ablation:
        ablation_study()

    if args.run_hyperparameters:
        hyperparameters_experiment()

    if args.measure_runtimes:
        measure_runtimes()
    

if __name__ == "__main__":
    main()