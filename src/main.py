import argparse as ap 
from Actor import Actor
from Model import * 
import gymnasium as gym
from Experiment import * 

def main():
    parser = create_parser()

    args = parser.parse_args()

    # Create the environment
    env = gym.make("CartPole-v0")





def create_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        prog = "DQN for CartPole problem.",
        description = "This is the entry point for the DQN experiment. It trains a DQN agent to solve the CartPole problem.",
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )

    # Actor parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="The discount factor to be used for training."
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="The tau value to be used for training."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The learning rate to be used for training."
    )

    parser.add_argument(
        "--memory_size",
        type=int,
        default=10000,
        help="The memory size to be used for training."
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="eps_greedy",
        help="The policy to be used for training."
    )

    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=0.9,
        help="The starting value for epsilon."
    )

    parser.add_argument(
        "--epsilon_end",
        type=float,
        default=0.05,
        help="The ending value for epsilon."
    )

    parser.add_argument(
        "--epsilon_decay",
        type=int,
        default=1000,
        help="The decay value for epsilon."
    )

    parser.add_argument(
        "--temp_start",
        type=float,
        default=0.9,
        help="The starting value for tau for boltzmann."
    )

    parser.add_argument(
        "--temp_end",
        type=float,
        default=0.05,
        help="The ending value for tau for boltzmann."
    )

    parser.add_argument(
        "--temp_decay",
        type=int,
        default=1000,
        help="The decay value for tau for boltzmann."
    )

    # Check if model must be loaded from file

    parser.add_argument(
        "--load_model",
        type=str,
        default=False,
        help="The model to be loaded from file."
    )

    parser.add_argument(
        "--load_path",
        type=str,
        default="./model.pt",
        help="The path to the model file."
    )

    # Check if model must be saved to file

    parser.add_argument(
        "--save_model",
        type=str,
        default=False,
        help="The model to be saved to file."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./model.pt",
        help="The path to the model file."
    )


    return parser


if __name__ == "__main__":
    main()