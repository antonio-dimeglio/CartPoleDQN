# Deep Q-Network - Cartpole Environment

The goal of the project is to implement a Deep Q-Network approach to the Cartpole Environment implementation provided in the Gymnasium Library
[[1]](https://github.com/Farama-Foundation/Gymnasium)
[[2]](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) .

## Environment

The environment was first described by Barto, Sutton and Anderson in [[3]](https://ieeexplore.ieee.org/document/6313077).
Here, a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

### Action space

The action space is ${0, 1}$, where $0$ means push the cart to the left while $1$ means push the cart to the right.

### State space

The state consists of 4 variables, namely

- $\xi \rightarrow$ the cart position, with values in the interval $[-4.8, 4.8]$

    An episode ends if $\xi_x$ leaves the interval $(-2.4, 2.4)$.

- $\vec{\xi} \rightarrow$ the cart velocity, with values in the interval $(-\infty, \infty)$.

- $\theta \rightarrow$ the pole angle, with values in the interval $[-24\degree, 24\degree]$.

    An episode ends if the angle leaves the interval $(-12\degree, 12\degree)$.

- $\vec{\theta} \rightarrow$ the pole angular velocity, with values in the interval $(-\infty, \infty)$.

The start state $s_0$ is set initially to a uniform random value $\mathcal{X} \in (-0.05, 0.05)$ for all 4 variables.

The end state is either one of the aforementioned conditions or if the episode length is greather than 500.

### Reward

A reward of $+1$ is received for every step taken, including the termination step, as we want to keep the pole upright for as long as possible.

## Usage

The libraries used for this implementation are:

- Gymnasium   (agent environment)
- Pytorch     (deep learning backend)
- Numpy       (data smoothing for plotting)
- Matplotlib  (plotting)
- tqdm        (loading bar for training)
- tabulate    (rendering of tables in command line)

If all these libraries are already installed then the entry point to run all the experiments provided, ```dqn.py``` can be directly ran. Otherwise, an environment.yml file is provided to install the required libraries.

To create an environment for these libraries the ```environment.yml``` file can be used, by pasting the following two lines into the terminal:

```bash
conda env create -f environment.yml
conda activate rlcartpole
```

In terms of possible experiments to run, different options are possible, namely:

- ```run-ablation``` performs an ablation study, comparing a complete DQN with one without experience replay, target model and both.
- ```run-hyperparameters``` performs multiple experiments, generating plot for different experiments testing the model behaviour when different hyperparameters (such as learning rate, model architecture, etc...) are changed.
- ```measure-runtimes``` performs a runtime experiment, where each agent (complete or otherwise) is ran for 100 epochs 100 times to test for the average runtime and the standard deviation.

Additionally, when running the script, it is possible to specify the number of episodes for each model used during each experiment, by using the flag ```--num_episodes```; this can also be done for the number of trials of each experiment (so for example we can specify that we want to collect data from the training of a model for 100 episodes 10 times), this is done by using the flag ```--num_trials```

For example, if we wanted to run all experiments available for 200 episodes for each agent for 10 episodes we would do:

```bash
python .\src\dqn.py --num_episodes 200 --num_trials 10 --run-ablation --run_hyperparameters --measure-runtimes
```
