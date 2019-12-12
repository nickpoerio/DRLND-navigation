[//]: # (Image References)

[image1]: DQN.png "DQN training"
[image2]: DDQN.png "DDQN training"
[image3]: DQN_exprep.png "DQN with prioritized experience training"


# Project 1: Navigation

For this project I decided to use a DQN implementation developed to solve the 'LunarLander' Openai gym environment, and then trying to improve it, introducing variations like DDQN and prioritized experience replay buffer.

## Learning Algorithm

### Deep QNetwork
The DQN algorithm is a modification of the Qlearning temporal difference algorithm, using deep neural network as a function approximations. Other important variations with respect to the vanilla Qlearning algorithm are:

- target network
- experience replay

See the original paper for more details..........
A further modification to the paper that has been introduced is to use a target network soft-update instead of a targed network update period. This assures a more progressive learning behavior over time.

For exploration, I used an epsilon-greedy approach with an 'epsilon_decay' factor in order to assure an asymptotic decrease of the epsilon, so to reduce exploration when the knowledge is increased.

The neural network used consists of 3 fully connected layers, using relu activations for the first 2 (hidden) and a linear activation for the output.

The reward history during training is shown in the following picture:

![DQN_trained][image1]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken 518 episodes to solve the problem, that is, in order to get an average reward greater than 13


### Dual Deep QNetwork
Modifying the temporal difference calculation...............
No modification has been introduced to the hyperparameters.

The reward history during training is shown in the following picture:

![DDQN_trained][image2]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken ......... episodes to solve the problem, that is, in order to get an average reward greater than 13

### Deep QNetwork with prioritized experience replay buffer

The replay buffer sampling has been modified in order to ........................

The reward history during training is shown in the following picture:
No modification has been introduced to the DQN hyperparameters, ...............

![DQN_prioritized_experience][image2]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken .......... episodes to solve the problem, that is, in order to get an average reward greater than 13

## Possible improvements



