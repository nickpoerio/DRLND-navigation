[//]: # (Image References)

[image1]: DQN.png "DQN training"
[image2]: DDQN.png "DDQN training"
[image3]: DQN_prioritized_experience.png "DQN with prioritized experience training"


# Project 1: Navigation

For this project I decided to use a DQN implementation developed to solve the 'LunarLander' Openai gym environment, and then trying to improve it, introducing variations like DDQN and prioritized experience replay buffer.

## Learning Algorithm

### Deep QNetwork
The DQN algorithm is a modification of the Qlearning temporal difference algorithm, using deep neural network as a function approximations. Other important variations with respect to the vanilla Qlearning algorithm are:

- target network with soft updates
- experience replay

For exploration, I used................

The neural network used consists of.....

The reward history is shown in the following picture

![VanillaDQN_trained][image1]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken 518 episodes to solve the problem, that is, to get an average reward greater than 13


### Dual Deep QNetwork

### Deep QNetwork with prioritized experience replay buffer



