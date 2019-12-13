[//]: # (Image References)

[image1]: DQN.png "DQN training"
[image2]: DDQN.png "DDQN training"
[image3]: DQN_exprep.png "DQN with prioritized experience training"


# Project 1: Navigation

For this project I decided to use a DQN implementation developed to solve the 'LunarLander' Openai gym environment, and then trying to improve it, introducing variations like DDQN and prioritized experience replay buffer.

## Learning Algorithm

### Deep QNetwork
The DQN algorithm is a modification of the Qlearning temporal difference algorithm, using deep neural network as a function approximations. Other important variations with respect to the vanilla Qlearning algorithm are:

- target network: in order to avoid instability issues, the expected Q-value at the time step t+1 is calculated using a network which is frozen periodically
- experience replay buffer: the learning steps are carried on by mini-batches backpropagations, after sampling randomly from a buffer of memory, in order to avoid that too much correlated transitions drive the process to overfitting.

See the original paper for more details: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.  
A further modification with respect to the paper is to use a target network soft-update (θ_target = τ*θ_local + (1 - τ)*θ_target) instead of a targed network update period. This assures a more progressive learning behavior over time.

For exploration, I used an epsilon-greedy approach with an 'epsilon_decay' factor in order to assure an asymptotic decrease of the epsilon, so to reduce exploration when the knowledge is increased.

The neural network used consists of 3 fully connected layers, with relu activations for the first 2 (hidden) and a linear activation for the output.  The reward history during training is shown in the following picture:

![DQN_trained][image1]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken 518 episodes to solve the problem, that is, in order to get an average reward greater than 13. The related weight file is 'checkpoint.pth'.


### Dual Deep QNetwork
Modifying the temporal difference calculation using the local policy in order to choose the action, while the target network for calulating the action-value, should lead to a less overestimation of Q, especially when learning proceeds.
No modification has been introduced to the hyperparameters.  

See the original paper for more details: https://arxiv.org/abs/1509.06461.  

The reward history during training is shown in the following picture:

![DDQN_trained][image2]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken 529 episodes to solve the problem, that is, in order to get an average reward greater than 13. The related weight file is 'checkpoint_ddqn.pth'.

### Deep QNetwork with prioritized experience replay buffer

The replay buffer sampling has been modified assigning to each transition a probability proportional to the related temporal difference error.
An importance weight has been introduced in order to balance the bias of a non uniform sampling, together with two exponent coefficients ('alpha' and 'beta') respectively to the probability and the sempling weight, in order to avoid overfitting to the transitions with highest TD error.  

See the original paper for more details: https://arxiv.org/abs/1511.05952.  

The reward history during training is shown in the following picture:
No modification has been introduced to the DQN hyperparameters, while I've made some tests on 'alpha' and 'beta'.

![DQN_prioritized_experience][image2]

that is also visible in the Navigation.ipynb file together with a verbose logging of average rewards over the last 100 steps: it has taken 555 episodes to solve the problem, that is, in order to get an average reward greater than 13. The related weight file is 'checkpoint_dqnprio.pth'.

## Possible improvements

The 3 approaches used have lead to very similar results. It is possible that the hyperparameters optimized for DQN should have been further refined, i.e. in order to cope with the fact that DDQN is more accurate in estimating the Q-value.  In general, the path to improving the performance of this approach would be certainly implementing an algorithm that is fully comprehensive of the latest innovations in the field of Value based Q-Learning, like Rainbow (https://arxiv.org/abs/1710.02298).

