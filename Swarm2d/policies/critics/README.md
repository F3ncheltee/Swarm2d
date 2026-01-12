# Critic Network and Logic

This directory contains the components related to the "critic" part of the actor-critic reinforcement learning framework. The critic's primary role is to evaluate the actions taken by the actor, providing a crucial learning signal that guides the actor toward better performance.

## Core Components

-   `advanced_criticGNN.py`: This script defines the neural network architecture for the critic. It uses a `UnifiedCriticCore` that processes a sequence of observations to estimate the value (e.g., Q-value) of state-action pairs. This is a key component of the Soft Actor-Critic (SAC) algorithm used in this project.

-   `updatecritic.py`: Contains the core logic for updating the critic's network weights. The `update_critic_fn_sac` function implements the critic loss function, which is based on minimizing the Bellman error. This script is where the critic "learns" to make more accurate value predictions.

-   `ReplayBuffer.py`: Implements the replay buffer, a fundamental component of off-policy RL algorithms like SAC. The buffer stores a large history of agent experiences (observations, actions, rewards). During training, batches of these experiences are sampled to update the critic and actor, which helps to stabilize the learning process and break temporal correlations.

-   `criticobservation.py`: This module is responsible for preparing the input data for the critic network. The critic often requires a different, more comprehensive view of the environment than the actor. This script  handles the aggregation of information from multiple agents and across time steps to construct the state representation that the critic uses for its value estimations. A crucial part here is that we test a "global critic" and a "limited critic" - with the fundamental difference in the observations they receive. While the "global critic" resembles more of a traditional, fully aware critic with a global view, the limited critic only receives the limited, aggreggated observation space of its "actors" team, therefore limiting it by only providing partial data.

-   `TITANhelpers.py`: A collection of helper functions, possibly related to the "TITAN" architecture or specific techniques used in the critic's observation processing.

## How It Works

The components in this directory work together during the training loop:

1.  **Experience Collection:** As agents interact with the environment, their experiences (`state`, `action`, `reward`, `next_state`) are stored in the `ReplayBuffer`.
2.  **Batch Sampling:** At each training step, a batch of experiences is sampled from the `ReplayBuffer`.
3.  **Critic Input Preparation:** `criticobservation.py` processes the sampled batch to create the specific input required by the critic network.
4.  **Critic Update:** The `update_critic_fn_sac` function in `updatecritic.py` uses this batch to calculate the loss and perform a backpropagation step on the `UnifiedCriticCore` network from `advanced_criticGNN.py`.
5.  **Actor Update:** The updated critic is then used to provide the learning signal for the actor update step (which is defined in the `/policies/actors/...` directories). The critic evaluates the actor's proposed actions, and this evaluation is used to improve the actor's policy.
