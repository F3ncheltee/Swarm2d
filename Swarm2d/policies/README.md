# Agent Policies

This directory contains the logic that governs agent behavior in the `Swarm2d` environment. A "policy" is a strategy that an agent uses to map its observations of the world to actions. This folder is structured to accommodate a wide range of policies, from simple, hard-coded behaviors to complex, learned neural networks.

## Policy Structure

A policy is a class that, at a minimum, implements an `act(observation)` method. This method takes an agent's observation (a dictionary containing "self", "map", and "graph" data) and returns an action dictionary (containing "movement" and "pickup" commands).

The `policy_template.py` in the `working_policies` directory provides a basic skeleton for creating new policies.

## Policy Types

The policies are broadly categorized into simple, rule-based policies and more advanced, learned reinforcement learning (RL) policies.

### Core Policies

These are the primary, well-supported policies intended for general use.

-   **/heuristicPolicy**: Implements a scripted, rule-based policy. Agents follow a set of predefined rules to perform tasks like finding resources, bringing them back to the hive, and engaging enemies. This is a strong baseline for comparison.
-   **/working_policies/randomPolicy**: The most basic policy where agents take random actions at each step. This is often used as a simple baseline to measure the effectiveness of other policies.
-   **/working_policies/trainedRLPolicy**: A wrapper policy designed to load and use a pre-trained neural network model for decision-making.

### Reinforcement Learning (RL) Policy Components

This is the core of the research-focused part of the project, containing the components needed for training sophisticated RL agents. The structure generally follows an Actor-Critic model design.

-   **/RLPolicy**: Contains a basic `RLPolicy` class that defines a simple neural network. This serves as a starting point or a simple learned policy but is superseded by the more advanced actor-critic structures.

-   **/actors**: Contains the neural network architectures for the "actor." The actor is the component that takes an agent's observation as input and outputs an action. The subdirectories (`/MAAC`, `/NCA`, `/SHARED`) contain different experimental network architectures using Graph Neural Networks (GNNs) and attention mechanisms. These are designed for research into complex coordinated behaviors.

-   **/critics**: Contains the "critic" part of the actor-critic model. The critic's role is to evaluate the actions taken by the actor by estimating a value function (e.g., Q-value). This evaluation provides a learning signal to improve the actor's decision-making. Key components include:
    -   `advanced_criticGNN.py`: The neural network architecture for the critic.
    -   `ReplayBuffer.py`: An essential component for off-policy RL algorithms. It stores past experiences (state, action, reward, next state) that are sampled during training to update the networks.
    -   `updatecritic.py`: Contains the logic for the critic's learning update step.

### Other Policies

-   **/simple_policies**: Contains simplified or alternative versions of MARL and RL policies, likely used for specific experiments or debugging.
-   **/working_policies/simple_heuristic_policy.py**: A more streamlined version of the main heuristic policy.

## Usage

Policies from this directory can be selected in the `simulation_gui.py` to control the behavior of different teams of agents. They can also be specified as command-line arguments when using `run_simulation.py`. This modular structure allows for easy A/B testing between different policies and is central to the experimental nature of `Swarm2d`.
