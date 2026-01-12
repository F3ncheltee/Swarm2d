# Basic RL Policy

This directory contains a very simple, standalone reinforcement learning policy.

## Description

The `RLPolicy` in `rl_policy.py` is a basic neural network implemented using `torch.nn.Module`. It consists of a simple Multi-Layer Perceptron (MLP) that takes an agent's "self" observation vector as input and produces outputs for movement and pickup actions.

### Purpose

This policy serves as a foundational example of a learned policy. It is not integrated into the advanced actor-critic training pipeline (`/policies/actors` and `/policies/critics`). Instead, it's intended for:

-   **Demonstration:** Showing the basic structure of a PyTorch-based policy.
-   **Testing:** Providing a simple learned agent for debugging the environment without the complexity of the full training framework.
-   **A Starting Point:** Acting as a simple template that can be expanded upon.

## Usage

This policy is not used in the main training script (`trainingCustom/run_training.py`). It could be used in a simpler, custom training loop, similar to the one in `training/main.py`, which is designed for such simple, self-contained policies.
