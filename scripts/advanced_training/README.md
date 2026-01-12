# Advanced Training Pipeline

This directory contains the primary, advanced training pipeline for the experimental, research-focused policies like `MAAC`, `NCA`, and `SharedActor`. This is the active development area for training sophisticated agents.

For general information about the training process and shared utilities, see the [README in the parent `/training` directory](../training/README.md).

## Core Components

-   `run_training.py`: **This is the main script for orchestrating a training trial.** It manages the entire training loop, including:
    -   Initializing policies, critics, and optimizers.
    -   Handling the main episode loop (resetting the environment, stepping, etc.).
    -   Collecting and batching agent experiences for the replay buffer.
    -   Triggering the update steps for the actor and critic networks.
    -   Managing logging, checkpointing, and evaluation phases.

-   `main.py`: The high-level entry point for the training process. Its primary role is to set up the environment and then call `run_training_trial` from `run_training.py`.

-   `config.py`: A centralized location for all training configurations. It defines the default hyperparameters for the critic and the different actor architectures (`MAAC`, `NCA`, `SharedActor`). This is the first place you should look when you want to modify training parameters.

-   `hyperparameter_search.py`: Implements an automated hyperparameter search using the Optuna library. This script can be used to systematically find the best set of hyperparameters for the policies by running multiple training trials with different configurations.

-   `policyinstatiation.py`: A helper module responsible for creating instances of the advanced policies (`MAACPolicy`, `NCA_PINSANPolicy`, `SharedActorPolicy`) and their corresponding optimizers. It reads configurations from `config.py` to ensure policies are set up correctly.

-   `batchhelpers.py`: Provides crucial utility functions for preparing data for the neural networks. This includes batching observations from multiple agents into a single tensor, handling graph data, and managing recurrent hidden states. These helpers are essential for efficient GPU utilization.

## Training Workflow

The advanced training pipeline is designed as follows:

1.  **Execution Start:** The process begins when `main.py` is run.
2.  **Environment Setup:** `main.py` initializes the `Swarm2DEnv`.
3.  **Trial Execution:** `main.py` calls the `run_training_trial` function within `run_training.py`.
4.  **Initialization:** `run_training_trial` uses `policyinstatiation.py` and `config.py` to create the actor policies, critics, and their optimizers. It also sets up the logger, replay buffers, and curriculum manager.
5.  **Training Loop:** The script enters the main training loop, where for each episode, it performs the following steps:
    a.  **Collect Data:** The environment is stepped, and experiences (observations, actions, rewards) are collected.
    b.  **Store Experience:** The collected experiences are stored in the team-specific `ReplayBuffer`.
    c.  **Sample and Batch:** When enough data is collected, a batch of experiences is sampled from the buffer. `batchhelpers.py` is used to format this data into batches suitable for the networks.
    d.  **Network Updates:** The batched data is used to update the critic and actor networks. The actual update logic is defined within the policy directories (e.g., `policies/critics/updatecritic.py`).
6.  **Logging and Saving:** Throughout the process, metrics are logged to TensorBoard, and model checkpoints are saved periodically.
7.  **Evaluation:** At regular intervals, the policies are put into evaluation mode, and their performance is measured over several episodes without exploration noise.
8.  **Hyperparameter Search (Optional):** The `hyperparameter_search.py` script wraps this entire workflow, running it multiple times with different parameters to find the optimal configuration.
