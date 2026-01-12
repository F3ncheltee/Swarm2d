# Reinforcement Learning Training

This directory contains the pipeline and utilities for training reinforcement learning (RL) agents in the `Swarm2d` environment. It is divided into two main subdirectories: `training` (this directory) and `trainingCustom`.

-   **`/training` (This Directory):** Contains the original, simpler training setup, this is the template to start from cleanly. Here you should be able to implement your own training loop and start training in the environment.
-   **`/trainingCustom`:** Contains the current, more advanced, and primary training pipeline. For any new training runs, you should use the scripts in `trainingCustom`.

## Experimental Custom Training Pipeline (`/trainingCustom`)

The `/trainingCustom` directory holds the experimental, more involved scripts for training agents with the experimental policies, which serve as a showcase of what can be done in the pipeline. It does hold a few helper modules that can be expanded or reused, but it is mostly specific to the actor / critic policies from the /policies folder.

A short description of the pipeline:
-   `run_training.py`: **This is the main entry point for launching a training run.** It orchestrates the entire training process, including environment interaction, policy updates, logging, and checkpointing.
-   `config.py`: Contains the configuration settings for the training runs, including hyperparameters for the actor and critic networks.
-   `hyperparameter_search.py`: Implements a hyperparameter search using Optuna to find the best set of parameters for the policies.
-   `policyinstatiation.py`: A helper script that handles the instantiation of the different policy networks (MAAC, NCA, SharedActor) with their respective configurations.
-   `batchhelpers.py`: Provides utility functions for batching observations and hidden states, which is crucial for efficient GPU utilization during training.

## Template Training Pipeline (`/training`)

This directory contains the older training scripts, which are simpler and can be useful for debugging or as a reference.

-   `main.py`: A template training loop that demonstrates how to train a basic RL policy in the environment. It's a good starting point for understanding the basic mechanics and building custom training pipelines.
-   `TRAINING_STRATEGY.md`: A markdown file that outlines the high-level strategy, goals, and phases for the overall training process. It provides a roadmap for the research and development of the training pipeline.

## Core Training Components

Several files in this directory provide core functionalities used by the `trainingCustom` pipeline:

-   `checkpointing.py`: Implements the logic for saving and loading model checkpoints. This is crucial for resuming long training runs and for saving the best-performing models.
-   `curriculum_learning.py`: Contains the `CurriculumManager`, a training strategy where agents are progressively trained on more difficult tasks. This helps agents learn more effectively by starting with simpler scenarios.
-   `log_utils.py`: A utility for setting up and managing logging using TensorBoard. It logs key metrics such as rewards, losses, and episode lengths, which is essential for monitoring the training process.
-   `PlateauManager.py`: A tool to detect when the training process has "plateaued" (i.e., performance stops improving). It can be used to trigger adjustments, such as advancing the curriculum stage.
-   `utils.py`: A collection of helper functions used across different training scripts.
-   `observation_debug.py`: A debugging tool for visualizing the complex observations that agents receive, which is invaluable for verifying the correctness of the observation generation.

## Training Workflow

1.  **Configure:** Modify the settings in `trainingCustom/config.py` to define the environment, policy architectures, and hyperparameters.
2.  **Launch Training:** Execute `trainingCustom/run_training.py` to start the training process.
3.  **Training Loop:** The script will then:
    a.  Run episodes in the `Swarm2d` environment.
    b.  Collect experiences (observations, actions, rewards).
    c.  Store experiences in the `ReplayBuffer`.
    d.  Sample from the buffer to update the actor and critic networks.
    e.  Log progress using `log_utils.py` to TensorBoard.
    f.  Periodically save model checkpoints using `checkpointing.py`.
4.  **Monitor and Evaluate:** Use TensorBoard to monitor the training progress and evaluate the performance of the trained models.
