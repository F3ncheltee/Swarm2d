# Swarm2D RL Project

This repository contains a multi-agent reinforcement learning (MARL) environment and training scripts for swarm robotics scenarios.

## Project Structure

- **Swarm2d/**: The core environment and policy packages.
  - `env/`: Environment logic (physics, rendering, rewards).
  - `policies/`: Agent policies (RL, Heuristic).
  - `training/`: Training utilities and trainer classes.

- **scripts/**: Executable scripts for training, evaluation, and visualization.
  - `advanced_training/`: **Primary training scripts** for advanced models (NCA, MAAC, Shared GNN).
    - `train.py`: Main entry point for advanced CTDE training.
  - `simple_training/`: Training scripts for simpler map-based observations.
    - `train.py`: Entry point for map-based PPO training.
  - `legacy/`: Older training scripts (Graph, Single Agent, etc.).
  - `visualization/`: Scripts to visualize agent behavior.
    - `visualize_policy.py`: Watch a trained agent.
    - `generate_heatmap.py`: Generate spatial heatmaps of agent exploration.
    - `record_video.py`: Record gameplay videos.
  - `evaluation/`: Scripts for benchmarking and metrics.
    - `run_competition.py`: Run competitions between policies.
    - `benchmark_heuristics_fast.py`: Benchmark heuristic baselines.

- **results/**: Output directory for training logs, checkpoints, and media.
  - `logs/`: Training logs (CSV/JSON).
  - `media/`: Generated videos and plots.
  - `rl_training_results_*/`: Checkpoints from training runs.

## Usage

### Advanced Training (Recommended)

To train advanced agents (MAAC, NCA, Shared) with full observation modalities:
```bash
python scripts/advanced_training/train.py
```

### Simple Training

To train a baseline map-based agent:
```bash
python scripts/simple_training/train.py --scenario unified
```

### Visualization

To visualize a trained policy:
```bash
python scripts/visualization/visualize_policy.py --scenario resource --checkpoint results/rl_training_results_cnn_resource_fast_v4/checkpoint_latest.pt
```

### Evaluation

To run a heuristic benchmark:
```bash
python scripts/evaluation/benchmark_heuristics_fast.py
```

## Installation

Ensure you have the required dependencies installed (see `requirements.txt`).
Run scripts from the project root directory. The scripts automatically add the project root to `sys.path`.
