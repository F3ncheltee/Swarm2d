"""
A multi-agent training template for the Swarm2D environment.

This script trains a team of 5 agents using a shared SimpleRLPolicy
against a team of 5 agents controlled by a fixed heuristic policy.

Key features:
- Implements a 5v5 multi-agent scenario.
- Uses parameter sharing for the learning agents.
- Pits learners against a static, rule-based opponent team.
- Clear, commented structure for multi-agent action selection and experience gathering.
- Command-line arguments for configuration, including a debug flag.

To run this script:
    python Swarm2d/training/main.py --num_episodes 5000 --render
"""

import argparse
import os
import sys
import time
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

import torch
import numpy as np
from tqdm import tqdm

# Adjust the path to import from the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.simple_policies.simple_rl_policy import SimpleRLPolicy
from Swarm2d.policies.heuristicPolicy.heuristic_policy import HeuristicPolicy
from Swarm2d.training.utils import set_seeds, setup_logging
from Swarm2d.constants import REWARD_COMPONENT_KEYS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Template training script for Swarm2D.")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of episodes to train for.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum number of steps per episode.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--render", action="store_true", help="Render the environment during training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default="models/template_ma", help="Directory to save trained models.")
    parser.add_argument("--patience", type=int, default=100, help="Patience for learning rate scheduler (in episodes).")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Factor by which to reduce learning rate.")
    parser.add_argument("--summary_interval", type=int, default=100, help="Interval for printing episode summaries.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed per-step debug prints for a single agent.")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    setup_logging(args.save_dir)

    logging.info(f"Starting training with config: {args}")

    # --- 1. Initialization ---
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Environment Setup for 5v5 ---
    # Team 0: RL agents (learning)
    # Team 1: Heuristic agents (fixed policy)
    env = Swarm2DEnv(
        num_teams=2,
        num_agents_per_team=5,
        max_steps=args.max_steps,
        render_mode="human" if args.render else None
    )

    # --- Policy Instantiation ---
    # Get observation and action dimensions from the environment.
    obs_dim = env.observation_space['self'].shape[0]
    movement_dim = env.action_space['movement'].shape[0]
    pickup_dim = env.action_space['pickup'].n

    # RL Policy for Team 0 (parameter sharing)
    rl_policy = SimpleRLPolicy(obs_dim, movement_dim, pickup_dim, lr=args.learning_rate, gamma=args.gamma).to(device)

    # Learning rate scheduler to reduce LR on reward plateau
    scheduler = ReduceLROnPlateau(
        rl_policy.optimizer, 
        mode='max', 
        factor=args.lr_factor, 
        patience=args.patience,
    )

    # Heuristic Policy for Team 1
    heuristic_policy = HeuristicPolicy(action_space=env.action_space)

    # --- 2. Training Loop ---
    start_time = time.time()
    # Keep track of the last N episode rewards for a running average
    reward_buffer = deque(maxlen=100)
    best_avg_reward = -float('inf')
    episode_summary_data = []


    # tqdm provides a progress bar for the training loop
    progress_bar = tqdm(range(args.num_episodes), desc="Training Episodes", file=sys.stdout)
    for episode in progress_bar:
        obs, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        
        # Data structures for the learning agents (Team 0)
        num_rl_agents = env.num_agents_per_team
        episode_trajectories = [[] for _ in range(num_rl_agents)]
        
        total_episode_reward_team0 = 0
        total_episode_reward_team1 = 0
        reward_component_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        step_count = 0

        # The --debug flag now controls ONLY per-step prints
        if args.debug:
            logging.info("\n" + "="*30 + f" Episode {episode + 1} Debug " + "="*30)

        while not terminated and not truncated:
            actions_list = [None] * env.num_agents
            log_probs_rl_agents = [None] * num_rl_agents

            # --- Render the environment ---
            if args.render:
                env.render()

            # --- Action Selection for All Agents ---
            for agent_idx in range(env.num_agents):
                agent_team = env.agents[agent_idx]['team']
                agent_obs_dict = obs[agent_idx]

                if agent_team == 0:  # RL Team
                    # The RL policy is shared, so we use the same network for all agents in Team 0
                    agent_obs_tensor = torch.as_tensor(agent_obs_dict['self'], dtype=torch.float32, device=device)
                    action, log_prob = rl_policy.act(agent_obs_tensor)
                    actions_list[agent_idx] = action
                    
                    # Store log_prob for the update step. agent_idx is also the index in the RL team.
                    log_probs_rl_agents[agent_idx] = log_prob

                elif agent_team == 1:  # Heuristic Team
                    action = heuristic_policy.act(agent_obs_dict)
                    actions_list[agent_idx] = action

            # --- Step the Environment ---
            next_obs, rewards, terminated, truncated, info = env.step(actions_list)

            # --- Store Experience for RL Agents ---
            for agent_idx in range(num_rl_agents):
                # The SimpleRLPolicy expects a single scalar reward. We sum the components.
                reward_dict = rewards[agent_idx]
                reward = sum(reward_dict.values())
                episode_trajectories[agent_idx].append((log_probs_rl_agents[agent_idx], reward))
                total_episode_reward_team0 += reward
                for key, value in reward_dict.items():
                    reward_component_totals[key] += value

                # --- Debug Prints for a Single Agent (Controlled by --debug flag) ---
                if args.debug and agent_idx == 0:
                    action_taken = actions_list[agent_idx]
                    non_zero_rewards = {k: round(v, 2) for k, v in reward_dict.items() if v != 0}
                    if non_zero_rewards:
                        # Use logging instead of progress_bar.write
                        logging.debug(f"  Step {step_count} | Agent 0 Action: "
                            f"Move[{action_taken['movement'][0]:.2f}, {action_taken['movement'][1]:.2f}], "
                            f"Pickup[{action_taken['pickup']}] | "
                            f"Rewards: {non_zero_rewards}")
            
            # For logging, also track heuristic team reward
            for agent_idx in range(num_rl_agents, env.num_agents):
                total_episode_reward_team1 += sum(rewards[agent_idx].values())

            obs = next_obs
            step_count += 1

        # --- 3. Policy Update ---
        # At the end of the episode, a single update is applied to the shared policy
        # using the collected trajectories from all agents on the learning team.
        loss = rl_policy.update(episode_trajectories)

        # --- 4. Logging & Summaries ---
        avg_reward_team0 = total_episode_reward_team0 / num_rl_agents
        avg_reward_team1 = total_episode_reward_team1 / env.num_agents_per_team
        reward_buffer.append(avg_reward_team0)
        running_avg_reward = np.mean(reward_buffer)

        # Store data for the summary
        episode_summary_data.append({
            "Episode": episode + 1,
            "Total Reward Team 0": total_episode_reward_team0,
            "Avg Reward Team 0": avg_reward_team0,
            "Running Avg Reward": running_avg_reward,
            "Steps": step_count,
            "Loss": loss
        })

        # Print summary at the specified interval
        if (episode + 1) % args.summary_interval == 0:
            logging.info("\n" + "="*30 + f" Episode Summary ({episode - args.summary_interval + 2}-{episode + 1}) " + "="*30)
            avg_summary = {key: np.mean([d[key] for d in episode_summary_data]) for key in episode_summary_data[0]}
            for key, val in avg_summary.items():
                logging.info(f"  - Average {key}: {val:.2f}")
            episode_summary_data = [] # Reset for next interval
            logging.info("=" * 80 + "\n")


        # Step the scheduler with the running average reward
        # The scheduler will reduce the learning rate if the reward plateaus
        if len(reward_buffer) == 100: # Only step scheduler when buffer is full for stability
            scheduler.step(running_avg_reward)
        
        # Create a dictionary of the most important reward components for logging
        avg_reward_components = {key: val / num_rl_agents for key, val in reward_component_totals.items()}
        log_rewards = {
            'delivery': avg_reward_components.get('r_delivery', 0),
            'combat': avg_reward_components.get('r_combat_win', 0),
            'death': avg_reward_components.get('r_death', 0),
            'explore': avg_reward_components.get('r_exploration_intrinsic', 0)
        }

        # Update the progress bar with the latest metrics
        progress_bar.set_postfix({
            'Loss': f'{loss:.3f}',
            'RL_Reward': f'{avg_reward_team0:.2f}',
            'Heuristic_Reward': f'{avg_reward_team1:.2f}',
            'Steps': step_count,
            **log_rewards
        })

        # --- 5. Model Checkpointing ---
        # Save the model only if the running average reward is the best we've seen.
        # This prevents saving a "stupid" policy.
        if len(reward_buffer) == 100 and running_avg_reward > best_avg_reward:
            best_avg_reward = running_avg_reward
            save_path = os.path.join(args.save_dir, 'best_policy.pth')
            torch.save(rl_policy.state_dict(), save_path)
            # Use logging to record this event
            logging.info(f"\n---> New best model saved to {save_path} with avg reward: {best_avg_reward:.2f} <---")


    # --- 6. Cleanup ---
    env.close()
    end_time = time.time()
    logging.info(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
    final_save_path = os.path.join(args.save_dir, 'policy_final.pth')
    torch.save(rl_policy.state_dict(), final_save_path)
    logging.info(f"Final model saved to {final_save_path}")


if __name__ == '__main__':
    main()
