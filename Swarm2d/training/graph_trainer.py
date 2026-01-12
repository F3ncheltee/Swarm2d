import numpy as np
import torch
import torch.optim as optim
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# Set backend to avoid display issues
matplotlib.use('Agg')

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.simple_policies.graph_mlp_policy import SimpleMLP
from Swarm2d.policies.heuristicPolicy.heuristic_policy import HeuristicPolicy

class GraphRLTrainer:
    def __init__(self, num_episodes=200, max_steps=300):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # MULTI-AGENT RL with HEURISTIC OPPONENTS
        self.env_config = {
            'num_teams': 2,                # 2 teams: learners vs heuristics
            'num_agents_per_team': 5,      # 5 vs 5 = 10 total agents
            'num_resources': 80,           # Enough resources for both teams
            'num_obstacles': 10,
            'max_steps': max_steps,
            'render_mode': "headless",
            'debug': False,
            'use_gpu_occlusion_in_env': False,
            'use_pybullet_raycasting': False,
        }
        
        # Create environment and policies
        self.env = Swarm2DEnv(**self.env_config)
        
        self.team_0_size = self.env_config['num_agents_per_team']
        self.team_1_size = self.env_config['num_agents_per_team']
        
        # Create RL policies for Team 0 only
        self.rl_policies = [SimpleMLP() for _ in range(self.team_0_size)]
        self.optimizers = [optim.Adam(policy.parameters(), lr=3e-4) for policy in self.rl_policies]
        
        # Create fixed heuristic policy for Team 1 (opponents)
        self.heuristic_policy = HeuristicPolicy(self.env.action_space)
        
        print(f"Multi-Agent RL Setup (Graph Based):")
        print(f"  - Total agents: {self.env.num_agents}")
        print(f"  - Team 0 (LEARNING): {self.team_0_size} agents with RL policies")
        print(f"  - Team 1 (OPPONENTS): {self.team_1_size} agents with heuristic policy")
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_resources = []
        self.training_times = []
        
    def collect_episode(self):
        """Collect one episode of experience"""
        obs_list, _ = self.env.reset()
        
        # Storage for episode (only for learning agents)
        trajectories = [{'log_probs': [], 'rewards': []} for _ in range(self.team_0_size)]
        episode_reward = 0
        resources_collected = 0
        
        for step in range(self.max_steps):
            actions = []
            
            # Get actions for each agent
            for i in range(self.env.num_agents):
                obs = obs_list[i]
                
                if i < self.team_0_size:
                    # Team 0: RL agents (learning)
                    action, log_prob = self.rl_policies[i].act(obs['graph'])
                    actions.append({'movement': action, 'pickup': 0})
                    trajectories[i]['log_probs'].append(log_prob)
                else:
                    # Team 1: Heuristic agents (fixed opponents)
                    action = self.heuristic_policy.act(obs)
                    actions.append(action)
            
            # Step environment
            obs_list, rewards, terminated, truncated, infos = self.env.step(actions)
            
            # Store rewards (only for learning agents)
            for i in range(self.team_0_size):
                reward_dict = rewards[i]
                r = sum(reward_dict.values())
                trajectories[i]['rewards'].append(r)
                episode_reward += r
            
            # Track resources
            if 'delivered_resource_ids_this_step' in infos and infos['delivered_resource_ids_this_step']:
                resources_collected += len(infos['delivered_resource_ids_this_step'])
            
            if terminated or truncated:
                break
        
        return trajectories, episode_reward, step + 1, resources_collected
    
    def train_episode(self):
        """Collect episode and update policies"""
        trajectories, episode_reward, episode_length, resources = self.collect_episode()
        
        # Update each learning agent's policy (Team 0 only)
        for i, (policy, optimizer, trajectory) in enumerate(zip(self.rl_policies, self.optimizers, trajectories)):
            if len(trajectory['rewards']) == 0:
                continue
            
            # Compute returns (simple Monte Carlo)
            returns = []
            G = 0
            for r in reversed(trajectory['rewards']):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
            
            # Compute policy loss
            policy_loss = []
            for log_prob, G in zip(trajectory['log_probs'], returns):
                policy_loss.append(-log_prob * G)
            
            if len(policy_loss) > 0:
                policy_loss = torch.stack(policy_loss).mean()
                
                # Update policy
                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        return episode_reward, episode_length, resources
    
    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("MULTI-AGENT RL TRAINING - Learning vs Heuristic Opponents")
        print("=" * 70)
        print(f"Episodes: {self.num_episodes}")
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            ep_start = time.time()
            
            # Train one episode
            episode_reward, episode_length, resources = self.train_episode()
            
            ep_time = time.time() - ep_start
            
            # Store stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_resources.append(resources)
            self.training_times.append(ep_time)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_resources = np.mean(self.episode_resources[-10:])
                elapsed = time.time() - start_time
                rate = (episode + 1) / elapsed
                eta = (self.num_episodes - episode - 1) / rate
                
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Reward: {episode_reward:.1f} (avg: {avg_reward:.1f}) | "
                      f"Resources: {resources} (avg: {avg_resources:.1f}) | "
                      f"Rate: {rate:.2f} ep/s | ETA: {eta/60:.1f}m")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    def save_results(self, output_dir='rl_training_results'):
        """Save training results and create learning curves"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        results = {
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'episode_lengths': [int(l) for l in self.episode_lengths],
            'episode_resources': [int(r) for r in self.episode_resources],
            'config': {
                'num_episodes': self.num_episodes,
                'max_steps': self.max_steps,
                'num_agents': self.env.num_agents,
                'algorithm': 'REINFORCE'
            }
        }
        
        results_file = os.path.join(output_dir, f'training_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create learning curves
        self._plot_learning_curves(output_dir, timestamp)
    
    def _plot_learning_curves(self, output_dir, timestamp):
        """Create learning curve plots"""
        
        # Moving average helper
        def moving_average(data, window=10):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RL Training Progress - Learning Demonstration', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 10:
            ma = moving_average(self.episode_rewards, window=10)
            ax.plot(range(9, len(self.episode_rewards)), ma, linewidth=2, label='Moving Avg (10)')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Total Reward', fontweight='bold')
        ax.set_title('Episode Rewards (Learning Curve)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Resources Collected
        ax = axes[0, 1]
        ax.plot(self.episode_resources, alpha=0.3, label='Raw')
        if len(self.episode_resources) > 10:
            ma = moving_average(self.episode_resources, window=10)
            ax.plot(range(9, len(self.episode_resources)), ma, linewidth=2, label='Moving Avg (10)')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Resources Collected', fontweight='bold')
        ax.set_title('Resources Collected (Task Performance)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Episode Length
        ax = axes[1, 0]
        ax.plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) > 10:
            ma = moving_average(self.episode_lengths, window=10)
            ax.plot(range(9, len(self.episode_lengths)), ma, linewidth=2, label='Moving Avg (10)')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Episode Length (steps)', fontweight='bold')
        ax.set_title('Episode Length (Survival)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Performance Statistics
        ax = axes[1, 1]
        window = 20
        if len(self.episode_rewards) >= window:
            windows = [self.episode_rewards[i:i+window] for i in range(0, len(self.episode_rewards), window)]
            means = [np.mean(w) for w in windows]
            stds = [np.std(w) for w in windows]
            x = np.arange(len(means)) * window + window/2
            
            ax.plot(x, means, marker='o', linewidth=2, markersize=8)
            ax.fill_between(x, np.array(means) - np.array(stds), 
                           np.array(means) + np.array(stds), alpha=0.3)
            ax.set_xlabel('Episode', fontweight='bold')
            ax.set_ylabel('Mean Reward ± Std', fontweight='bold')
            ax.set_title(f'Performance Over Time (window={window})', fontweight='bold')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor windowed stats', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f'learning_curves_{timestamp}.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
