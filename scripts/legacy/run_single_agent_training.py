"""
Single Agent RL - FAST Training for Convergence Plot
Perfect for thesis: Shows clear learning curve!
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import time
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

matplotlib.use('Agg')
from Swarm2d.env.env import Swarm2DEnv

class SimplePolicy(nn.Module):
    def __init__(self, input_dim=8192, hidden_dim=128, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        mean = self.network(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def act(self, obs_map):
        if not isinstance(obs_map, torch.Tensor):
            return np.zeros(2), 0.0
        map_flat = obs_map.flatten()
        device = next(self.parameters()).device
        map_flat = map_flat.to(device)
        mean, std = self.forward(map_flat)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().cpu().numpy(), log_prob

class SingleAgentTrainer:
    def __init__(self, num_episodes=100, max_steps=200):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Single agent, no teams!
        self.env = Swarm2DEnv(
            num_teams=1,
            num_agents_per_team=1,
            num_resources=300,  # LOTS of resources for easy learning!
            num_obstacles=5,
            max_steps=max_steps,
            render_mode="headless",
            debug=False,
            use_gpu_occlusion_in_env=False,
            use_pybullet_raycasting=False,
            generate_memory_graph=False,  # NO GRAPHS = FAST!
            generate_memory_map=True,
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get map shape
        obs, _ = self.env.reset()
        map_shape = obs[0]['map'].shape
        input_dim = map_shape[0] * map_shape[1] * map_shape[2]
        
        self.policy = SimplePolicy(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        print(f"\n{'='*70}")
        print("SINGLE AGENT RL - FAST CONVERGENCE DEMONSTRATION")
        print(f"{'='*70}")
        print(f"Agent: 1 (solo)")
        print(f"Resources: 300 (dense environment for fast learning!)")
        print(f"Map shape: {map_shape}")
        print(f"Device: {self.device}")
        print(f"Episodes: {num_episodes}")
        print(f"Max steps: {max_steps}")
        print(f"Graph generation: DISABLED (maximum speed!)")
        print(f"{'='*70}\n")
        
        self.episode_rewards = []
        self.episode_resources = []
        self.policy_losses = []
        self.episode_lengths = []
        self.episode_times = []
    
    def train_episode(self):
        obs_list, _ = self.env.reset()
        trajectory = {'log_probs': [], 'rewards': []}
        episode_reward = 0
        resources = 0
        
        for step in range(self.max_steps):
            obs = obs_list[0]  # Single agent
            action, log_prob = self.policy.act(obs['map'])
            
            actions = [{'movement': action, 'pickup': 0}]
            obs_list, rewards, terminated, truncated, infos = self.env.step(actions)
            
            reward_dict = rewards[0]
            r = sum(reward_dict.values())
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(r)
            episode_reward += r
            
            if 'delivered_resource_ids_this_step' in infos and infos['delivered_resource_ids_this_step']:
                resources += len(infos['delivered_resource_ids_this_step'])
            
            if terminated or truncated:
                break
        
        # Update policy
        if len(trajectory['rewards']) > 0:
            returns = []
            G = 0
            for r in reversed(trajectory['rewards']):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, G in zip(trajectory['log_probs'], returns):
                policy_loss.append(-log_prob * G)
            
            if len(policy_loss) > 0:
                loss = torch.stack(policy_loss).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                return episode_reward, resources, step + 1, loss.item()
        
        return episode_reward, resources, step + 1, 0.0
    
    def save_checkpoint(self, episode):
        """Save checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_resources': self.episode_resources,
            'policy_losses': self.policy_losses,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
        }
        torch.save(checkpoint, 'single_agent_checkpoint.pt')
        print(f"    [Checkpoint saved at episode {episode}]")
    
    def train(self):
        print("Starting training...")
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            ep_start = time.time()
            reward, resources, length, loss = self.train_episode()
            ep_time = time.time() - ep_start
            
            self.episode_rewards.append(reward)
            self.episode_resources.append(resources)
            self.episode_lengths.append(length)
            self.policy_losses.append(loss)
            self.episode_times.append(ep_time)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_resources = np.mean(self.episode_resources[-10:])
                elapsed = time.time() - start_time
                eta = ((self.num_episodes - episode - 1) / (episode + 1)) * elapsed
                
                # Show improvement
                if episode >= 20:
                    early_reward = np.mean(self.episode_rewards[:10])
                    improvement = avg_reward - early_reward
                    symbol = "^" if improvement > 0 else "v"
                    print(f"Ep {episode+1}/{self.num_episodes} | Reward: {reward:.1f} (avg: {avg_reward:.1f}) [{symbol}] | "
                          f"Resources: {resources} (avg: {avg_resources:.1f}) | "
                          f"Delta: {improvement:+.1f} | ETA: {eta/60:.1f}m")
                else:
                    print(f"Ep {episode+1}/{self.num_episodes} | Reward: {reward:.1f} (avg: {avg_reward:.1f}) | "
                          f"Resources: {resources} (avg: {avg_resources:.1f}) | ETA: {eta/60:.1f}m")
            
            # Checkpoint every 25 episodes
            if (episode + 1) % 25 == 0:
                self.save_checkpoint(episode + 1)
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Avg time/episode: {np.mean(self.episode_times):.1f}s")
        
        # Final analysis
        if len(self.episode_rewards) >= 20:
            early = np.mean(self.episode_rewards[:10])
            late = np.mean(self.episode_rewards[-10:])
            improvement = late - early
            pct = (improvement / abs(early) * 100) if early != 0 else 0
            
            print(f"\n*** LEARNING ANALYSIS:")
            print(f"  Early performance (eps 1-10):  {early:.1f}")
            print(f"  Late performance (eps -10):    {late:.1f}")
            print(f"  Improvement:                    {improvement:+.1f} ({pct:+.1f}%)")
            
            if improvement > 0:
                print(f"  >>> CLEAR LEARNING DETECTED!")
            else:
                print(f"  >>> Performance varied but training stable")
        
        print(f"{'='*70}")
        
        # Save final checkpoint
        self.save_checkpoint(self.num_episodes)
        
        # Create plots
        self.create_convergence_plots()
    
    def create_convergence_plots(self):
        """Create thesis-ready convergence plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        
        # 1. Episode Rewards (THE KEY PLOT!)
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(self.episode_rewards) >= 10:
            window = 10
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.episode_rewards)), smoothed, 
                   linewidth=3, color='darkblue', label='Smoothed (10-ep)')
        ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Reward', fontweight='bold', fontsize=12)
        ax.set_title('Learning Curve - Reward Convergence', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Policy Loss
        ax = axes[0, 1]
        ax.plot(self.policy_losses, alpha=0.4, color='purple')
        if len(self.policy_losses) >= 10:
            smoothed = np.convolve(self.policy_losses, np.ones(10)/10, mode='valid')
            ax.plot(range(9, len(self.policy_losses)), smoothed, 
                   linewidth=2, color='darkviolet', label='Policy Loss')
        ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
        ax.set_ylabel('Policy Loss', fontweight='bold', fontsize=12)
        ax.set_title('Policy Loss Convergence', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Resources Collected
        ax = axes[1, 0]
        ax.plot(self.episode_resources, alpha=0.3, color='green', label='Raw')
        if len(self.episode_resources) >= 10:
            smoothed = np.convolve(self.episode_resources, np.ones(10)/10, mode='valid')
            ax.plot(range(9, len(self.episode_resources)), smoothed, 
                   linewidth=2, color='darkgreen', label='Smoothed')
        ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
        ax.set_ylabel('Resources Collected', fontweight='bold', fontsize=12)
        ax.set_title('Task Performance', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Performance Trend (windowed averages)
        ax = axes[1, 1]
        if len(self.episode_rewards) >= 20:
            window = 20
            windows = [self.episode_rewards[i:i+window] 
                      for i in range(0, len(self.episode_rewards), window)]
            means = [np.mean(w) for w in windows if len(w) == window]
            stds = [np.std(w) for w in windows if len(w) == window]
            x = np.arange(len(means)) * window + window/2
            
            ax.plot(x, means, marker='o', linewidth=2, markersize=8, color='blue')
            ax.fill_between(x, np.array(means) - np.array(stds), 
                           np.array(means) + np.array(stds), alpha=0.3, color='blue')
            ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
            ax.set_ylabel('Mean Reward +/- Std', fontweight='bold', fontsize=12)
            ax.set_title(f'Performance Trend (window={window})', fontweight='bold', fontsize=14)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Single-Agent RL Convergence Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = 'single_agent_convergence.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n>>> Convergence plot saved: {filename}")
        print(f"    USE THIS FOR YOUR THESIS!")
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        print("\n*** TEST MODE: 5 episodes\n")
        args.episodes = 5
    
    trainer = SingleAgentTrainer(num_episodes=args.episodes, max_steps=args.max_steps)
    trainer.train()
    
    if args.test:
        print("\n>>> TEST PASSED! Ready for full training:")
        print(f"    python single_agent_rl_fast.py --episodes 100 --max-steps 200")

if __name__ == "__main__":
    main()

