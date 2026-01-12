"""
FAST Benchmark Script for Heuristic Policy Comparison
Optimized for speed - disables graph observations and uses minimal config
"""
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

matplotlib.use('Agg')  # Use non-interactive backend
from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.heuristicPolicy.heuristic_policy import HeuristicPolicy
from Swarm2d.policies.heuristicPolicy.aggressive_heuristic import AggressiveHeuristic
from Swarm2d.policies.heuristicPolicy.defensive_heuristic import DefensiveHeuristic
from Swarm2d.policies.heuristicPolicy.greedy_heuristic import GreedyHeuristic
from Swarm2d.policies.working_policies.randomPolicy.random_policy import RandomPolicy

class FastHeuristicBenchmark:
    def __init__(self, num_episodes=50, max_steps=500):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Policy configurations
        self.policies = {
            'Balanced': HeuristicPolicy,
            'Aggressive': AggressiveHeuristic,
            'Defensive': DefensiveHeuristic,
            'Greedy': GreedyHeuristic,
            'Random': RandomPolicy
        }
        
        # REALISTIC environment config for proper validation
        # More resources/obstacles/teams to show mechanics work properly
        self.env_config = {
            'num_teams': 4,              # Multiple teams for combat
            'num_agents_per_team': 8,    # Decent swarm size
            'num_resources': 150,        # Plenty of resources to collect
            'num_obstacles': 15,         # Obstacles for navigation
            'max_steps': max_steps,
            'render_mode': "headless",
            'debug': False,
            'use_gpu_occlusion_in_env': False,
            'use_pybullet_raycasting': False,  # Disabled for speed
            'movement_force_scale': 15.0,
        }
        
        self.results = {}
        
    def run_episode(self, policy_class, policy_name):
        """Run a single episode with the given policy"""
        env = Swarm2DEnv(**self.env_config)
        
        # Create policy instances for each agent
        policies = [policy_class(env.action_space) for _ in range(env.num_agents)]
        
        obs_list, info = env.reset()
        
        # Metrics to track
        episode_metrics = {
            'total_reward': 0,
            'resources_collected': 0,
            'combat_wins': 0,
            'combat_losses': 0,
            'agent_deaths': 0,
            'steps_survived': 0,
            'total_movement': 0,  # Track total movement to verify agents are active
            'unique_positions': set(),  # Track unique positions visited
        }
        
        for step in range(self.max_steps):
            actions = []
            for i in range(env.num_agents):
                agent_obs = obs_list[i]
                action = policies[i].act(agent_obs)
                actions.append(action)
            
            obs_list, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate metrics
            for reward_dict in rewards:
                episode_metrics['total_reward'] += sum(reward_dict.values())
            
            episode_metrics['steps_survived'] = step + 1
            
            # Track movement to verify agents are active
            for i, agent in enumerate(env.agents):
                if agent and agent.get('alive'):
                    pos = agent.get('pos')
                    if pos is not None:
                        # Track movement magnitude
                        vel = agent.get('vel', np.zeros(2))  # FIXED: 'vel' not 'velocity'
                        episode_metrics['total_movement'] += np.linalg.norm(vel)
                        # Track unique positions (rounded to grid)
                        grid_pos = (int(pos[0] / 10), int(pos[1] / 10))
                        episode_metrics['unique_positions'].add(grid_pos)
            
            # Extract info from environment
            if 'delivered_resource_ids_this_step' in infos and infos['delivered_resource_ids_this_step']:
                episode_metrics['resources_collected'] += len(infos['delivered_resource_ids_this_step'])
            
            # Track deaths from rewards
            for reward_dict in rewards:
                if 'r_death' in reward_dict and reward_dict['r_death'] < 0:
                    episode_metrics['agent_deaths'] += 1
                if 'r_combat_win' in reward_dict and reward_dict['r_combat_win'] > 0:
                    episode_metrics['combat_wins'] += 1
                if 'r_combat_lose' in reward_dict and reward_dict['r_combat_lose'] < 0:
                    episode_metrics['combat_losses'] += 1
            
            if terminated or truncated:
                break
        
        # Convert set to count before returning
        episode_metrics['unique_positions'] = len(episode_metrics['unique_positions'])
        
        env.close()
        return episode_metrics
    
    def run_benchmark(self):
        """Run benchmark for all policies"""
        print("=" * 70)
        print("FAST HEURISTIC POLICY BENCHMARK - Environment Validation")
        print("=" * 70)
        print(f"Episodes per policy: {self.num_episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        print(f"Teams: {self.env_config['num_teams']}, Agents per team: {self.env_config['num_agents_per_team']}")
        print("OPTIMIZED FOR SPEED - Graph obs disabled")
        print("=" * 70)
        print()
        
        for policy_name, policy_class in self.policies.items():
            print(f"\n[{policy_name}] Running {self.num_episodes} episodes...")
            start_time = time.time()
            
            # Initialize results storage
            self.results[policy_name] = {
                'total_reward': [],
                'resources_collected': [],
                'combat_wins': [],
                'combat_losses': [],
                'agent_deaths': [],
                'steps_survived': [],
                'total_movement': [],
                'unique_positions': [],
            }
            
            # Run episodes
            for episode in range(self.num_episodes):
                episode_metrics = self.run_episode(policy_class, policy_name)
                
                # Store metrics
                for key in self.results[policy_name].keys():
                    self.results[policy_name][key].append(episode_metrics[key])
                
                # Progress indicator
                if (episode + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (episode + 1) / elapsed
                    remaining = (self.num_episodes - episode - 1) / rate
                    print(f"  Episode {episode + 1}/{self.num_episodes} | Rate: {rate:.1f} ep/s | ETA: {remaining:.0f}s")
            
            elapsed = time.time() - start_time
            print(f"[{policy_name}] Completed in {elapsed:.2f}s ({elapsed/self.num_episodes:.2f}s per episode)")
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETED")
        print("=" * 70)
    
    def compute_statistics(self):
        """Compute mean and std for all metrics"""
        stats = {}
        for policy_name, metrics in self.results.items():
            stats[policy_name] = {}
            for metric_name, values in metrics.items():
                stats[policy_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats
    
    def print_results(self):
        """Print formatted results"""
        stats = self.compute_statistics()
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        # Print each metric
        metrics_to_show = ['total_reward', 'resources_collected', 'combat_wins', 
                          'combat_losses', 'agent_deaths', 'steps_survived',
                          'total_movement', 'unique_positions']
        
        for metric in metrics_to_show:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 70)
            for policy_name in self.policies.keys():
                mean = stats[policy_name][metric]['mean']
                std = stats[policy_name][metric]['std']
                print(f"  {policy_name:12s}: {mean:10.2f} ± {std:8.2f}")
    
    def save_results(self, output_dir='benchmark_results'):
        """Save results to JSON and create visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'fast_benchmark_results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for policy_name, metrics in self.results.items():
            json_results[policy_name] = {}
            for metric_name, values in metrics.items():
                json_results[policy_name][metric_name] = [float(v) for v in values]
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create visualizations
        self.create_visualizations(output_dir, timestamp)
    
    def create_visualizations(self, output_dir, timestamp):
        """Create comparison charts"""
        stats = self.compute_statistics()
        
        # Metrics to visualize
        metrics_to_plot = [
            ('total_reward', 'Total Reward'),
            ('resources_collected', 'Resources Collected'),
            ('combat_wins', 'Combat Wins'),
            ('steps_survived', 'Steps Survived'),
            ('agent_deaths', 'Agent Deaths')
        ]
        
        # Create bar charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Fast Heuristic Policy Comparison - Environment Validation', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            policy_names = list(self.policies.keys())
            means = [stats[p][metric]['mean'] for p in policy_names]
            stds = [stats[p][metric]['std'] for p in policy_names]
            
            x = np.arange(len(policy_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            ax.set_xlabel('Policy', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(policy_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f'fast_benchmark_comparison_{timestamp}.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {chart_file}")
        plt.close()
        
        # Create detailed comparison table
        self._create_comparison_table(output_dir, timestamp, stats)
    
    def _create_comparison_table(self, output_dir, timestamp, stats):
        """Create a detailed comparison table as an image"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        metrics = ['total_reward', 'resources_collected', 'combat_wins', 
                  'combat_losses', 'agent_deaths', 'steps_survived']
        
        table_data = [['Metric'] + list(self.policies.keys())]
        
        for metric in metrics:
            row = [metric.replace('_', ' ').title()]
            for policy_name in self.policies.keys():
                mean = stats[policy_name][metric]['mean']
                std = stats[policy_name][metric]['std']
                row.append(f'{mean:.1f}±{std:.1f}')
            table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2] + [0.16] * len(self.policies))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(self.policies) + 1):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style metric column
        for i in range(1, len(metrics) + 1):
            table[(i, 0)].set_facecolor('#E7E6E6')
            table[(i, 0)].set_text_props(weight='bold')
        
        plt.title('Detailed Metrics Comparison (Fast Benchmark)\nEnvironment Validation for Thesis', 
                 fontsize=14, fontweight='bold', pad=20)
        
        table_file = os.path.join(output_dir, f'fast_benchmark_table_{timestamp}.png')
        plt.savefig(table_file, dpi=300, bbox_inches='tight')
        print(f"Comparison table saved to: {table_file}")
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast benchmark of heuristic policies for environment validation')
    parser.add_argument('--episodes', type=int, default=50, 
                       help='Number of episodes per policy (default: 50)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = FastHeuristicBenchmark(num_episodes=args.episodes, max_steps=args.max_steps)
    start_time = time.time()
    benchmark.run_benchmark()
    benchmark.print_results()
    benchmark.save_results(args.output_dir)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results and visualizations saved to: {args.output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()

