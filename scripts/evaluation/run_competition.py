"""
Multi-Policy Competition Benchmark
All 5 heuristic policies compete in the SAME environment!
Much faster than running separately - only need ~10 episodes
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

matplotlib.use('Agg')
from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.heuristicPolicy.heuristic_policy import HeuristicPolicy
from Swarm2d.policies.heuristicPolicy.aggressive_heuristic import AggressiveHeuristic
from Swarm2d.policies.heuristicPolicy.defensive_heuristic import DefensiveHeuristic
from Swarm2d.policies.heuristicPolicy.greedy_heuristic import GreedyHeuristic
from Swarm2d.policies.working_policies.randomPolicy.random_policy import RandomPolicy

class MultiPolicyCompetition:
    def __init__(self, num_episodes=10, max_steps=200):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # 5 teams, each with different policy!
        self.team_policies = {
            0: ('Balanced', HeuristicPolicy),
            1: ('Aggressive', AggressiveHeuristic),
            2: ('Defensive', DefensiveHeuristic),
            3: ('Greedy', GreedyHeuristic),
            4: ('Random', RandomPolicy)
        }
        
        # Environment config - 5 teams competing!
        self.env_config = {
            'num_teams': 5,              # 5 teams for 5 policies
            'num_agents_per_team': 6,    # Smaller teams for speed
            'num_resources': 120,        # Enough for all teams
            'num_obstacles': 12,
            'max_steps': max_steps,
            'render_mode': "headless",
            'debug': False,
            'use_gpu_occlusion_in_env': False,
            'use_pybullet_raycasting': False,
        }
        
        self.results = {name: {
            'total_reward': [],
            'resources_collected': [],
            'combat_wins': [],
            'combat_losses': [],
            'agent_deaths': [],
            'steps_survived': [],
        } for _, (name, _) in self.team_policies.items()}
        
    def run_episode(self):
        """Run one episode with all policies competing"""
        env = Swarm2DEnv(**self.env_config)
        
        # Create policy instances for each team
        policies = []
        for team_id in range(env.num_teams):
            policy_name, policy_class = self.team_policies[team_id]
            team_agents = env.num_agents_per_team
            # Each agent on the team gets the same policy type
            for _ in range(team_agents):
                policies.append(policy_class(env.action_space))
        
        obs_list, _ = env.reset()
        
        # Track metrics per team
        team_metrics = {team_id: {
            'total_reward': 0,
            'resources_collected': 0,
            'combat_wins': 0,
            'combat_losses': 0,
            'agent_deaths': 0,
            'steps_survived': 0,
        } for team_id in range(env.num_teams)}
        
        for step in range(self.max_steps):
            actions = []
            
            # Get actions from each agent's policy
            for i in range(env.num_agents):
                obs = obs_list[i]
                action = policies[i].act(obs)
                actions.append(action)
            
            # Step environment
            obs_list, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate metrics per team
            for i, reward_dict in enumerate(rewards):
                agent = env.agents[i]
                team_id = agent.get('team', -1)
                if team_id >= 0:
                    team_metrics[team_id]['total_reward'] += sum(reward_dict.values())
                    
                    # Track specific rewards
                    if 'r_death' in reward_dict and reward_dict['r_death'] < 0:
                        team_metrics[team_id]['agent_deaths'] += 1
                    if 'r_combat_win' in reward_dict and reward_dict['r_combat_win'] > 0:
                        team_metrics[team_id]['combat_wins'] += 1
                    if 'r_combat_lose' in reward_dict and reward_dict['r_combat_lose'] < 0:
                        team_metrics[team_id]['combat_losses'] += 1
            
            # Track resources delivered by each team
            if 'delivered_resource_ids_this_step' in infos and infos['delivered_resource_ids_this_step']:
                for res_id in infos['delivered_resource_ids_this_step']:
                    # Find which team delivered it (would need to track in environment)
                    # For now, we'll track total in infos
                    pass
            
            # Update steps survived
            for team_id in range(env.num_teams):
                team_metrics[team_id]['steps_survived'] = step + 1
            
            if terminated or truncated:
                break
        
        # Count resources collected per team by checking hive scores
        for team_id, hive in env.hives.items():
            if hive:
                resources = hive.get('resources_collected', 0)
                team_metrics[team_id]['resources_collected'] = resources
        
        env.close()
        return team_metrics
    
    def run_competition(self):
        """Run the multi-policy competition"""
        print("=" * 70)
        print("MULTI-POLICY COMPETITION - All Policies Competing Together!")
        print("=" * 70)
        print(f"Episodes: {self.num_episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        print(f"Teams: {self.env_config['num_teams']} (each with different policy)")
        print(f"Agents per team: {self.env_config['num_agents_per_team']}")
        print(f"Total agents: {self.env_config['num_teams'] * self.env_config['num_agents_per_team']}")
        print()
        for team_id, (name, _) in self.team_policies.items():
            print(f"  Team {team_id}: {name}")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            ep_start = time.time()
            print(f"Episode {episode + 1}/{self.num_episodes}...")
            
            team_metrics = self.run_episode()
            
            # Store results
            for team_id, metrics in team_metrics.items():
                policy_name = self.team_policies[team_id][0]
                for key in self.results[policy_name].keys():
                    self.results[policy_name][key].append(metrics[key])
            
            ep_time = time.time() - ep_start
            elapsed = time.time() - start_time
            rate = (episode + 1) / elapsed
            eta = (self.num_episodes - episode - 1) / rate
            
            print(f"  Completed in {ep_time:.1f}s | Rate: {rate:.2f} ep/s | ETA: {eta/60:.1f}m")
            
            # Show quick results
            for team_id in range(min(3, self.env_config['num_teams'])):
                policy_name = self.team_policies[team_id][0]
                reward = team_metrics[team_id]['total_reward']
                resources = team_metrics[team_id]['resources_collected']
                print(f"    {policy_name}: Reward={reward:.0f}, Resources={resources}")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("COMPETITION COMPLETE!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average per episode: {total_time/self.num_episodes:.1f}s")
        print("=" * 70)
    
    def print_results(self):
        """Print summary statistics"""
        print("\n" + "=" * 70)
        print("COMPETITION RESULTS")
        print("=" * 70)
        
        for policy_name, metrics in self.results.items():
            print(f"\n{policy_name}:")
            print("-" * 70)
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f"  {metric_name:20s}: {mean:8.2f} ± {std:6.2f}")
    
    def save_results(self, output_dir='competition_results'):
        """Save results and create visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        results_file = os.path.join(output_dir, f'competition_results_{timestamp}.json')
        json_results = {}
        for policy_name, metrics in self.results.items():
            json_results[policy_name] = {}
            for metric_name, values in metrics.items():
                json_results[policy_name][metric_name] = [float(v) for v in values]
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create visualization
        self._create_comparison_chart(output_dir, timestamp)
    
    def _create_comparison_chart(self, output_dir, timestamp):
        """Create comparison bar chart"""
        import os  # Fix: import os here
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Multi-Policy Competition Results', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('total_reward', 'Total Reward'),
            ('resources_collected', 'Resources Collected'),
            ('combat_wins', 'Combat Wins'),
            ('combat_losses', 'Combat Losses'),
            ('agent_deaths', 'Agent Deaths'),
            ('steps_survived', 'Steps Survived')
        ]
        
        policy_names = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            means = [np.mean(self.results[p][metric]) for p in policy_names]
            stds = [np.std(self.results[p][metric]) for p in policy_names]
            
            x = np.arange(len(policy_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
            
            ax.set_xlabel('Policy', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(policy_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f'competition_chart_{timestamp}.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {chart_file}")
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-policy competition benchmark')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Max steps per episode (default: 200)')
    parser.add_argument('--output-dir', type=str, default='competition_results',
                       help='Output directory (default: competition_results)')
    
    args = parser.parse_args()
    
    # Run competition
    competition = MultiPolicyCompetition(num_episodes=args.episodes, max_steps=args.max_steps)
    competition.run_competition()
    competition.print_results()
    competition.save_results(args.output_dir)

if __name__ == "__main__":
    main()

