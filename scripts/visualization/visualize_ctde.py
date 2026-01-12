import torch
import numpy as np
import time
import argparse
import os
import sys

# Add current directory to path so we can import Swarm2d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.env.env import Swarm2DEnv
from scripts.training.run_ctde_training import PPOAgent, get_global_critic_obs

def visualize(checkpoint_path, scenario='unified'):
    # --- CONFIGURATION (Must match training) ---
    env_config = {
        'num_teams': 2,
        'num_agents_per_team': 10,
        'max_steps': 500, # Longer visualization
        'render_mode': "human", # GUI MODE
        'width': 500, 'height': 500,
        'generate_memory_graph': False,
        'generate_memory_map': False,
        'sensing_range_fraction': 0.10,
    }
    
    # Apply Scenario Settings (Must match training)
    if scenario == 'resource':
        env_config.update({'num_resources': 60, 'num_obstacles': 5})
    elif scenario == 'combat':
        env_config.update({'num_resources': 10, 'num_obstacles': 10})
    elif scenario == 'unified':
         env_config.update({
            'num_resources': 120, 'num_obstacles': 10,
            # Rewards don't matter for viz, but good to keep consistency
        })
        
    print(f"Initializing Environment for {scenario}...")
    env = Swarm2DEnv(**env_config)
    
    # Init Agent Architecture
    # We need a dummy reset to get shapes
    sample_obs, _ = env.reset()
    map_ch = sample_obs[0]['map'].shape[0]
    map_sz = sample_obs[0]['map'].shape[1]
    self_dim = sample_obs[0]['self'].shape[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    agent = PPOAgent(map_ch, map_sz, self_dim, device=device)
    
    # Load Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load State Dicts (Strict=False to be safe against minor version diffs)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
    # We don't need the critic for inference/visualization
    
    agent.actor.eval() # Set to eval mode
    
    # Heuristic for Team 1 (The Opponents)
    from Swarm2d.policies.heuristicPolicy.map_heuristic import MapHeuristic
    heuristic = MapHeuristic(env.action_space)
    
    print("\nStarting Visualization... (Press Ctrl+C to stop)")
    
    # Initial Render to open window
    env.render()
    
    try:
        while True: # Loop episodes
            obs_list, _ = env.reset()
            global_critic_map = get_global_critic_obs(env)
            
            steps = 0
            ep_reward = 0
            # Track detailed metrics for visualization
            ep_metrics = {
                "Deliv": 0.0, "Combat": 0.0, "Surv": 0.0, "Misc": 0.0
            }
            
            # Pause briefly at start of episode
            time.sleep(1.0)
            
            while True:
                # Handle GUI events
                # Force render, ignore return value for first step to prevent premature closing
                is_open = env.render()
                if not is_open and steps > 5: 
                    print("Window closed.")
                    return

                actions = []
                num_agents = len(env.agents)
                half_agents = num_agents // 2
                
                # Team 0 (RL Agent)
                with torch.no_grad():
                    for i in range(half_agents):
                        o = obs_list[i]
                        # Convert to tensor
                        om = torch.tensor(o['map'], dtype=torch.float32).unsqueeze(0).to(device)
                        osf = torch.tensor(o['self'], dtype=torch.float32).unsqueeze(0).to(device)
                        
                        mv, pk, _, _, _ = agent.get_action(om, osf, global_critic_map)
                        actions.append({'movement': mv, 'pickup': pk})
                
                # Team 1 (Heuristic)
                for i in range(half_agents, num_agents):
                    actions.append(heuristic.act(obs_list[i]))
                    
                # Step
                next_obs, rewards, term, trunc, _ = env.step(actions)
                
                # Track Team 0 Reward Breakdown
                for i in range(half_agents):
                    agent_rew = rewards[i]
                    total_r = sum(agent_rew.values())
                    ep_reward += total_r
                    
                    for k, v in agent_rew.items():
                        if 'delivery' in k: ep_metrics['Deliv'] += v
                        elif 'combat' in k or 'win' in k or 'lose' in k: ep_metrics['Combat'] += v
                        elif 'death' in k or 'health' in k: ep_metrics['Surv'] += v
                        else: ep_metrics['Misc'] += v
                    
                obs_list = next_obs
                global_critic_map = get_global_critic_obs(env)
                steps += 1
                
                # Print running stats every 50 steps
                if steps % 50 == 0:
                    print(f"Step {steps}: Deliv={ep_metrics['Deliv']:.1f}, Cmbt={ep_metrics['Combat']:.1f}")
                
                # Slow down for viewing pleasure
                time.sleep(0.05) 
                
                if term or trunc:
                    print(f"Episode finished. Total: {ep_reward:.1f} | Deliv: {ep_metrics['Deliv']:.1f} | Cmbt: {ep_metrics['Combat']:.1f}")
                    break
                    
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to the .pt checkpoint file')
    parser.add_argument('--scenario', type=str, default='unified', help='Scenario name')
    args = parser.parse_args()
    
    visualize(args.checkpoint, args.scenario)

