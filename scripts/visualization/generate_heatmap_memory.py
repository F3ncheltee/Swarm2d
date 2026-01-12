import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse
from simple_rl_training_MEMORY_ONLY import FastRLTrainer

def generate_heatmap(scenario='exploration', episodes=5, checkpoint_path=None):
    print(f"--- Generating MEMORY-BASED Heatmap for Scenario: {scenario} ---")
    
    # 1. Initialize Trainer (handles Env and Policy setup)
    # We use the trainer class to easily load the correct config and policy structure
    trainer = FastRLTrainer(num_episodes=1, max_steps=500, scenario=scenario)
    
    # 2. Load Checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(trainer.output_dir, 'checkpoint_latest.pt')
        
    if not trainer.load_checkpoint(checkpoint_path):
        print(f"[ERROR] Could not load checkpoint: {checkpoint_path}")
        return

    print("Policy loaded. Collecting trajectories...")
    trainer.rl_policy.eval()
    
    # 3. Data Collection
    # Grid for heatmap
    width = trainer.env_config['width']
    height = trainer.env_config['height']
    
    # We'll use a bin size of 5 units (matching cell size usually)
    bin_size = 5
    x_bins = int(width / bin_size)
    y_bins = int(height / bin_size)
    
    heatmap = np.zeros((y_bins, x_bins))
    
    all_x = []
    all_y = []
    
    for ep in range(episodes):
        obs_list, _ = trainer.env.reset()
        print(f"  Collecting Episode {ep+1}/{episodes}...")
        
        for step in range(trainer.max_steps):
            actions = []
            
            # Team 0 (RL)
            for i in range(trainer.team_0_size):
                obs = obs_list[i]
                agent = trainer.env.agents[i]
                
                # Record Position if Alive
                if agent['alive']:
                    px, py = agent['pos']
                    all_x.append(px)
                    all_y.append(py)
                    
                    # Add to heatmap grid
                    bx = min(int(px / bin_size), x_bins - 1)
                    by = min(int(py / bin_size), y_bins - 1)
                    heatmap[by, bx] += 1
                
                # Act
                with torch.no_grad():
                    # Memory Policy expectes: obs_map, obs_memory, obs_self
                    # But the trainer.rl_policy.act matches the signature
                    move, pickup, _, _ = trainer.rl_policy.act(obs['map'], obs['memory_map'], obs['self'])
                actions.append({'movement': move, 'pickup': pickup})

            # Team 1 (Heuristic) - Just to keep sim running
            for i in range(trainer.team_0_size, trainer.env.num_agents):
                 actions.append(trainer.heuristic_policy.act(obs_list[i]))
            
            obs_list, _, terminated, truncated, _ = trainer.env.step(actions)
            
            if terminated or truncated:
                break
                
    # 4. Plotting
    print("Plotting heatmap...")
    plt.figure(figsize=(10, 10))
    
    # Use imshow for the heatmap
    # Origin is usually top-left in image, but simulation might be bottom-left. 
    # PyGame uses top-left (0,0). Matplotlib imshow uses top-left by default.
    plt.imshow(heatmap, cmap='hot', interpolation='gaussian', origin='upper', extent=[0, width, height, 0])
    plt.colorbar(label='Agent Presence (Steps)')
    plt.title(f"Memory-Based Agent Exploration Heatmap ({episodes} Episodes)\nScenario: {scenario}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Overlay Hive Position (Team 0)
    if 0 in trainer.env.hives:
        hx, hy = trainer.env.hives[0]['pos']
        plt.plot(hx, hy, 'bo', markersize=10, label='Team 0 Hive', markeredgecolor='white')
        plt.legend()

    # Save
    output_file = f"heatmap_{scenario}_memory.png"
    plt.savefig(output_file, dpi=150)
    print(f"Heatmap saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='exploration')
    parser.add_argument('--episodes', type=int, default=5, help="Number of episodes to aggregate")
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    generate_heatmap(args.scenario, args.episodes, args.ckpt)


