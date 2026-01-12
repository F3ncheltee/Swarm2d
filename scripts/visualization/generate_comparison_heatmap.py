import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Append path to ensure we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.simple_policies.map_policy import SimpleCNNMapPolicy

def get_agent_positions(checkpoint_path, env, device, num_episodes=5):
    print(f"--- Processing {checkpoint_path} ---")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return [], []

    # Initialize Policy
    # We need to peek at env dims to init policy
    obs, _ = env.reset()
    map_shape = obs[0]['map'].shape
    self_dim = obs[0]['self'].shape[0]
    
    policy = SimpleCNNMapPolicy(
        map_channels=map_shape[0], 
        map_size=map_shape[1], 
        memory_channels=0, # Dummy
        memory_size=0, 
        self_dim=self_dim
    ).to(device)
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    all_x = []
    all_y = []
    
    for ep in range(num_episodes):
        obs_list, _ = env.reset()
        for step in range(300): # Limit steps
            actions = []
            
            # Team 0 (Controlled)
            for i in range(5):
                # Record Pos
                pos = env.agents[i]['pos']
                all_x.append(pos[0])
                all_y.append(pos[1])
                
                obs = obs_list[i]
                # Prepare tensors
                obs_map = torch.tensor(obs['map'], dtype=torch.float32, device=device).unsqueeze(0)
                obs_self = torch.tensor(obs['self'], dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    # Forward
                    move_mean, _, pickup_logits = policy(obs_map, obs_self)
                    move_action = move_mean.cpu().numpy()[0]
                    pickup_action = torch.argmax(pickup_logits).item()
                    
                actions.append({'movement': move_action, 'pickup': pickup_action})
                
            # Team 1 (Dummy - Stationary or Random)
            for i in range(5, 10):
                actions.append({'movement': np.zeros(2), 'pickup': 0})
                
            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break
                
    return all_x, all_y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'metrics_visuals_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Config matching training
    env_config = {
        'num_teams': 2,
        'num_agents_per_team': 5,
        'width': 400,
        'height': 400,
        'max_steps': 300,
        'render_mode': 'headless',
        'generate_memory_graph': False,
        'generate_memory_map': False,
        'num_obstacles': 10,
        'num_resources': 20
    }
    
    env = Swarm2DEnv(**env_config)
    
    # 1. Early
    x1, y1 = get_agent_positions('rl_training_results_cnn_combat_fast_v3/checkpoint_ep25.pt', env, device)
    
    # 2. Late
    x2, y2 = get_agent_positions('rl_training_results_cnn_combat_fast_v4/checkpoint_latest.pt', env, device)
    
    # Plot
    print("Generating Heatmaps...")
    sns.set(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Early
    if len(x1) > 0:
        sns.kdeplot(x=x1, y=y1, ax=axes[0], fill=True, cmap="Blues", levels=20, thresh=0.05)
        axes[0].set_title("Early Training (Ep 25): Wandering/Random", fontsize=14)
        axes[0].set_xlim(0, 400)
        axes[0].set_ylim(0, 400)
        axes[0].invert_yaxis()
        axes[0].set_aspect('equal')
        
    # Late
    if len(x2) > 0:
        sns.kdeplot(x=x2, y=y2, ax=axes[1], fill=True, cmap="Reds", levels=20, thresh=0.05)
        axes[1].set_title("Late Training (Ep 250): Combat Focus", fontsize=14)
        axes[1].set_xlim(0, 400)
        axes[1].set_ylim(0, 400)
        axes[1].invert_yaxis()
        axes[1].set_aspect('equal')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'proof_5_spatial_strategy.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()






