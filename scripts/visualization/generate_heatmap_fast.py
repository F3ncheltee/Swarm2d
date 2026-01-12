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

def get_positions_fast(checkpoint_path, env, device):
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return [], []
    
    print(f"Processing {checkpoint_path}...")
    
    # Init Policy
    obs, _ = env.reset()
    map_shape = obs[0]['map'].shape
    self_dim = obs[0]['self'].shape[0]
    
    policy = SimpleCNNMapPolicy(
        map_channels=map_shape[0],
        map_size=map_shape[1],
        memory_channels=0,
        memory_size=0,
        self_dim=self_dim
    ).to(device)
    
    # Load
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    
    all_x = []
    all_y = []
    
    # Run ONLY 1 EPISODE
    obs_list, _ = env.reset()
    for _ in range(300):
        actions = []
        # Team 0
        for i in range(5):
            pos = env.agents[i]['pos']
            all_x.append(pos[0])
            all_y.append(pos[1])
            
            obs_map = torch.tensor(obs_list[i]['map'], dtype=torch.float32, device=device).unsqueeze(0)
            obs_self = torch.tensor(obs_list[i]['self'], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                move_mean, _, pickup_logits = policy(obs_map, obs_self)
                move = move_mean.cpu().numpy()[0]
                pick = torch.argmax(pickup_logits).item()
            actions.append({'movement': move, 'pickup': pick})
            
        # Team 1
        for i in range(5, 10):
            actions.append({'movement': np.zeros(2), 'pickup': 0})
            
        obs_list, _, done, trunc, _ = env.step(actions)
        if done or trunc: break
            
    return all_x, all_y

def main():
    device = torch.device('cpu') # Force CPU for simplicity
    os.makedirs('metrics_visuals_v2', exist_ok=True)
    
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
    
    x1, y1 = get_positions_fast('rl_training_results_cnn_combat_fast_v3/checkpoint_ep25.pt', env, device)
    x2, y2 = get_positions_fast('rl_training_results_cnn_combat_fast_v4/checkpoint_latest.pt', env, device)
    
    print("Plotting...")
    sns.set(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Use HISTPLOT (Fast)
    if x1:
        sns.histplot(x=x1, y=y1, ax=axes[0], bins=40, cmap="Blues", cbar=False, stat='density')
        axes[0].set_title("Early Training (Ep 25)")
        axes[0].set_xlim(0, 400); axes[0].set_ylim(0, 400); axes[0].invert_yaxis()
        axes[0].set_aspect('equal')

    if x2:
        sns.histplot(x=x2, y=y2, ax=axes[1], bins=40, cmap="Reds", cbar=False, stat='density')
        axes[1].set_title("Late Training (Ep 250)")
        axes[1].set_xlim(0, 400); axes[1].set_ylim(0, 400); axes[1].invert_yaxis()
        axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('metrics_visuals_v2/proof_5_spatial_heatmap_fast.png')
    print("Saved proof_5_spatial_heatmap_fast.png")

if __name__ == "__main__":
    main()






