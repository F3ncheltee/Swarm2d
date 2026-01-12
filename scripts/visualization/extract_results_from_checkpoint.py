import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.env.env import Swarm2DEnv
# Import your policy from your training script
from Swarm2d.policies.simple_policies.map_policy import SimpleCNNMapPolicy 

# ==========================================================
# CONFIGURATION: Choose your checkpoint here!
# ==========================================================
CHECKPOINT_TO_RUN = "rl_training_results_cnn_resource_fast_v4/checkpoint_latest.pt"
SCENARIO_NAME = "Resource_Logistics" # Used for folder naming
# ==========================================================

def run_evaluation_benchmark(checkpoint_path, num_episodes=5, max_steps=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = f"thesis_benchmarks_{SCENARIO_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. TEST SCENARIOS
    scenarios = {
        "Baseline": {"num_resources": 60, "num_obstacles": 5},
        "Stress_Test_Obstacles": {"num_resources": 60, "num_obstacles": 15},
        "Scarcity_Test": {"num_resources": 20, "num_obstacles": 5}
    }

    print(f"Loading weights from {checkpoint_path}...")
    
    # FIX: weights_only=False bypasses the UnpicklingError
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize Policy (Ensure dims match your training config)
    policy = SimpleCNNMapPolicy(map_channels=8, map_size=32, memory_channels=0, 
                                memory_size=0, self_dim=27).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()

    all_data = []

    for name, config in scenarios.items():
        print(f"\n>>> Evaluating: {name}")
        
        env_config = {
            'num_teams': 2,
            'num_agents_per_team': 5,
            'max_steps': max_steps,
            'width': 400, 'height': 400,
            'render_mode': "headless",
            **config
        }
        env = Swarm2DEnv(**env_config)

        for ep in range(num_episodes):
            obs_list, _ = env.reset()
            ep_deliveries = 0
            ep_reward = 0
            ep_dist = 0
            
            init_pos = {r['id']: r['pos'].copy() for r in env.resources}

            for step in range(max_steps):
                actions = []
                for i in range(5):
                    o = obs_list[i]
                    with torch.no_grad():
                        # deterministic=True is crucial for a "Proof"
                        mv, pk, _, _ = policy.act(o['map'], None, o['self'], deterministic=True)
                    actions.append({'movement': mv, 'pickup': pk})
                
                # Opponents are stationary
                for i in range(5, 10):
                    actions.append({'movement': np.zeros(2), 'pickup': 0})

                obs_list, rewards, term, trunc, _ = env.step(actions)
                
                for i in range(5):
                    ep_reward += sum(rewards[i].values())
                    if rewards[i].get('r_delivery', 0) > 0:
                        ep_deliveries += 1
                
                if term or trunc: break
            
            for r in env.resources:
                ep_dist += np.linalg.norm(r['pos'] - init_pos[r['id']])

            all_data.append({
                "Scenario": name,
                "Episode": ep,
                "Deliveries": ep_deliveries,
                "Efficiency": ep_deliveries / (ep_dist + 1e-6),
                "Displacement": ep_dist
            })

    # Save Results
    df = pd.DataFrame(all_data)
    df.to_csv(f"{out_dir}/raw_metrics.csv", index=False)
    
    summary = df.groupby("Scenario").mean().drop(columns="Episode")
    summary.to_csv(f"{out_dir}/thesis_summary_table.csv")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Scenario", y="Deliveries", data=df, palette="magma")
    plt.title(f"Generalization Proof: {SCENARIO_NAME}")
    plt.savefig(f"{out_dir}/benchmark_plot.png", dpi=300)
    
    print(f"\n✅ SUCCESS! Files saved to /{out_dir}/")
    print(summary)

if __name__ == "__main__":
    run_evaluation_benchmark(CHECKPOINT_TO_RUN)