import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# Import your training class to load the policy architecture
# Ensure 'simple_rl_training_MAP_ONLY.py' is in the same folder!
from simple_rl_training_MAP_ONLY import FastRLTrainer

# --- CONFIGURATION ---
# UPDATE THIS PATH to your actual Resource Run checkpoint location
CHECKPOINT_PATH = "rl_training_results_cnn_combat_fast_v5/checkpoint_latest.pt"
NUM_EVAL_EPISODES = 1
MAX_STEPS = 1000

def collect_comprehensive_metrics(checkpoint_path, n_episodes):
    print(f"--- LOADING RESOURCE MODEL: {checkpoint_path} ---")
    
    # 1. Init Environment (Force 'resource' scenario)
    trainer = FastRLTrainer(num_episodes=1, scenario='unified', max_steps=MAX_STEPS)
    
    # 2. Load Weights
    if not trainer.load_checkpoint(checkpoint_path):
        print(f"CRITICAL ERROR: Could not load {checkpoint_path}")
        return None

    trainer.rl_policy.eval() # Deterministic mode for evaluation
    
    # 3. METRIC CONTAINERS
    data = {
        'agent_positions': [],      # Heatmap
        'resource_paths': defaultdict(list), # Trajectories (id -> [(x,y), ...])
        'actions_dist': [],         # Histogram [0, 1, 2]
        'holding_ratio': [],        # % of time spent holding
        'distance_to_hive': [],     # Homing behavior
        'hive_loc': None
    }
    
    print(f"Running {n_episodes} evaluation episodes...")
    
    for ep in range(n_episodes):
        obs_list, _ = trainer.env.reset()
        
        # Store Hive Location for reference (Team 0)
        if ep == 0:
            # Handle case where hive 0 might not exist immediately (rare but safer)
            if 0 in trainer.env.hives:
                data['hive_loc'] = trainer.env.hives[0]['pos']
            else:
                data['hive_loc'] = np.array([50.0, 50.0]) # Fallback
            
        # Snapshot initial resource positions
        # We only track resources present at START of episode
        res_start_pos = {r['id']: r['pos'].copy() for r in trainer.env.resources if r.get('pos') is not None}
        
        ep_holding_steps = 0
        ep_total_steps = 0
        
        for step in range(MAX_STEPS):
            actions = []
            
            # --- TEAM 0 (RL) ---
            for i in range(trainer.team_0_size):
                # 1. Record Position
                agent_pos = trainer.env.agents[i]['pos']
                data['agent_positions'].append(agent_pos)
                
                # 2. Record Distance to Hive
                if data['hive_loc'] is not None:
                    dist = np.linalg.norm(agent_pos - data['hive_loc'])
                    data['distance_to_hive'].append(dist)
                
                # 3. Get Action
                obs = obs_list[i]
                move, pickup, _, _ = trainer.rl_policy.act(obs['map'], obs['memory_map'], obs['self'], deterministic=True)
                actions.append({'movement': move, 'pickup': pickup})
                
                # 4. Record Action Logic
                data['actions_dist'].append(pickup)
                
                # 5. Check Holding Status (Telemetry from Obs)
                # obs['self'][8] is is_carrying
                try:
                    # Handle both Tensor and Numpy inputs
                    val = obs['self'][8]
                    if hasattr(val, 'item'): val = val.item()
                    if val > 0.5: ep_holding_steps += 1
                except: pass
                
                ep_total_steps += 1

            # --- TEAM 1 (Heuristic) ---
            for i in range(trainer.team_0_size, trainer.env.num_agents):
                actions.append(trainer.heuristic_policy.act(obs_list[i]))
                
            # Step Env
            obs_list, _, term, trunc, _ = trainer.env.step(actions)
            
            # 6. Record Resource Movements (Physics Proof)
            # FIX: Only track resources that were present at start to avoid KeyError
            for r in trainer.env.resources:
                if r.get('pos') is not None:
                    rid = r['id']
                    # SAFEGUARD: Check if this resource existed at start
                    if rid in res_start_pos:
                        # Only record if it moved significantly from start
                        if np.linalg.norm(r['pos'] - res_start_pos[rid]) > 1.0:
                            data['resource_paths'][f"ep{ep}_r{rid}"].append(r['pos'].copy())

            if term or trunc: break
            
        # End Episode Stats
        ratio = ep_holding_steps / max(1, ep_total_steps)
        data['holding_ratio'].append(ratio)
        print(f"  Ep {ep+1}: Holding Ratio = {ratio*100:.1f}%")

    return data

def generate_thesis_plots(data):
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: SPATIAL HEATMAP (Navigation) ---
    print("Generating Heatmap...")
    plt.figure(figsize=(8, 8))
    x_pos = [p[0] for p in data['agent_positions']]
    y_pos = [p[1] for p in data['agent_positions']]
    
    plt.hist2d(x_pos, y_pos, bins=60, cmap='inferno', range=[[0, 400], [0, 400]], cmin=1)
    plt.colorbar(label='Agent Dwell Density')
    
    # Draw Hive
    if data['hive_loc'] is not None:
        hx, hy = data['hive_loc']
        plt.scatter(hx, hy, s=300, c='cyan', marker='H', edgecolors='white', linewidth=2, label='Home Hive')
    
    plt.title("Agent Spatial Distribution (Navigation Strategy)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_proof_spatial_heatmap.png', dpi=300)
    plt.close()
    
    # --- PLOT 2: RESOURCE TRAJECTORIES (Physics Manipulation) ---
    print("Generating Flow Map...")
    plt.figure(figsize=(8, 8))
    plt.xlim(0, 400); plt.ylim(0, 400)
    
    # Plot Hive
    if data['hive_loc'] is not None:
        hx, hy = data['hive_loc']
        plt.scatter(hx, hy, s=300, c='cyan', marker='H', edgecolors='black', label='Delivery Zone', zorder=10)
    
    # Plot Paths
    count = 0
    for rid, path in data['resource_paths'].items():
        # Only show significant movements (>10 steps of history)
        if len(path) > 10: 
            arr = np.array(path)
            # Plot line
            plt.plot(arr[:, 0], arr[:, 1], color='orange', alpha=0.3, linewidth=1)
            # Plot end point
            plt.scatter(arr[-1, 0], arr[-1, 1], s=10, c='green', marker='.') 
            count += 1
            
    plt.title(f"Resource Displacement Vectors (n={count} objects moved)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_proof_resource_flow.png', dpi=300)
    plt.close()
    
    # --- PLOT 3: ACTION DISTRIBUTION (Discrete Logic) ---
    print("Generating Action Histogram...")
    plt.figure(figsize=(8, 5))
    actions = np.array(data['actions_dist'])
    counts = [np.sum(actions == 0), np.sum(actions == 1), np.sum(actions == 2)]
    labels = ['No-Op (0)', 'Interact/Grip (1)', 'Release (2)']
    
    sns.barplot(x=labels, y=counts, palette=['gray', 'green', 'red'])
    plt.title("Discrete Interaction Policy Distribution (Deterministic Eval)")
    plt.ylabel("Frequency (Steps)")
    plt.tight_layout()
    plt.savefig('fig_proof_action_dist.png', dpi=300)
    plt.close()
    
    print("\nSUCCESS: Generated 3 Proof Figures:")
    print("1. fig_proof_spatial_heatmap.png (Where they go)")
    print("2. fig_proof_resource_flow.png (What they moved)")
    print("3. fig_proof_action_dist.png (How they interacted)")

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Please check path: {CHECKPOINT_PATH}")
    else:
        # Run collection
        metrics = collect_comprehensive_metrics(CHECKPOINT_PATH, NUM_EVAL_EPISODES)
        
        # Run plotting if data collected
        if metrics:
            generate_thesis_plots(metrics)