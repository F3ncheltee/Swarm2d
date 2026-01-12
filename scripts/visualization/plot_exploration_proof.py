import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse

def plot_exploration_results(log_file, output_dir):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    # Create output dir for images
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(log_file)
    
    # Set style
    sns.set_theme(style="darkgrid")
    
    # --- 1. Total Reward Trend (Smoothed) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='reward', alpha=0.3, label='Raw Reward')
    df['reward_smooth'] = df['reward'].rolling(window=5, min_periods=1).mean()
    sns.lineplot(data=df, x='episode', y='reward_smooth', linewidth=2.5, label='Smoothed (MA-5)')
    plt.title("Learning Curve: Total Episode Reward", fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "exp_proof_1_total_reward.png"), dpi=150)
    plt.close()
    
    # --- 2. Exploration Reward Only (The "Proof") ---
    # rew_explore includes r_resource_found, r_exploration_intrinsic, etc.
    plt.figure(figsize=(10, 6))
    if 'rew_explore' in df.columns:
        sns.lineplot(data=df, x='episode', y='rew_explore', color='green', linewidth=2.5)
        plt.title("Exploration Efficacy (Intrinsic + Discovery)", fontsize=16)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Exploration Reward Points", fontsize=12)
        plt.savefig(os.path.join(output_dir, "exp_proof_2_exploration_reward.png"), dpi=150)
    else:
        print("Warning: 'rew_explore' not found in log.")
    plt.close()

    # --- 3. Discovery Breakdown (Resource vs Obstacle) ---
    # We might have columns for specific rewards if logged in detail, 
    # but the CSV header usually has 'rew_explore'.
    # However, 'res_moved' or 'pickups' might be proxies for interaction.
    # Let's check if we have detailed discovery metrics.
    # The header in simple_rl_training_MEMORY_ONLY.py was:
    # episode,reward,resources,loss,time,pickups,deliveries,rew_delivery,rew_explore,rew_combat,rew_pickup,rew_prog_pos,rew_prog_neg,act_move_x,act_move_y,act_pickup_0,act_pickup_1,act_pickup_2,res_disp,res_moved,kills,deaths,damage_dealt,grapples_won
    
    # We don't have explicit 'obstacles_found' count in the CSV, but 'rew_explore' aggregates it.
    # We can plot 'res_disp' (displacement) as a proxy for "interacting with the world".
    plt.figure(figsize=(10, 6))
    if 'res_disp' in df.columns:
        df['res_disp_smooth'] = df['res_disp'].rolling(window=5, min_periods=1).mean()
        sns.lineplot(data=df, x='episode', y='res_disp_smooth', color='orange', linewidth=2.5)
        plt.title("World Interaction: Object Displacement", fontsize=16)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Displacement (Units)", fontsize=12)
        plt.savefig(os.path.join(output_dir, "exp_proof_3_interaction.png"), dpi=150)
    plt.close()

    # --- 4. Movement/Activity Level ---
    # Calculate magnitude of average move vector
    if 'act_move_x' in df.columns and 'act_move_y' in df.columns:
        df['move_mag'] = (df['act_move_x']**2 + df['act_move_y']**2)**0.5
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='episode', y='move_mag', color='purple', linewidth=2.5)
        plt.title("Agent Activity Level (Avg Movement Magnitude)", fontsize=16)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Avg Speed/Step", fontsize=12)
        plt.ylim(0, 1.0) # Normalized output
        plt.savefig(os.path.join(output_dir, "exp_proof_4_activity.png"), dpi=150)
        plt.close()
        
    print(f"Generated 4 proof graphs in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--out', type=str, default='exploration_proofs')
    args = parser.parse_args()
    
    plot_exploration_results(args.log, args.out)


