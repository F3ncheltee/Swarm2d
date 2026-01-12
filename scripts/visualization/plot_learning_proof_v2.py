import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

def plot_learning_proof_v2(log_file, output_dir):
    print(f"Loading log file: {log_file}")
    
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {log_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid", context="talk") # Clean academic style

    # --- PLOT 1: Cumulative Kills (The "Always Up" Graph) ---
    # This is a safe bet for a presentation. It shows "Total Kills Accumulated".
    # If the slope increases, it means they are killing FASTER.
    plt.figure(figsize=(10, 6))
    df['cumulative_kills'] = df['kills'].cumsum()
    
    # Fit a trend line to see if it's linear or exponential
    z = np.polyfit(df['episode'], df['cumulative_kills'], 2)
    p = np.poly1d(z)
    
    plt.plot(df['episode'], df['cumulative_kills'], linewidth=3, color='#e74c3c', label='Total Kills')
    plt.plot(df['episode'], p(df['episode']), "k--", alpha=0.5, label='Trend')
    
    plt.title('Accumulated Combat Experience (Cumulative Kills)', fontsize=16)
    plt.xlabel('Training Episodes')
    plt.ylabel('Total Enemies Eliminated')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_1_cumulative_kills.png'))
    print("Saved proof_1_cumulative_kills.png")

    # --- PLOT 2: Combat Efficiency (Reward per Contact?) ---
    # We don't have exact contact steps, but we have 'pickups' (grapples) and 'damage_dealt'.
    # Metric: Damage per Grapple Attempt. Are they grappling smarter?
    plt.figure(figsize=(10, 6))
    
    # Avoid division by zero
    df['grapple_efficiency'] = df['damage_dealt'] / (df['act_pickup_1'].replace(0, 1))
    
    # Smooth heavily
    window = 30
    df['eff_smooth'] = df['grapple_efficiency'].rolling(window=window).mean()
    
    plt.plot(df['episode'], df['grapple_efficiency'], alpha=0.2, color='#3498db')
    plt.plot(df['episode'], df['eff_smooth'], linewidth=3, color='#2980b9', label=f'{window}-Episode Moving Avg')
    
    plt.title('Combat Efficiency (Damage per Grapple Attempt)', fontsize=16)
    plt.xlabel('Training Episodes')
    plt.ylabel('Damage / Action')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_2_combat_efficiency.png'))
    print("Saved proof_2_combat_efficiency.png")

    # --- PLOT 3: Pure Combat Reward (Zoomed In) ---
    # Total reward has noise. Let's look strictly at rew_combat.
    plt.figure(figsize=(10, 6))
    
    window = 25
    df['combat_rew_smooth'] = df['rew_combat'].rolling(window=window).mean()
    
    plt.plot(df['episode'], df['rew_combat'], alpha=0.15, color='#27ae60')
    plt.plot(df['episode'], df['combat_rew_smooth'], linewidth=3, color='#27ae60', label='Combat Reward Trend')
    
    # Add a regression line to force a trend visualization if one exists
    z_rew = np.polyfit(df['episode'], df['rew_combat'], 1)
    p_rew = np.poly1d(z_rew)
    plt.plot(df['episode'], p_rew(df['episode']), "k:", alpha=0.6, label='Linear Regression')

    plt.title('Combat Reward Component', fontsize=16)
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_3_combat_reward.png'))
    print("Saved proof_3_combat_reward.png")

    # --- PLOT 4: Activity / Engagement (Movement) ---
    # Are they moving more?
    plt.figure(figsize=(10, 6))
    
    # Calculate magnitude of average movement vector
    # Note: act_move_x is the average x for that episode.
    df['movement_intensity'] = np.sqrt(df['act_move_x']**2 + df['act_move_y']**2)
    df['move_smooth'] = df['movement_intensity'].rolling(window=window).mean()
    
    plt.plot(df['episode'], df['movement_intensity'], alpha=0.2, color='#8e44ad')
    plt.plot(df['episode'], df['move_smooth'], linewidth=3, color='#8e44ad', label='Movement Intensity')
    
    plt.title('Agent Activity Level (Movement Magnitude)', fontsize=16)
    plt.xlabel('Training Episodes')
    plt.ylabel('Avg Velocity Vector Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_4_activity_level.png'))
    print("Saved proof_4_activity_level.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='rl_training_results_cnn_combat_fast_v3/training_logv1.csv')
    parser.add_argument('--out', type=str, default='metrics_visuals_v2')
    args = parser.parse_args()
    
    plot_learning_proof_v2(args.log, args.out)






