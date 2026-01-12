import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metrics(log_file, output_dir):
    print(f"Loading log: {log_file}")
    if not os.path.exists(log_file):
        print("Log file not found!")
        return

    df = pd.read_csv(log_file)
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid", context="talk")
    
    window = 25
    
    # 1. Kill / Death Ratio (Dominance)
    plt.figure(figsize=(10, 6))
    # Add epsilon to deaths to avoid division by zero
    df['kd_ratio'] = df['kills'] / (df['deaths'].replace(0, 1))
    df['kd_smooth'] = df['kd_ratio'].rolling(window=window).mean()
    
    sns.lineplot(data=df, x='episode', y='kd_ratio', alpha=0.2, color='#e74c3c')
    sns.lineplot(data=df, x='episode', y='kd_smooth', color='#c0392b', linewidth=3, label='K/D Ratio')
    
    plt.axhline(1.0, color='gray', linestyle='--', label='Break-even (1.0)')
    plt.title("Combat Dominance (Kill/Death Ratio)", fontsize=16)
    plt.ylabel("K/D Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_6_kd_ratio.png'))
    print("Saved proof_6_kd_ratio.png")
    
    # 2. Survival Rate
    plt.figure(figsize=(10, 6))
    # Assuming 5 agents per team. Deaths = Team 0 deaths.
    df['survivors'] = 5 - df['deaths']
    df['surv_smooth'] = df['survivors'].rolling(window=window).mean()
    
    sns.lineplot(data=df, x='episode', y='survivors', alpha=0.2, color='#2ecc71')
    sns.lineplot(data=df, x='episode', y='surv_smooth', color='#27ae60', linewidth=3, label='Avg Survivors')
    
    plt.title("Team Survival Rate", fontsize=16)
    plt.ylabel("Surviving Agents (Max 5)")
    plt.ylim(0, 5.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_7_survival.png'))
    print("Saved proof_7_survival.png")

    # 3. Aggression (Pickups/Grapples Attempted)
    plt.figure(figsize=(10, 6))
    # act_pickup_1 is Grapple/Pickup
    df['aggression'] = df['act_pickup_1']
    df['agg_smooth'] = df['aggression'].rolling(window=window).mean()
    
    sns.lineplot(data=df, x='episode', y='aggression', alpha=0.2, color='#8e44ad')
    sns.lineplot(data=df, x='episode', y='agg_smooth', color='#9b59b6', linewidth=3, label='Grapple Attempts')
    
    plt.title("Aggression Level (Grapple Frequency)", fontsize=16)
    plt.ylabel("Actions per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proof_8_aggression.png'))
    print("Saved proof_8_aggression.png")

if __name__ == "__main__":
    plot_metrics('rl_training_results_cnn_combat_fast_v2/training_log.csv', 'metrics_visuals_v2')






