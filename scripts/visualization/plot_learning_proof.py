import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_combat_learning(log_file, output_dir):
    print(f"Loading log file: {log_file}")
    
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {log_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="darkgrid")
    
    # Rolling window for smoothing
    window_size = 20
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Combat Scenario: Proof of Learning (250 Episodes)', fontsize=20)
    
    # 1. Total Reward
    sns.lineplot(ax=axes[0, 0], data=df, x='episode', y='reward', alpha=0.3, color='blue', label='Raw')
    df['reward_smooth'] = df['reward'].rolling(window=window_size).mean()
    sns.lineplot(ax=axes[0, 0], data=df, x='episode', y='reward_smooth', color='blue', linewidth=2, label=f'{window_size}-Ep Avg')
    axes[0, 0].set_title('Total Episode Reward', fontsize=14)
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()

    # 2. Kills
    if 'kills' in df.columns:
        sns.lineplot(ax=axes[0, 1], data=df, x='episode', y='kills', alpha=0.3, color='red', label='Raw')
        df['kills_smooth'] = df['kills'].rolling(window=window_size).mean()
        sns.lineplot(ax=axes[0, 1], data=df, x='episode', y='kills_smooth', color='red', linewidth=2, label=f'{window_size}-Ep Avg')
        axes[0, 1].set_title('Kills per Episode', fontsize=14)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
    
    # 3. Damage Dealt
    if 'damage_dealt' in df.columns:
        sns.lineplot(ax=axes[1, 0], data=df, x='episode', y='damage_dealt', alpha=0.3, color='orange', label='Raw')
        df['damage_smooth'] = df['damage_dealt'].rolling(window=window_size).mean()
        sns.lineplot(ax=axes[1, 0], data=df, x='episode', y='damage_smooth', color='orange', linewidth=2, label=f'{window_size}-Ep Avg')
        axes[1, 0].set_title('Total Damage Dealt', fontsize=14)
        axes[1, 0].set_ylabel('Damage')
        axes[1, 0].legend()

    # 4. Grapples Won / Action Distribution
    # Let's plot Grapples Won vs Pickups Attempted (Grapple attempts) to see efficiency?
    # Or just Grapples Won
    if 'grapples_won' in df.columns:
        sns.lineplot(ax=axes[1, 1], data=df, x='episode', y='grapples_won', alpha=0.3, color='green', label='Raw')
        df['grapples_smooth'] = df['grapples_won'].rolling(window=window_size).mean()
        sns.lineplot(ax=axes[1, 1], data=df, x='episode', y='grapples_smooth', color='green', linewidth=2, label=f'{window_size}-Ep Avg')
        axes[1, 1].set_title('Grapples Broken/Won', fontsize=14)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'combat_learning_proof.png')
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

    # --- SECOND PLOT: Action Distribution Over Time ---
    plt.figure(figsize=(12, 6))
    if 'act_pickup_0' in df.columns: # 0=None, 1=Pickup, 2=Drop/Attack
        # Normalize counts to percentages
        total_actions = df['act_pickup_0'] + df['act_pickup_1'] + df['act_pickup_2']
        
        # We really care about the ratio of "Aggressive" actions (Pickup/Grapple) vs "Passive"
        # In combat, 1 is Grapple/Pickup.
        
        plt.stackplot(df['episode'], 
                      df['act_pickup_0']/total_actions, 
                      df['act_pickup_1']/total_actions, 
                      df['act_pickup_2']/total_actions,
                      labels=['None', 'Grapple/Pickup', 'Drop/Break'],
                      colors=['lightgray', 'red', 'orange'], alpha=0.8)
        
        plt.title('Action Distribution Over Training (Policy Behavior)', fontsize=16)
        plt.xlabel('Episode')
        plt.ylabel('Proportion of Actions')
        plt.legend(loc='upper left')
        plt.xlim(0, df['episode'].max())
        plt.ylim(0, 1)
        
        output_path_actions = os.path.join(output_dir, 'combat_action_distribution.png')
        plt.savefig(output_path_actions)
        print(f"Saved action distribution to {output_path_actions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='rl_training_results_cnn_combat_fast_v3/training_log.csv')
    parser.add_argument('--out', type=str, default='.')
    args = parser.parse_args()
    
    plot_combat_learning(args.log, args.out)






