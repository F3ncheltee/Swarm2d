import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_training(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    # Read data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    if len(data) < 2:
        print("Not enough data to plot yet.")
        return

    # Calculate Rolling Average (Smooths the noise)
    window_size = min(50, len(data))
    if window_size > 0:
        data['Reward_Smooth'] = data['RawReward'].rolling(window=window_size).mean()
        data['Combat_Smooth'] = data['Combat'].rolling(window=window_size).mean()

    # Create Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Raw Data (faint)
    plt.plot(data['Episode'], data['RawReward'], alpha=0.2, color='gray', label='Raw Reward')
    
    # Plot Smoothed Data (Main Proof)
    if window_size > 0:
        plt.plot(data['Episode'], data['Reward_Smooth'], color='blue', linewidth=2, label=f'Total Reward ({window_size}-Ep Avg)')
        plt.plot(data['Episode'], data['Combat_Smooth'], color='red', linestyle='--', linewidth=1.5, label=f'Combat Reward ({window_size}-Ep Avg)')

    plt.title('Agent Learning Progress (CTDE)', fontsize=16)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "training_log_unified.csv"
    plot_training(file_path)




