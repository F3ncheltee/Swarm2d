import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import glob
import numpy as np
import pandas as pd
import time
import argparse

def check_status(scenario='resource'):
    results_dir = f'rl_training_results_map_{scenario}'
    
    print("\n" + "="*70)
    print(f"MAP-BASED TRAINING STATUS CHECK - {scenario.upper()} SCENARIO")
    print("="*70)
    
    log_file = os.path.join(results_dir, 'training_log.csv')
    
    # 1. Check CSV Log (Most accurate/recent)
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            if not df.empty:
                last_row = df.iloc[-1]
                ep = int(last_row['episode'])
                
                print(f"✅ Training Active! (Based on CSV log)")
                print(f"   Episode: {ep}")
                print(f"   Reward:     {last_row['reward']:.1f} (Mean last 10: {df['reward'].tail(10).mean():.1f})")
                
                if 'resources' in df.columns:
                    print(f"   Resources:  {int(last_row['resources'])} (Mean last 10: {df['resources'].tail(10).mean():.1f})")
                
                if 'kills' in df.columns:
                     print(f"   Kills:      {int(last_row['kills'])} (Mean last 10: {df['kills'].tail(10).mean():.1f})")
                
                if 'damage' in df.columns:
                     print(f"   Damage:     {float(last_row['damage']):.1f} (Mean last 10: {df['damage'].tail(10).mean():.1f})")

                print(f"   Survival:   {last_row['survival']:.2f}")
                print(f"   Loss:       {last_row['loss']:.4f}")
                print(f"   Team 0 Rew: {last_row['team0_reward']:.1f} (Learning)")
                print(f"   Team 1 Rew: {last_row['team1_reward']:.1f} (Opponent)")
                
                # Check file age
                mod_time = os.path.getmtime(log_file)
                age = time.time() - mod_time
                if age < 60:
                    status = "RUNNING 🟢"
                elif age < 300:
                    status = "STALLED? 🟡"
                else:
                    status = "STOPPED 🔴"
                print(f"   Status: {status} (Last update {age:.0f}s ago)")
                
                # Convergence check
                if len(df) > 50:
                    recent_avg = df['reward'].tail(20).mean()
                    prev_avg = df['reward'].iloc[-40:-20].mean()
                    if abs(recent_avg - prev_avg) < 1.0:
                        print(f"\n   ℹ️  Possible Convergence? Reward stable around {recent_avg:.1f} over last 20 eps.")
                    elif recent_avg > prev_avg + 5:
                        print(f"\n   📈 Improving! Reward up {recent_avg - prev_avg:.1f} in last 20 eps.")
                    
            else:
                print("   CSV log exists but is empty.")
        except pd.errors.EmptyDataError:
             print("   CSV log exists but is empty (EmptyDataError).")
        except KeyError as e:
             print(f"   ⚠️  CSV Format Error: Column {e} not found. Deleting corrupt log...")
             try:
                 os.remove(log_file)
                 print("       Deleted. Please restart training script to recreate header.")
             except:
                 pass
        except Exception as e:
            print(f"   ⚠️  Could not read CSV log: {e}")
    else:
        print(f"   ❌ No CSV log found at {log_file} (maybe training hasn't started?)")

    # 2. Health Check / Interpretation
    print("\n   🏥 Health Check:")
    if 'df' in locals() and not df.empty:
        # Survival
        srv = df['survival'].tail(10).mean()
        if srv < 0.2:
            print("      ⚠️  CRITICAL: Agents are dying too fast (<20% survival).")
            print("          Check penalties for death or map boundaries.")
        
        # Scenario-specific checks
        if scenario == 'resource':
            if 'resources' in df.columns:
                res = df['resources'].tail(10).mean()
                if res < 1.0 and ep > 50:
                     print("      ⚠️  WARNING: No resources being collected after 50 eps.")
                     print("          Agents might be ignoring the objective.")
        
        elif scenario == 'combat':
            if 'damage' in df.columns:
                dmg = df['damage'].tail(10).mean()
                if dmg < 1.0 and ep > 25:
                    print("      ⚠️  WARNING: Very low combat damage. Agents might be avoiding fights.")
            
            if 'kills' in df.columns:
                kills = df['kills'].tail(10).mean()
                if kills < 0.1 and ep > 50:
                     print("      ⚠️  WARNING: No kills after 50 eps. Combat might be too difficult or rewards unbalanced.")
        
        # Loss stability
        loss = df['loss'].tail(10).mean()
        if abs(loss) > 1000 or pd.isna(loss):
             print("      ⚠️  CRITICAL: Loss explosion. Network might be diverging.")
        
        if srv >= 0.2 and abs(loss) <= 1000 and not pd.isna(loss):
             # Only check resources for resource scenario
             resources_ok = True
             if scenario == 'resource' and 'resources' in df.columns:
                 # Check if 'res' is defined from previous block, if not recalculate or use safe default
                 res_val = df['resources'].tail(10).mean()
                 if res_val < 1.0 and ep > 50:
                     resources_ok = False
             
             if resources_ok:
                print("      ✅ Training looks healthy!")

    # 3. Check Plots
    print("\n   Plots:")
    plots = glob.glob(os.path.join(results_dir, '*.png'))
    if plots:
        # Get the most recent file based on modification time
        latest_plot = max(plots, key=os.path.getmtime)
        age_min = (time.time() - os.path.getmtime(latest_plot)) / 60
        print(f"   📈 Latest plot: {os.path.basename(latest_plot)}")
        print(f"      (Created {age_min:.1f} min ago)")
    else:
        print("      No plots generated yet (wait for ep 25).")
            
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='resource', choices=['resource', 'combat'],
                       help='Which scenario to check: resource or combat')
    args = parser.parse_args()
    
    check_status(args.scenario)
