import torch
import numpy as np
import time
import argparse
import os
import pygame
import sys

# Append path to ensure we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.policies.simple_policies.map_policy import SimpleCNNMapPolicy, CNNPolicyWrapper
from Swarm2d.training.map_trainer import MapRLTrainer as FastRLTrainer

def visualize(scenario='resource', checkpoint_path=None, sensing_override=None, team2_checkpoint=None, render_mode='both'):
    # Determine default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = f'rl_training_results_cnn_{scenario}_fast_v4/checkpoint_latest.pt'

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize trainer to get environment and policy structure
    # We set max_steps high to watch for a while
    trainer = FastRLTrainer(num_episodes=1, max_steps=1000, scenario=scenario)
    
    # Hack: Close the headless env and re-open with render_mode='human'
    trainer.env.close()
    
    # Update config for rendering
    trainer.env_config['render_mode'] = render_mode

    # Apply sensing override if provided
    if sensing_override is not None:
        print(f"Overriding sensing_range_fraction to {sensing_override}")
        trainer.env_config['sensing_range_fraction'] = sensing_override
        
        # ALSO update randomization factors if present, otherwise they take precedence!
        if 'agent_randomization_factors' in trainer.env_config:
            rand_factors = trainer.env_config['agent_randomization_factors']
            if 'sensing_range_fraction' in rand_factors:
                print(f"  [Override] Updating randomization base for sensing_range_fraction from {rand_factors['sensing_range_fraction'].get('base')} to {sensing_override}")
                rand_factors['sensing_range_fraction']['base'] = sensing_override

    # Configure Team 2 (3rd Team) if requested
    team2_policy = None
    if team2_checkpoint:
        print(f"Configuring 3rd Team (Team 2) with checkpoint: {team2_checkpoint}")
        trainer.env_config['num_teams'] = 3
        
        # [FIX] Force raw_ch_count to 8 (Standard 2-Team/Relative Channel Count)
        # This prevents the environment from expanding channels to 29 when num_teams > 2.
        from Swarm2d.constants import ChannelConstants
        trainer.env_config['raw_ch_count'] = ChannelConstants.RAW_CH_COUNT
        print(f"  [Config] Forcing raw_ch_count={trainer.env_config['raw_ch_count']} for compatibility.")

        # Load Team 2 Policy
        # Use same architecture as Team 0
        team2_raw_policy = SimpleCNNMapPolicy(
            map_channels=trainer.map_channels, 
            map_size=trainer.map_size,
            memory_channels=trainer.memory_channels,
            memory_size=trainer.memory_size,
            self_dim=trainer.self_dim
        ).to(trainer.device)
        
        if os.path.exists(team2_checkpoint):
            ckpt = torch.load(team2_checkpoint, map_location=trainer.device, weights_only=False)
            team2_raw_policy.load_state_dict(ckpt['policy'])
            team2_raw_policy.eval()
            print(f"  [OK] Team 2 loaded (Episode {ckpt.get('episode', '?')})")
            team2_policy = CNNPolicyWrapper(team2_raw_policy)
        else:
            print(f"  [ERROR] Team 2 checkpoint not found: {team2_checkpoint}")
            print(f"  [WARNING] Team 2 will use Heuristic Policy as fallback.")
    
    # Re-initialize environment
    from Swarm2d.env.env import Swarm2DEnv
    trainer.env = Swarm2DEnv(**trainer.env_config)
    
    # Load weights for Team 0
    if not trainer.load_checkpoint(checkpoint_path):
        print(f"Failed to load checkpoint from {checkpoint_path}")
        return

    print("Starting Visualization...")
    trainer.rl_policy.eval() # Set to eval mode
    
    obs_list, _ = trainer.env.reset()
    
    total_reward = 0
    steps = 0
    
    try:
        while True:
            # start_time = time.time()
            actions = []
            
            # Team 0: RL Policy
            for i in range(trainer.team_0_size):
                obs = obs_list[i]
                # No grad needed for visualization
                with torch.no_grad():
                    # Use deterministic=True for visualization
                    move, pickup, _, _ = trainer.rl_policy.act(obs['map'], obs['memory_map'], obs['self'], deterministic=True)
                actions.append({'movement': move, 'pickup': pickup})
                
                # --- DEBUG PRINT FOR USER ---
                if i == 0 and steps % 20 == 0:
                    # Extract relative hive vector from observation (indices 6 and 7)
                    rel_hive_x = obs['self'][6]
                    rel_hive_y = obs['self'][7]
                    
                    # Calculate ACTUAL vector for comparison
                    agent_pos = trainer.env.agents[i]['pos']
                    team_id = trainer.env.agents[i]['team']
                    if team_id in trainer.env.hives:
                        hive_pos = trainer.env.hives[team_id]['pos']
                        d_max = np.sqrt(trainer.env.width**2 + trainer.env.height**2)
                        
                        true_vec = hive_pos - agent_pos
                        true_vec_norm_x = true_vec[0] / d_max
                        true_vec_norm_y = true_vec[1] / d_max
                        
                        # Interpret vectors for user
                        obs_dir = "RIGHT" if rel_hive_x > 0 else "LEFT"
                        true_dir = "RIGHT" if true_vec_norm_x > 0 else "LEFT"
                        act_dir = "RIGHT" if move[0] > 0.1 else ("LEFT" if move[0] < -0.1 else "STILL")
                        
                        pickup_action = pickup # int 0, 1, 2
                        pickup_str = "NONE"
                        if pickup_action == 1: pickup_str = "GRAPPLE/PICK"
                        elif pickup_action == 2: pickup_str = "ATTACK/BREAK"
                        
                        print(f"[VISUAL DEBUG] Agent 0 | Hive Vec OBS: ({rel_hive_x:.2f}, {rel_hive_y:.2f}) [{obs_dir}] | ACTUAL: ({true_vec_norm_x:.2f}, {true_vec_norm_y:.2f}) [{true_dir}] | Action: ({move[0]:.2f}, {move[1]:.2f}) [{act_dir}] | {pickup_str}")

            
            # Team 1: Heuristic (Usually Team 1 is heuristic_policy)
            start_idx_t1 = trainer.team_0_size
            end_idx_t1 = start_idx_t1 + trainer.team_1_size
            
            for i in range(start_idx_t1, end_idx_t1):
                obs = obs_list[i]
                actions.append(trainer.heuristic_policy.act(obs))
                
            # Team 2: Loaded Policy (If Active) OR Heuristic Fallback
            if trainer.env.num_teams >= 3:
                start_idx_t2 = end_idx_t1
                end_idx_t2 = trainer.env.num_agents # 15
                
                for i in range(start_idx_t2, end_idx_t2):
                    obs = obs_list[i]
                    if team2_policy:
                        actions.append(team2_policy.act(obs))
                    else:
                        # Fallback to heuristic if policy missing
                        actions.append(trainer.heuristic_policy.act(obs))
            
            # Step
            if len(actions) != len(trainer.env.agents):
                print(f"[ERROR] Action count mismatch! Expected {len(trainer.env.agents)}, got {len(actions)}")
                
            obs_list, rewards, terminated, truncated, infos = trainer.env.step(actions)
            trainer.env.render()
            
            step_rew = sum([sum(rewards[i].values()) for i in range(trainer.team_0_size)])
            total_reward += step_rew
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps} | Team Reward: {step_rew:.2f} | Total: {total_reward:.2f}")
            
            # Cap FPS
            # time.sleep(0.05) 
            
            if terminated or truncated:
                print("Episode finished.")
                obs_list, _ = trainer.env.reset()
                steps = 0
                total_reward = 0
                
    except KeyboardInterrupt:
        print("Visualization stopped.")
    finally:
        trainer.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='resource')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--sensing', type=float, default=None, help='Override sensing_range_fraction')
    parser.add_argument('--team2', type=str, default=None, help='Path to checkpoint for Team 2 (Optional 3rd Team)')
    parser.add_argument('--render_mode', type=str, default='both', help='Render mode: human (Pygame), gui (PyBullet), or both')
    args = parser.parse_args()
    
    visualize(args.scenario, args.checkpoint, args.sensing, args.team2, args.render_mode)
