import cv2
import numpy as np
import pygame
import os
import torch
import argparse
import sys

# Append path to ensure we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.simple_policies.memory_policy import SimpleCNNMemoryPolicy

def record_episode(checkpoint_path, output_filename, steps=500, scenario='combat', fps=30.0):
    print(f"--- Recording Video (MEMORY POLICY): {output_filename} ---")
    print(f"--- Playback Speed: {fps} FPS ---")
    
    # 1. Setup Environment
    print(f"Initializing Environment for scenario: {scenario}...")
    
    # Default Config
    env_config = {
        'num_teams': 2,
        'num_agents_per_team': 5,
        'num_resources': 10,
        'num_obstacles': 5,
        'width': 500,
        'height': 500,
        'max_steps': steps,
        'render_mode': 'human',
        'generate_memory_graph': False,
        'generate_memory_map': True, # ENABLE MEMORY!
    }

    # Scenario Overrides (Match simple_rl_training_MEMORY_ONLY.py)
    if scenario == 'exploration':
        env_config.update({
            'num_resources': 50,
            'num_obstacles': 15,
            'width': 600,
            'height': 600,
            'hive_min_distance': 100.0,
            'sensing_range_fraction': 0.09,
        })
    elif scenario == 'resource':
        env_config.update({
            'num_resources': 69, # Match training hack
            'num_obstacles': 5,
            'width': 400,
            'height': 400,
             'hive_min_distance': 120.0,
             'resource_hive_buffer': 20.0,
        })
    elif scenario == 'combat':
        env_config.update({
            'num_resources': 20,
            'num_obstacles': 10,
            'width': 400,
            'height': 400,
            'hive_min_distance': 80.0,
            'agent_spawn_radius': 60.0,
        })
    
    # Init Env
    env = Swarm2DEnv(**env_config)
    obs_list, _ = env.reset()
    
    # 2. Load Policy
    print(f"Loading Policy from {checkpoint_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Infer dimensions from observation
    sample_obs = obs_list[0]
    map_ch, map_sz, _ = sample_obs['map'].shape
    mem_ch, mem_sz, _ = sample_obs['memory_map'].shape
    self_dim = sample_obs['self'].shape[0]
    
    policy = SimpleCNNMemoryPolicy(map_ch, map_sz, mem_ch, mem_sz, self_dim).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint['policy'])
        print("Policy Loaded Successfully.")
    else:
        print(f"ERROR: Checkpoint {checkpoint_path} not found!")
        return

    # 3. Setup Video Writer
    # Capture one frame to get size
    env.render()
    surface = pygame.display.get_surface()
    width, height = surface.get_size()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 codec
    out = cv2.VideoWriter(output_filename, fourcc, float(fps), (width, height))
    
    print("Recording started...")
    
    # 4. Simulation Loop
    from Swarm2d.policies.heuristicPolicy.map_heuristic import MapHeuristic
    heuristic = MapHeuristic(env.action_space) # Opponent
    
    try:
        for step in range(steps):
            actions = []
            
            # Team 0 (RL)
            for i in range(env_config['num_agents_per_team']):
                obs = obs_list[i]
                # Check ALIVE status from ENV, not observation dict
                if env.agents[i]['alive']:
                    move, pickup, _, _ = policy.act(obs['map'], obs['memory_map'], obs['self'])
                    actions.append({'movement': move, 'pickup': pickup})
                else:
                    actions.append({'movement': np.zeros(2), 'pickup': 0})
            
            # Team 1 (Heuristic)
            for i in range(env_config['num_agents_per_team'], env.num_agents):
                actions.append(heuristic.act(obs_list[i]))
                
            obs_list, rewards, terminated, truncated, _ = env.step(actions)
            
            # Render & Capture
            env.render()
            
            # Convert Pygame surface to OpenCV image
            view = pygame.surfarray.array3d(surface)
            view = view.transpose([1, 0, 2]) # Pygame (W,H,C) -> NumPy (H,W,C)
            img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR) # RGB -> BGR for OpenCV
            
            out.write(img_bgr)
            
            if (step+1) % 50 == 0:
                print(f"Recorded step {step+1}/{steps}")
                
            if terminated or truncated:
                break
                
    except KeyboardInterrupt:
        print("Recording interrupted.")
    finally:
        out.release()
        env.close()
        print(f"Video saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint .pt file')
    parser.add_argument('--out', type=str, default='combat_demo.mp4', help='Output video filename')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps to record')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback FPS (30=SlowMo, 60=RealTime)')
    parser.add_argument('--scenario', type=str, default='combat', help='Scenario to record: combat, resource, exploration')
    args = parser.parse_args()
    
    record_episode(args.ckpt, args.out, args.steps, scenario=args.scenario, fps=args.fps)


