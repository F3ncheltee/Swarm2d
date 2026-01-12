import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import json
import traceback
from collections import defaultdict
import math
import argparse
from itertools import product

# --- Python Path (ensure Swarm2DEnv is found) ---
# Calculate the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import (AGENT_RADIUS, FPS, AGENT_BASE_STRENGTH,
                           HIVE_DELIVERY_RADIUS,
                           RESOURCE_MIN_SIZE as ENV_RESOURCE_MIN_SIZE,
                           RESOURCE_MAX_SIZE as ENV_RESOURCE_MAX_SIZE, HIVE_RADIUS_ASSUMED,
                           COLLISION_GROUP_RESOURCE, COLLISION_GROUP_AGENT,
                           COLLISION_GROUP_HIVE, COLLISION_GROUP_GROUND,
                           COLLISION_GROUP_OBSTACLE)
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Debugging ---
VERBOSE_DEBUG = True
DEBUG_PRINT_FREQ = 10

# ==============================================================================
# ===                         HELPER FUNCTIONS                             ===
# ==============================================================================
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def simple_move_controller(agent_pos: np.ndarray, target_pos: np.ndarray, approach_radius: float = -1.0) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    
    direction_to_target = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    distance = np.linalg.norm(direction_to_target)
    
    # --- Braking Logic ---
    # Scale force down linearly as the agent enters the approach_radius
    if approach_radius > 0 and distance < approach_radius:
        force_scale = max(0.1, distance / approach_radius) # Don't go to zero force
    else:
        force_scale = 1.0
        
    return normalize_vector(direction_to_target) * force_scale

# ==============================================================================
# ===                      MAIN TUNING LOGIC                                 ===
# ==============================================================================

def run_single_pickup_test(env_params: dict, run_config: dict):
    """
    Runs a single simulation for the pickup test.
    - Spawns N agents and N resources.
    - Each agent is tasked to pick up one resource and move to a hive.
    - Collects detailed metrics per agent.
    """
    run_name = run_config['name']
    seed = run_config['seed']
    render = run_config.get('render', False)
    max_steps = run_config.get('max_steps', 500)

    print(f"\n{'='*15} Starting Run: {run_name} {'='*15}")
    print(f"  Seed: {seed}")
    print(f"  Params: {json.dumps({k: (f'{v:.3f}' if isinstance(v, float) else v) for k,v in env_params.items()}, indent=2)}")
    
    env = None
    agent_results = []

    try:
        env = Swarm2DEnv(**env_params, max_steps=max_steps)
        _, _ = env.reset(seed=seed)

        num_agents = len(env.agents)
        if num_agents == 0:
            raise RuntimeError("No agents were spawned.")

        # --- Custom Setup: N agents, N resources, close proximity ---
        hive_pos = np.array([env.width * 0.9, env.height * 0.5])
        if 0 in env.hives:
            env.hives[0]['pos'] = hive_pos
            p.resetBasePositionAndOrientation(env.hive_body_ids[0], [hive_pos[0], hive_pos[1], HIVE_RADIUS_ASSUMED], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)

        start_y = env.height * 0.2
        spacing = (env.height * 0.6) / max(1, num_agents)

        for i in range(num_agents):
            # Place agent
            agent_pos = np.array([env.width * 0.4, start_y + i * spacing])
            p.resetBasePositionAndOrientation(env.agents[i]['body_id'], [agent_pos[0], agent_pos[1], AGENT_RADIUS], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)
            
            # Spawn a resource for this agent
            res_pos = np.array([env.width * 0.2, start_y + i * spacing])
            # Corrected method call:
            env.spawn_manager.resource_spawn._spawn_resource_at_location(res_pos, size=3.0, cooperative=False)

        # Let the simulation settle for a moment
        for _ in range(5):
            env.step([{'movement': np.zeros(2), 'pickup': 0}] * num_agents)

        agent_tasks = {i: {'resource_id': i, 'state': 'MOVING_TO_RESOURCE', 'picked_up': False, 'delivered': False, 'pickup_step': -1, 'delivery_step': -1} for i in range(num_agents)}

        # --- Main Simulation Loop ---
        for step in range(max_steps):
            if render:
                env.render()
                time.sleep(0.01)

            actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(num_agents)]
            
            all_done = True
            for i in range(num_agents):
                if agent_tasks[i]['state'] == 'DONE':
                    continue
                
                all_done = False
                agent = env.agents[i]

                resource_id_to_find = agent_tasks[i]['resource_id']
                resource = next((r for r in env.resources if r['id'] == resource_id_to_find), None)

                if resource is None:
                    if not agent_tasks[i]['delivered']: # If we haven't already marked as delivered
                        agent_tasks[i]['state'] = 'DONE'
                        agent_tasks[i]['delivered'] = True
                        agent_tasks[i]['delivery_step'] = step
                    continue
                
                if agent['has_resource']:
                    if not agent_tasks[i]['picked_up']:
                        agent_tasks[i]['picked_up'] = True
                        agent_tasks[i]['pickup_step'] = step
                    agent_tasks[i]['state'] = 'MOVING_TO_HIVE'
                    
                    # --- Simplified Full Thrust Controller ---
                    target_pos = hive_pos
                    agent_pos = agent['pos']
                    
                    # Always apply full force in the direction of the hive.
                    direction_to_hive = np.array(target_pos[:2]) - np.array(agent_pos[:2])
                    actions[i]['movement'] = normalize_vector(direction_to_hive)
                else:
                    dist_to_res = np.linalg.norm(agent['pos'] - resource['pos'])
                    if dist_to_res < (agent['agent_radius'] + resource['radius_pb']) * 1.2:
                        actions[i]['pickup'] = 1
                    
                    # Use the improved controller with braking
                    pickup_approach_radius = (agent['agent_radius'] + resource['radius_pb']) * 5.0
                    actions[i]['movement'] = simple_move_controller(agent['pos'], resource['pos'], approach_radius=pickup_approach_radius)

                # Check for delivery
                if resource['delivered']:
                    agent_tasks[i]['state'] = 'DONE'
                    agent_tasks[i]['delivered'] = True
                    agent_tasks[i]['delivery_step'] = step
            
            if all_done:
                break

            _, _, terminated, truncated, _ = env.step(actions)
            
            if VERBOSE_DEBUG and (step % DEBUG_PRINT_FREQ == 0):
                print(f"--- [Step {step:03d}] ---")
                for i in range(num_agents):
                    agent = env.agents[i]
                    task = agent_tasks[i]
                    
                    res_id_to_find = task['resource_id']
                    res = next((r for r in env.resources if r['id'] == res_id_to_find), None)
                    
                    agent_vel = agent.get('vel', np.zeros(2))
                    speed = np.linalg.norm(agent_vel)

                    if res:
                        dist_to_res = np.linalg.norm(agent['pos'] - res['pos'])
                        grip_str_text = ""
                        if agent['has_resource']:
                            grip = env.physics_manager._get_effective_grip_strength(agent)
                            grip_str_text = f"| Grip={grip:.0f} "
                        print(f"  Agent {i}: State={task['state']:<18} | HasRes={agent['has_resource']} | Pos=({agent['pos'][0]:.1f}, {agent['pos'][1]:.1f}) | Speed={speed:.1f} {grip_str_text}| Vel=({agent_vel[0]:.1f}, {agent_vel[1]:.1f}) | DistToRes={dist_to_res:.1f}")
                    else:
                        print(f"  Agent {i}: State={task['state']:<18} | HasRes={agent['has_resource']} | Pos=({agent['pos'][0]:.1f}, {agent['pos'][1]:.1f}) | Speed={speed:.1f} | Vel=({agent_vel[0]:.1f}, {agent_vel[1]:.1f}) | (Resource Delivered)")

                for i in range(len(env.resources)):
                    res = env.resources[i]
                    res_speed_val = 0.0
                    lin_vel_raw = (0,0,0)
                    if res.get('body_id') is not None:
                        try:
                            lin_vel, _ = p.getBaseVelocity(res['body_id'], physicsClientId=env.physicsClient)
                            lin_vel_raw = lin_vel
                            res_speed_val = np.linalg.norm(lin_vel[:2])
                        except p.error:
                            pass # Body might not exist yet
                    print(f"  Resource {i}: Pos=({res['pos'][0]:.1f}, {res['pos'][1]:.1f}) | Speed={res_speed_val:.1f} | Vel=({lin_vel_raw[0]:.1f}, {lin_vel_raw[1]:.1f}) | Carriers={len(res['carriers'])}")


            if terminated or truncated or all_done:
                break
        
        # --- Collect Final Results ---
        for i in range(num_agents):
            agent_results.append({
                'agent_id': i,
                **agent_tasks[i]
            })

    except Exception as e:
        print(f"ERROR during simulation for {run_name}: {e}")
        traceback.print_exc()
    finally:
        if env:
            env.close()

    return agent_results

if __name__ == "__main__":
    # Initialize Pygame if rendering
    if not pygame.get_init():
        pygame.init()
        pygame.display.init()

    # --- Define Parameter Sweep ---
    parameter_sweep = {
        # STEP 1: Find a stable constraint force
        # It's not about the absolute strength, but the stability of the simulation.
        'pb_resource_constraint_max_force': [3000],
        # STEP 2: Find a resource mass that provides significant slowdown.
        'resource_base_mass': [0.075], 
        
        # STEP 3: Find a damping value that prevents oscillation but doesn't feel like moving through mud.
        # This should be higher to handle the forces involved.
        'pb_res_damping_dynamic': [0.25],
        'pb_res_friction_dynamic': [0.35], 
        'pb_res_friction_static': [0.35],
        'resource_interaction_force_scale': [1.2], 
        'pb_res_damping_static': [0.25],
    }

    # --- Base Environment Config ---
    # These are assumed to be pre-calibrated from previous phases
    base_env_config = {
        # Increased force to compensate for the higher agent damping, roughly 15x the target damping.
        'movement_force_scale': 15.0, 
        'pb_agent_linear_damping': 0.11, # Keep this from previous tuning        
        # Lateral friction is crucial for preventing sideways skids during turns. 0.5 is the calibrated value.
        'pb_agent_lateral_friction': 0.5,
        'pb_agent_angular_damping': 0.4,
        'bee_speed': 200.0,
        'num_teams': 1,
        'num_agents_per_team': 4, # Test with 4 agents
        'num_resources': 0, # We spawn them manually
        'num_obstacles': 0,
        'render_mode': True,
    }

    all_results = []
    
    sweep_keys = list(parameter_sweep.keys())
    param_combinations = list(product(*(parameter_sweep[key] for key in sweep_keys)))
    
    base_seed = random.randint(0, 100000)
    print(f"Using base seed: {base_seed}")

    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(sweep_keys, combo))
        
        run_name = f"P2_Run_{i}"
        
        env_config = {**base_env_config, **current_params}
        
        run_config = {
            'name': run_name,
            'seed': base_seed + i,
            'render': True,
            'max_steps': 400
        }

        # Run the simulation
        agent_run_results = run_single_pickup_test(env_config, run_config)

        # Aggregate results for this run
        if agent_run_results:
            num_agents = len(agent_run_results)
            successful_pickups = sum(1 for r in agent_run_results if r['picked_up'])
            successful_deliveries = sum(1 for r in agent_run_results if r['delivered'])
            
            pickup_ratio = successful_pickups / num_agents if num_agents > 0 else 0
            delivery_ratio = successful_deliveries / num_agents if num_agents > 0 else 0

            run_summary = {
                'run_name': run_name,
                'params': current_params,
                'pickup_ratio': pickup_ratio,
                'delivery_ratio': delivery_ratio
            }
            all_results.append(run_summary)
            print(f"--- Result for {run_name} ---")
            print(f"  Pickup Ratio: {pickup_ratio:.2f} | Delivery Ratio: {delivery_ratio:.2f}")

    # --- Final Analysis ---
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n\n" + "="*20 + " FINAL SWEEP SUMMARY " + "="*20)
        
        # You can add scoring logic here to find the best parameter set
        # For now, just print the results sorted by delivery ratio
        
        # Flatten the params dict for better display
        params_df = pd.json_normalize(df['params'])
        results_df = df.drop(columns=['params'])
        final_df = pd.concat([results_df, params_df], axis=1)

        print(final_df.sort_values(by=['delivery_ratio', 'pickup_ratio'], ascending=False).to_string())

    if pygame.get_init():
        pygame.quit()
    
    print("\nPhase 2 tuning script finished.")
