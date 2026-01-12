import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import json
import math
from itertools import product

# --- Python Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import AGENT_RADIUS, AGENT_BASE_STRENGTH, BEE_SPEED
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
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

def unified_move_controller(agent_pos: np.ndarray, target_pos: np.ndarray, is_final_approach: bool = False) -> np.ndarray:
    """
    A unified controller that handles both long-distance travel and final approach,
    slowing down as it gets closer to the target.
    """
    if agent_pos is None or target_pos is None: return np.zeros(2)
    
    direction_vector = target_pos - agent_pos
    distance = np.linalg.norm(direction_vector)
    
    if distance < 1e-6:
        return np.zeros(2)

    # Define thresholds for slowing down
    slowdown_radius = AGENT_RADIUS * (3.0 if is_final_approach else 8.0)
    stop_radius = AGENT_RADIUS * 0.5

    if distance < stop_radius:
        return np.zeros(2)

    # Calculate force scale
    if distance < slowdown_radius:
        # Scale down force proportionally within the slowdown radius
        force_scale = (distance - stop_radius) / (slowdown_radius - stop_radius)
        force_scale = max(0.1, force_scale) # Ensure a minimum movement force
    else:
        # Full force outside the slowdown radius
        force_scale = 1.0

    return normalize_vector(direction_vector) * force_scale


def coop_move_controller(resource_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    if resource_pos is None or target_pos is None: return np.zeros(2)
    direction_to_target = np.array(target_pos[:2]) - np.array(resource_pos[:2])
    return normalize_vector(direction_to_target)

def get_attachment_points(resource_pos: np.ndarray, num_agents: int, resource_radius: float) -> np.ndarray:
    """Calculates evenly spaced attachment points around the resource."""
    points = []
    for i in range(num_agents):
        angle = (2 * np.pi / num_agents) * i
        offset = np.array([np.cos(angle), np.sin(angle)]) * (resource_radius + AGENT_RADIUS * 0.9)
        points.append(resource_pos + offset)
    return np.array(points)

# ==============================================================================
# ===                         TEST SCENARIO                                  ===
# ==============================================================================

def run_coop_carry_test(env_params: dict, run_config: dict):
    """
    Runs a cooperative carrying test with a specified number of agents.
    """
    env = None
    result = {"avg_carrying_speed": 0.0, "delivered": False}
    render = run_config.get('render', False)
    num_agents_for_task = run_config['num_agents_for_task']
    
    try:
        env = Swarm2DEnv(**env_params, max_steps=run_config['max_steps'])
        _, _ = env.reset(seed=run_config['seed'])

        # --- Standardize Agent Properties for Consistent Tuning ---
        for agent in env.agents:
            if agent:
                agent['strength'] = AGENT_BASE_STRENGTH
                agent['speed'] = BEE_SPEED
                agent['agent_radius'] = AGENT_RADIUS
        # --- End Standardization ---

        # Spawn a large cooperative resource
        res_pos = np.array([env.width * 0.2, env.height * 0.5])
        env.spawn_manager.resource_spawn._spawn_resource_at_location(res_pos, size=9.0, cooperative=True)
        resource = env.resources[0]

        # Agents are not pre-positioned around the resource anymore.
        # They will be spawned at default locations and navigate.

        hive_pos = np.array([env.width * 0.8, env.height * 0.5])
        if 0 in env.hives: env.hives[0]['pos'] = hive_pos
        
        speeds = []
        agent_states = {i: "MOVING_TO_RESOURCE" for i in range(num_agents_for_task)}
        
        for step in range(run_config['max_steps']):
            actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
            
            attachment_points = get_attachment_points(resource['pos'], num_agents_for_task, resource['radius_pb'])

            # Control assigned agents
            for i in range(num_agents_for_task):
                agent = env.agents[i]
                agent_pos = agent['pos']

                if agent['has_resource']:
                    agent_states[i] = "MOVING_TO_HIVE"
                    actions[i]['movement'] = coop_move_controller(agent_pos, hive_pos)
                else:
                    # Move to assigned attachment point
                    target_pos = attachment_points[i]
                    actions[i]['movement'] = unified_move_controller(agent_pos, target_pos, is_final_approach=True)

                    # Attempt pickup when close enough to the resource itself
                    if np.linalg.norm(agent_pos - resource['pos']) < resource['radius_pb'] + AGENT_RADIUS:
                        agent_states[i] = "ATTEMPTING_PICKUP"
                        actions[i]['pickup'] = 1
                    else:
                        agent_states[i] = "MOVING_TO_RESOURCE"


            _, _, terminated, truncated, _ = env.step(actions)

            if VERBOSE_DEBUG and (step % DEBUG_PRINT_FREQ == 0):
                res_speed_val = 0.0
                if resource.get('body_id') is not None:
                    try:
                        lin_vel, _ = p.getBaseVelocity(resource['body_id'], physicsClientId=env.physicsClient)
                        res_speed_val = np.linalg.norm(lin_vel[:2])
                    except p.error: pass
                
                print(f"--- [Coop Step {step:03d}] ---")
                print(f"  Resource -> Pos: ({resource['pos'][0]:.1f}, {resource['pos'][1]:.1f}) | Speed: {res_speed_val:.1f} | Carriers: {len(resource['carriers'])}/{num_agents_for_task}")
                
                # Detailed agent and constraint info
                for i in range(num_agents_for_task):
                    agent = env.agents[i]
                    agent_state_info = agent_states.get(i, "IDLE")
                    
                    force_info = ""
                    if agent['has_resource']:
                        constraint_id = resource.get('carrier_constraints', {}).get(agent['id'])
                        if constraint_id is not None:
                            try:
                                _, _, joint_reaction_forces = p.getConstraintState(constraint_id, physicsClientId=env.physicsClient)
                                force_magnitude = np.linalg.norm(joint_reaction_forces)
                                force_info = f"| Force: {force_magnitude:<7.2f}"
                            except (p.error, ValueError):
                                force_info = "| Force: N/A"
                    print(f"    Agent {agent['id']}: State: {agent_state_info:<18} {force_info}")
            
            if len(resource['carriers']) == num_agents_for_task:
                lin_vel, _ = p.getBaseVelocity(resource['body_id'], physicsClientId=env.physicsClient)
                speeds.append(np.linalg.norm(lin_vel[:2]))
            
            if resource['delivered']:
                result['delivered'] = True
                break
            if terminated or truncated:
                break
            if render: env.render(); time.sleep(0.01)

        if speeds:
            result['avg_carrying_speed'] = float(np.mean(speeds))
    finally:
        if env: env.close()
    return result

# ==============================================================================
# ===                      MAIN EXECUTION BLOCK                              ===
# ==============================================================================

if __name__ == "__main__":
    if not pygame.get_init():
        pygame.init(); pygame.display.init()

    parameter_sweep = {
        'resource_mass_scale_factor': [1.4],
        'pb_coop_resource_constraint_max_force': [10000],
        'pb_res_damping_dynamic': [0.25],
    }
    
    # --- Assumed calibrated values from previous phases ---
    base_env_config = {
        'movement_force_scale': 15.0, 'pb_agent_linear_damping': 0.11,
        'pb_agent_lateral_friction': 0.5, 'pb_agent_angular_damping': 0.4,
        'resource_base_mass': 0.075, 'resource_interaction_force_scale': 1.2,
        'pb_resource_constraint_max_force': 3000, 'pb_res_friction_dynamic': 0.25,
        'pb_res_damping_dynamic': 0.25, 'agent_interaction_force_scale': 0.3,
        'pb_agent_constraint_max_force': 3000, 'bee_speed': 200.0,
        'num_teams': 1, 'num_agents_per_team': 5,
        'num_resources': 0, 'num_obstacles': 0,
        'render_mode': True,
    }

    N_AGENTS = 3
    all_results = []
    sweep_keys = list(parameter_sweep.keys())
    param_combinations = list(product(*(parameter_sweep[key] for key in sweep_keys)))
    base_seed = random.randint(0, 100000)

    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(sweep_keys, combo))
        env_config = {**base_env_config, **current_params}
        
        # Test with N agents
        config_n = {'seed': base_seed + i, 'render': True, 'max_steps': 300, 'num_agents_for_task': N_AGENTS}
        result_n = run_coop_carry_test(env_config, config_n)
        
        # Test with N-1 agents
        config_n1 = {'seed': base_seed + i + 1000, 'render': True, 'max_steps': 300, 'num_agents_for_task': N_AGENTS - 1}
        result_n1 = run_coop_carry_test(env_config, config_n1)

        summary = {
            'params': current_params,
            'speed_N_agents': result_n['avg_carrying_speed'],
            'delivered_N_agents': result_n['delivered'],
            'speed_N-1_agents': result_n1['avg_carrying_speed'],
            'delivered_N-1_agents': result_n1['delivered'],
        }
        all_results.append(summary)
        print(f"--- Result for {current_params} ---")
        print(f"  N Agents: Speed={summary['speed_N_agents']:.2f}, Delivered={summary['delivered_N_agents']}")
        print(f"  N-1 Agents: Speed={summary['speed_N-1_agents']:.2f}, Delivered={summary['delivered_N-1_agents']}")

    if all_results:
        df = pd.DataFrame(all_results)
        params_df = pd.json_normalize(df['params'])
        results_df = df.drop(columns=['params'])
        final_df = pd.concat([results_df, params_df], axis=1)

        # Scoring: N agents should deliver, N-1 should not (or be very slow)
        final_df['score'] = (final_df['delivered_N_agents'] * 1.0) + \
                            ((1 - final_df['delivered_N-1_agents']) * 0.5) + \
                            (final_df['speed_N_agents'] / 100) - \
                            (final_df['speed_N-1_agents'] / 100)
        
        print("\n\n" + "="*20 + " FINAL SWEEP SUMMARY (PHASE 4) " + "="*20)
        print(final_df.sort_values(by='score', ascending=False).to_string())
        
        best_run = final_df.loc[final_df['score'].idxmax()]
        print("\n--- Best Run ---")
        print(best_run)

    if pygame.get_init():
        pygame.quit()
    print("\nPhase 4 tuning script finished.")
