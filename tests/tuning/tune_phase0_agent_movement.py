import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import json
import traceback
from itertools import product

# --- Python Path (ensure Swarm2DEnv is found) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import AGENT_RADIUS, FPS
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Debugging and Simulation Constants ---
VERBOSE_DEBUG = True
# NOTE: The script now prints every step during the crucial accel/decel phases,
# so this frequency is for the main travel phase.
DEBUG_PRINT_FREQ = 50 
SIM_TIME_STEP = 1.0 / FPS

# ==============================================================================
# ===                         HELPER FUNCTIONS                             ===
# ==============================================================================
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def simple_move_controller(agent_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    direction_to_target = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    return normalize_vector(direction_to_target)

# ==============================================================================
# ===                      MAIN TUNING LOGIC                                 ===
# ==============================================================================

def run_movement_test(env_params: dict, run_config: dict):
    """
    Runs a single simulation to test an agent's top speed.
    """
    run_name = run_config['name']
    seed = run_config['seed']
    render = run_config.get('render', False)
    max_steps = run_config.get('max_steps', 500)
    settle_steps = run_config.get('settle_steps', 30)
    
    print(f"\n{'='*15} Starting Run: {run_name} {'='*15}")
    print(f"  Seed: {seed}")
    print(f"  Params: {json.dumps({k: (f'{v:.3f}' if isinstance(v, float) else v) for k,v in env_params.items() if k in run_config['sweep_keys']}, indent=2)}")

    env = None
    result = { 
        "run_name": run_name, "achieved_avg_speed": 0.0, "stability_ok": True,
        "time_to_max_speed": 0.0, "deceleration_rate": 0.0
    }

    try:
        # Reduce max_steps as we are doing more focused testing
        env = Swarm2DEnv(**env_params, max_steps=settle_steps + 200) 
        _, _ = env.reset(seed=seed)

        if not env.agents:
            raise RuntimeError("No agents were spawned for the test.")
        
        agent = env.agents[0]
        
        # Move hive out of the way to not interfere
        if env.hives and 0 in env.hives:
            p.resetBasePositionAndOrientation(env.hive_body_ids[0], [env.width + 100, env.height + 100, 0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)

        start_pos = np.array([env.width * 0.1, env.height * 0.5])
        p.resetBasePositionAndOrientation(agent['body_id'], [start_pos[0], start_pos[1], agent['agent_radius']], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], [0,0,0],[0,0,0], physicsClientId=env.physicsClient)

        # Let simulation settle
        for _ in range(settle_steps):
            env.step([{'movement': np.zeros(2), 'pickup': 0}])

        # --- Detailed Acceleration Test ---
        accel_speeds = []
        print("  --- Acceleration Phase ---")
        for step in range(60): # A 60-step window is plenty for acceleration
            move_cmd = simple_move_controller(agent['pos'], np.array([env.width, env.height / 2]))
            env.step([{'movement': move_cmd, 'pickup': 0}])
            lin_vel, _ = p.getBaseVelocity(agent['body_id'], physicsClientId=env.physicsClient)
            speed = np.linalg.norm(lin_vel[:2])
            accel_speeds.append(speed)
            if VERBOSE_DEBUG:
                print(f"    [Accel Step {step:02d}] Speed: {speed:.2f}")
            if render: env.render(); time.sleep(0.01)

        # --- Detailed Deceleration Test ---
        decel_speeds = []
        print("  --- Deceleration Phase ---")
        for step in range(60): # A 60-step window for deceleration
            env.step([{'movement': np.zeros(2), 'pickup': 0}]) # No movement command
            lin_vel, _ = p.getBaseVelocity(agent['body_id'], physicsClientId=env.physicsClient)
            speed = np.linalg.norm(lin_vel[:2])
            decel_speeds.append(speed)
            if VERBOSE_DEBUG:
                print(f"    [Decel Step {step:02d}] Speed: {speed:.2f}")
            if render: env.render(); time.sleep(0.01)

        # --- Analysis ---
        if accel_speeds:
            top_speed = max(accel_speeds)
            # Find time to 90% of top speed
            t90_threshold = top_speed * 0.9
            time_to_t90 = next((i for i, s in enumerate(accel_speeds) if s >= t90_threshold), 0) * SIM_TIME_STEP
            result["time_to_max_speed"] = time_to_t90
            result["achieved_avg_speed"] = float(np.mean(accel_speeds[-20:])) # Avg of the last 20 steps of accel

        if len(decel_speeds) > 1:
            # Calculate average deceleration rate
            decelerations = -np.diff(decel_speeds) / SIM_TIME_STEP
            result["deceleration_rate"] = float(np.mean(decelerations)) if decelerations.size > 0 else 0.0

    except Exception as e:
        print(f"ERROR during simulation for {run_name}: {e}")
        traceback.print_exc()
        result["stability_ok"] = False
    finally:
        if env:
            env.close()

    return result


if __name__ == "__main__":
    if not pygame.get_init():
        pygame.init()
        pygame.display.init()

    # --- Define Parameter Sweep ---
    # We are testing specific pairs to achieve a target speed of ~200 u/s while tuning for acceleration feel.
    # The relationship is: speed ≈ 4.25 * (force_scale / damping)
    # To get speed ≈ 200, we need force_scale / damping ≈ 47.
    param_combinations = [
        # Lower damping = slower, more gradual acceleration.
        {'movement_force_scale': 23.5, 'pb_agent_linear_damping': 0.11, 'pb_agent_angular_damping': 0.6},
        
        # Medium damping = our previous baseline for a stable, responsive agent.
        {'movement_force_scale': 47.0, 'pb_agent_linear_damping': 1.0, 'pb_agent_angular_damping': 0.6},

        # Higher damping = stiffer, faster acceleration.
        {'movement_force_scale': 70.0, 'pb_agent_linear_damping': 1.5, 'pb_agent_angular_damping': 0.6},
    ]

    # --- Base Environment Config ---
    DESIRED_SOLO_AGENT_MAX_SPEED = 200.0
    base_env_config = {
        'pb_agent_mass': 1.0,
        'pb_agent_lateral_friction': 0.5,
        'pb_agent_restitution': 0.3,
        'bee_speed': DESIRED_SOLO_AGENT_MAX_SPEED,
        'num_teams': 1,
        'num_agents_per_team': 1,
        'num_resources': 0,
        'num_obstacles': 0,
        'render_mode': True,
    }

    all_results = []
    
    base_seed = random.randint(0, 100000)
    print(f"Using base seed: {base_seed}")

    for i, current_params in enumerate(param_combinations):
        run_name = f"P0_Run_{i}"
        
        env_config = {**base_env_config, **current_params}
        
        run_config = {
            'name': run_name,
            'seed': base_seed + i,
            'render': True,
            'max_steps': 400,
            'sweep_keys': list(current_params.keys())
        }

        # Run the simulation
        run_result = run_movement_test(env_config, run_config)
        
        if run_result:
            summary = {
                'run_name': run_name,
                'params': current_params,
                'avg_speed': run_result['achieved_avg_speed'],
                'time_to_max_speed': run_result['time_to_max_speed'],
                'deceleration_rate': run_result['deceleration_rate'],
                'stability_ok': run_result['stability_ok']
            }
            all_results.append(summary)
            print(f"--- Result for {run_name} ---")
            print(f"  Avg Speed: {summary['avg_speed']:.2f} | T_t90: {summary['time_to_max_speed']:.3f}s | Decel Rate: {summary['deceleration_rate']:.2f} | Stable: {summary['stability_ok']}")

    # --- Final Analysis ---
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n\n" + "="*20 + " FINAL SWEEP SUMMARY (PHASE 0) " + "="*20)
        
        params_df = pd.json_normalize(df['params'])
        results_df = df.drop(columns=['params'])
        final_df = pd.concat([results_df, params_df], axis=1)

        # Scoring: Find stable runs closest to the target speed
        stable_runs = final_df[final_df['stability_ok']].copy()
        if not stable_runs.empty:
            stable_runs['speed_diff'] = abs(stable_runs['avg_speed'] - DESIRED_SOLO_AGENT_MAX_SPEED)
            print("--- All Stable Runs ---")
            print(stable_runs.sort_values(by='speed_diff').to_string())
            
            best_run = stable_runs.loc[stable_runs['speed_diff'].idxmin()]
            print("\n--- Best Run ---")
            print(best_run)
        else:
            print("No stable runs found.")

    if pygame.get_init():
        pygame.quit()
    
    print("\nPhase 0 tuning script finished.")
