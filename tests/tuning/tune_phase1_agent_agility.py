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
import math

# --- Python Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import AGENT_RADIUS, FPS
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    sys.exit(1)

# --- Simulation Constants ---
VERBOSE_DEBUG = True
DEBUG_PRINT_FREQ = 5 # Print less frequently to speed up analysis
SIM_TIME_STEP = 1.0 / FPS
ACCEL_STEPS = 150
BRAKE_STEPS = 150
TURN_STEPS = 100
ARC_TEST_STEPS = 150
WAYPOINT_TEST_STEPS = 200 # Increased for comprehensive waypoint tests

# ==============================================================================
# ===                         HELPER FUNCTIONS                             ===
# ==============================================================================
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Returns a normalized version of the vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def simple_move_controller(agent_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    direction_to_target = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    return normalize_vector(direction_to_target)

def PD_turn_controller(agent: dict, target_yaw: float, p_client_id: int):
    """
    A robust PD controller to apply TORQUE for standstill turns ONLY.
    This prevents oscillation and ensures a pure pivot.
    """
    _, orientation = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=p_client_id)
    _, ang_vel_3d = p.getBaseVelocity(agent['body_id'], physicsClientId=p_client_id)
    _, _, current_yaw = p.getEulerFromQuaternion(orientation)
    current_angular_vel = ang_vel_3d[2]

    yaw_error = target_yaw - current_yaw
    if yaw_error > math.pi: yaw_error -= 2 * math.pi
    if yaw_error < -math.pi: yaw_error += 2 * math.pi

    p_gain = 70.0
    d_gain = 5.0
    torque = (p_gain * yaw_error) - (d_gain * current_angular_vel)
    
    p.applyExternalTorque(agent['body_id'], -1, [0, 0, torque], p.WORLD_FRAME, physicsClientId=p_client_id)
    return np.zeros(2) # No linear movement for a pivot turn




# ==============================================================================
# ===                         TEST SCENARIOS                                 ===
# ==============================================================================

def run_coasting_test(env_params: dict, run_config: dict) -> dict:
    """
    Tests how quickly an agent coasts to a near-stop from top speed due to friction.
    """
    env = None
    result = {"max_speed": 0.0, "coasting_dist": 0.0, "coasting_steps": 0}
    try:
        env = Swarm2DEnv(**env_params, max_steps=ACCEL_STEPS + BRAKE_STEPS)
        _, _ = env.reset(seed=run_config['seed'])
        agent = env.agents[0]

        start_pos = np.array([env.width * 0.1, env.height * 0.5])
        target_pos = np.array([env.width * 0.9, env.height * 0.5])
        p.resetBasePositionAndOrientation(agent['body_id'], [start_pos[0], start_pos[1], agent['agent_radius']], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)

        coasting_started = False
        coast_start_pos = None
        coast_start_step = -1
        max_speed_achieved = 0.0

        # --- Acceleration Phase ---
        for step in range(ACCEL_STEPS):
            move_cmd = simple_move_controller(agent['pos'], target_pos) * 500.0 # Apply strong force
            env.step([{'movement': move_cmd, 'pickup': 0}])
            
            pos3d, _ = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=env.physicsClient)
            vel3d, _ = p.getBaseVelocity(agent['body_id'], physicsClientId=env.physicsClient)
            agent['pos'], agent['vel'] = np.array(pos3d[:2]), np.array(vel3d[:2])

            current_speed = np.linalg.norm(agent['vel'])
            max_speed_achieved = max(max_speed_achieved, current_speed)

            if VERBOSE_DEBUG and (step % DEBUG_PRINT_FREQ == 0):
                print(f"  [Accelerating Step {step:03d}] Speed: {current_speed:.2f}")

        # --- Coasting Phase ---
        coasting_started = True
        coast_start_pos = agent['pos'].copy()
        result["max_speed"] = max_speed_achieved
        
        for step in range(BRAKE_STEPS):
            env.step([{'movement': np.zeros(2), 'pickup': 0}]) # No force applied

            pos3d, _ = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=env.physicsClient)
            current_vel3d, _ = p.getBaseVelocity(agent['body_id'], physicsClientId=env.physicsClient)
            speed = np.linalg.norm(current_vel3d[:2])
            
            dist_traveled = np.linalg.norm(np.array(pos3d[:2]) - coast_start_pos)
            if VERBOSE_DEBUG and (step % DEBUG_PRINT_FREQ == 0):
                print(f"  [Coasting Step {step:03d}] Speed: {speed:.2f}, Coast Dist: {dist_traveled:.2f}")

            if speed < 20.0: # Target speed of 20-30 u/s
                result["coasting_steps"] = step
                result["coasting_dist"] = dist_traveled
                break
            
            if run_config['render']: env.render(); time.sleep(0.01)

        if result["coasting_steps"] == 0:
            print(f"  [Coasting Test FAIL] Did not slow to < 20 u/s after {BRAKE_STEPS} steps.")
            result["coasting_steps"] = BRAKE_STEPS
            result["coasting_dist"] = np.linalg.norm(np.array(p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=env.physicsClient)[0][:2]) - coast_start_pos)

    finally:
        if env: env.close()
    return result

def run_standstill_turn_test(env_params: dict, run_config: dict):
    """
    Tests how quickly an agent can turn 180 degrees FROM A STANDSTILL.
    Now uses a proper PD controller to prevent oscillation.
    """
    env = None
    result = {"turn_180_steps": -1}
    try:
        env = Swarm2DEnv(**env_params, max_steps=TURN_STEPS)
        _, _ = env.reset(seed=run_config['seed'])
        agent = env.agents[0]
        
        # --- Test Setup (CENTERED in the map) ---
        center_x, center_y = env.width / 2.0, env.height / 2.0
        p.resetBasePositionAndOrientation(agent['body_id'], [center_x, center_y, agent['agent_radius']], p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=env.physicsClient)

        target_yaw = math.pi # 180 degrees
        
        for i in range(TURN_STEPS):
            # The PD controller handles both applying torque and ensuring no linear movement.
            move_cmd = PD_turn_controller(agent, target_yaw, env.physicsClient)
            env.step([{'movement': move_cmd, 'pickup': 0}])

            _, orientation = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=env.physicsClient)
            _, _, yaw = p.getEulerFromQuaternion(orientation)

            if VERBOSE_DEBUG and (i % DEBUG_PRINT_FREQ == 0):
                print(f"  [Turn Step {i:03d}] Yaw: {math.degrees(yaw):.1f} (Target: 180)")

            if abs(yaw - math.pi) < math.radians(10): # 10-degree tolerance
                result["turn_180_steps"] = i + 1
                break

            if run_config['render']: env.render(); time.sleep(0.01)
        
        if result["turn_180_steps"] == -1:
            print(f"  [Turn Test FAIL] Timed out after {TURN_STEPS} steps.")
            result["turn_180_steps"] = TURN_STEPS
    except Exception as e:
        print(f"Error during turning test: {e}")
    finally:
        if env: env.close()
    return result

def run_waypoint_turn_test(env_params: dict, run_config: dict) -> dict:
    """
    NEW: A clean, flexible test for high-speed turns based on your design.
    - Takes turn_angle and speed_fraction as parameters.
    - Measures drift and steps to complete the turn.
    - Includes a straight segment after the turn to observe stabilization.
    """
    env = None
    # Add new metrics to the result dictionary
    result = {
        "turn_drift": -1, 
        "turn_steps": -1,
        "per_step_data": [] # To store detailed metrics for each step
    }
    try:
        env = Swarm2DEnv(**env_params, max_steps=WAYPOINT_TEST_STEPS)
        _, _ = env.reset(seed=run_config['seed'])
        agent = env.agents[0]

        # --- Test Geometry ---
        center_x, center_y = env.width / 2.0, env.height / 2.0
        
        start_pos = [center_x - 200, center_y, AGENT_RADIUS]
        corner_pos = np.array([center_x, center_y])
        
        # Calculate the second waypoint based on the turn angle
        turn_angle_rad = math.radians(run_config['turn_angle'])
        exit_segment_len = 200

        waypoint_B_dir = np.array([math.cos(turn_angle_rad), math.sin(turn_angle_rad)])
        waypoint_B = corner_pos + waypoint_B_dir * exit_segment_len
        
        # Add a final destination to ensure the agent continues straight
        final_destination = waypoint_B + waypoint_B_dir * 300 # 300 units past waypoint_B

        p.resetBasePositionAndOrientation(agent['body_id'], start_pos, p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=env.physicsClient)

        current_target = corner_pos
        turn_initiated = False
        turn_finished_step = -1
        max_drift = 0.0
        
        for step in range(WAYPOINT_TEST_STEPS):
            pos3d, orientation = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=env.physicsClient)
            pos = pos3d[:2]
            vel3d, ang_vel3d = p.getBaseVelocity(agent['body_id'], physicsClientId=env.physicsClient)
            current_speed = np.linalg.norm(vel3d[:2])
            _, _, yaw = p.getEulerFromQuaternion(orientation)

            # --- State Transitions ---
            if not turn_initiated and np.linalg.norm(pos - corner_pos) < 20.0:
                turn_initiated = True
                current_target = waypoint_B

            if turn_initiated and turn_finished_step == -1:
                heading_to_target = normalize_vector(waypoint_B - pos)
                agent_heading = np.array([math.cos(yaw), math.sin(yaw)])
                if np.dot(agent_heading, heading_to_target) > 0.96: # approx 15 degrees
                    turn_finished_step = step
                    current_target = final_destination # Head for the exit point
            
            # --- Controller ---
            move_cmd = simple_move_controller(pos, current_target)
            thrust = run_config['max_speed'] * run_config['speed_fraction']
            env.step([{'movement': move_cmd * thrust, 'pickup': 0}])
            
            # --- Metric Collection (only after turn starts) ---
            drift = 0
            if turn_initiated:
                line_vec = waypoint_B - corner_pos
                point_vec = pos - corner_pos
                line_len_sq = np.dot(line_vec, line_vec)
                projection_len = np.dot(point_vec, line_vec) / (line_len_sq if line_len_sq > 0 else 1)
                # We don't clip projection here to correctly measure drift even if the agent is "behind" or "ahead" of the line segment
                projection = corner_pos + projection_len * line_vec
                drift = np.linalg.norm(pos - projection)
                max_drift = max(max_drift, drift)

            # Store per-step data for analysis
            result["per_step_data"].append({
                "step": step,
                "speed": current_speed,
                "drift": drift,
                "pos_x": pos[0],
                "pos_y": pos[1],
                "yaw": yaw
            })

            if run_config.get('render', False):
                env.render()
                time.sleep(1./FPS)
                if step % DEBUG_PRINT_FREQ == 0:
                    status = "Approach"
                    if turn_initiated and turn_finished_step != -1:
                        status = "Exiting"
                    elif turn_initiated:
                        status = "Turning"
                    print(f"  [WPT Step {step:03d}] Status: {status}, Speed: {current_speed:.1f}, Drift: {drift:.2f}")

        result["turn_drift"] = max_drift
        result["turn_steps"] = turn_finished_step if turn_finished_step != -1 else WAYPOINT_TEST_STEPS

    except Exception as e:
        print(f"Error during waypoint test: {e}")
        traceback.print_exc()
    finally:
        if env: env.close()
    return result


# ==============================================================================
# ===                      MAIN EXECUTION BLOCK                              ===
# ==============================================================================

if __name__ == "__main__":
    if not pygame.get_init():
        pygame.init(); pygame.display.init()

    # Final, focused parameter sweep.
    parameter_sweep = {
        'pb_agent_lateral_friction': [0.5],        # Test a range of "grip"
        'pb_agent_angular_damping':  [0.4],             # Test a range of "stability"
        'pb_agent_linear_damping':   [0.11],                 # This value is consistently good.
        'pb_agent_spinning_friction': [0.1],                 # Less important for this test.
    }

    base_env_config = {
        'movement_force_scale': 15.0,
        'bee_speed': 200.0,
        'num_teams': 1, 'num_agents_per_team': 1,
        'num_resources': 0, 'num_obstacles': 0,
        'render_mode': True,
    }

    all_results = []
    sweep_keys = list(parameter_sweep.keys())
    param_combinations = list(product(*(parameter_sweep[key] for key in sweep_keys)))
    base_seed = random.randint(0, 100000)

    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(sweep_keys, combo))
        env_config = {**base_env_config, **current_params}
        print(f"\n===== RUN {i+1}/{len(param_combinations)}: PARAMS: {current_params} =====")

        # --- Run one-time tests for this param set ---
        # Always render the first run's coasting and standstill tests for visual feedback
        first_run_render = i == 0
        
        coasting_results = run_coasting_test({**env_config, 'render_mode': first_run_render}, {'seed': base_seed + i, 'render': first_run_render})
        standstill_turn_results = run_standstill_turn_test({**env_config, 'render_mode': first_run_render}, {'seed': base_seed + i + 1000, 'render': first_run_render})

        if coasting_results['max_speed'] <= 0:
            print("  [FAIL] Agent failed to move in coasting test. Skipping.")
            continue

        # --- Run Comprehensive Waypoint Tests ---
        turn_angles_to_test = [45, 90, 180]
        speed_fractions_to_test = [0.8, 1.0] # 10%, 50%, 80%, 100%
        waypoint_results = {}
        all_per_step_data = {}

        for angle in turn_angles_to_test:
            for speed_frac in speed_fractions_to_test:
                # Render 50%, 80%, 100% speed tests
                should_render = speed_frac >= 0.5
                print(f"  --- Running Waypoint Test: {angle}deg @ {int(speed_frac*100)}% Speed (Render: {should_render}) ---")
                
                waypoint_config = {
                    'seed': base_seed + i + int(angle * 100) + int(speed_frac * 100), 
                    'render': should_render, 
                    'max_speed': coasting_results['max_speed'],
                    'speed_fraction': speed_frac,
                    'turn_angle': angle
                }
                result = run_waypoint_turn_test({**env_config, 'render_mode': should_render}, waypoint_config)
                
                key_prefix = f'{angle}deg_{int(speed_frac*100)}'
                waypoint_results[f'drift_{key_prefix}'] = result['turn_drift']
                waypoint_results[f'steps_{key_prefix}'] = result['turn_steps']
                all_per_step_data[key_prefix] = result['per_step_data']

        summary = {
            'params': current_params,
            'max_speed': coasting_results['max_speed'],
            'coasting_dist': coasting_results['coasting_dist'],
            'standstill_turn_steps': standstill_turn_results['turn_180_steps'],
            **waypoint_results
        }
        all_results.append(summary)

        # Optionally save the detailed per-step data to a file for deeper analysis
        # with open(f'per_step_data_run_{i}.json', 'w') as f:
        #     json.dump(all_per_step_data, f, indent=4)
    
    # --- Analysis ---
    if all_results:
        df = pd.DataFrame(all_results)
        params_df = pd.json_normalize(df['params'])
        results_df = df.drop(columns=['params'])
        final_df = pd.concat([results_df, params_df], axis=1)

        # Basic scoring for now: prioritize low drift at 70% speed 90-degree turn
        final_df['score'] = 0
        
        # Update score to use a more relevant metric, e.g., drift at 90deg 80% speed
        score_metric = 'drift_90deg_80'
        if score_metric in final_df.columns:
            valid_mask = final_df[score_metric] > 0
            if valid_mask.any():
                norm_drift = final_df.loc[valid_mask, score_metric] / final_df.loc[valid_mask, score_metric].max()
                norm_steps = final_df.loc[valid_mask, f'steps_{score_metric.replace("drift_", "")}'] / final_df.loc[valid_mask, f'steps_{score_metric.replace("drift_", "")}'].max()
                norm_standstill = final_df.loc[valid_mask, 'standstill_turn_steps'] / final_df.loc[valid_mask, 'standstill_turn_steps'].max()
                
                final_df.loc[valid_mask, 'score'] = (1 - norm_drift) * 0.6 + (1 - norm_steps) * 0.2 + (1 - norm_standstill) * 0.2
        else:
            valid_mask = pd.Series([True] * len(final_df)) # All runs are valid if score metric is absent
        
        print("\n\n" + "="*20 + " FINAL SWEEP SUMMARY (PHASE 1) " + "="*20)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Display a subset of the most important columns for readability
        display_cols = [
            'pb_agent_lateral_friction', 'pb_agent_angular_damping', 'score',
            'standstill_turn_steps', 
            'drift_90deg_80', 'steps_90deg_80',
            'drift_45deg_100', 'steps_45deg_100', 
            'drift_180deg_80', 'steps_180deg_80',
            'drift_360deg_100', 'steps_360deg_100'
        ]
        display_cols = [col for col in display_cols if col in final_df.columns]
        
        print(final_df.sort_values(by='score', ascending=False)[display_cols].to_string(index=False))
        
        valid_runs = final_df[valid_mask]
        if not valid_runs.empty:
            best_run_idx = valid_runs['score'].idxmax()
            print("\n--- Best Run ---")
            print(valid_runs.loc[best_run_idx])
        else:
            print("\n--- No valid runs found to determine a best run. ---")

    if pygame.get_init():
        pygame.quit()
        
    print("\nPhase 1 tuning script finished.")
