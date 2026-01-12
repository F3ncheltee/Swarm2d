#!/usr/bin/env python3
import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import traceback
from collections import defaultdict
from itertools import product

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
                           COLLISION_GROUP_OBSTACLE, BEE_SPEED, COMBAT_RADIUS,
                           GRAPPLE_TORQUE_ESCAPE_STRENGTH, GRAPPLE_FATIGUE_RATE, GRAPPLE_CRUSH_DAMAGE_RATE)
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    traceback.print_exc()
    sys.exit(1)



# ==============================================================================
# ===                 TESTING FOCUS & CONTROL CONSTANTS                      ===
# ==============================================================================
MAX_STEPS_PER_SCENARIO = 250
RENDER_MODE = True
STEP_DELAY = 0.015 
PAUSE_BETWEEN_SCENARIOS_SEC = 1.0
SCENARIO_BASE_SEED = random.randint(0, 100000)
VERBOSE_DEBUG = True

# ============================================================================
# ===                         HELPER FUNCTIONS                             ===
# ============================================================================

def print_debug_to_console(scenario_data: dict, all_agents: list, pairs: list, env):
    """Prints detailed, real-time debug information to the console for tuning."""
    header = f"--- Step: {scenario_data['steps_completed']:<4} | Scenario: {scenario_data['scenario_name']} ---"
    print(header)

    for i, pair_indices in enumerate(pairs):
        grappler = all_agents[pair_indices[0]]
        target = all_agents[pair_indices[1]]
        
        print(f"  --- Pair {i+1} (IDs: {grappler.get('id', 'N/A')} vs {target.get('id', 'N/A')}) ---")

        for agent_role, agent in [("Grappler", grappler), ("Target", target)]:
            if not agent or not agent.get('alive'):
                print(f"    {agent_role}: DEAD")
                continue

            status = "IDLE/MOVING"
            if agent.get('is_grappling'): status = "GRAPPLING"
            elif agent.get('is_grappled'): status = "GRAPPLED"

            # Use the new detailed grip strength component function
            grip_components = {}
            if hasattr(env.physics_manager, '_get_grip_strength_components'):
                grip_components = env.physics_manager._get_grip_strength_components(agent)

            potential_grip = grip_components.get('final_grip', 0.0)
            
            # This is the actual force applied to the constraint in the last step
            actual_constraint_force = agent.get('grapple_last_set_force', 0.0)
            
            combat_strength = 0.0
            if hasattr(env.physics_manager, '_get_effective_combat_strength'):
                combat_strength = env.physics_manager._get_effective_combat_strength(agent)

            health = agent.get('health', 0.0)
            energy = agent.get('energy', 0.0)
            speed = np.linalg.norm(agent.get('vel', np.zeros(2)))
            pos = agent.get('pos', np.zeros(2))
            mass = agent.get('mass', 0.0)
            torque = agent.get('applied_torque', 0.0)
            momentum_bonus = agent.get('grapple_momentum_bonus', 0.0)
            applied_mov_force = agent.get('debug_applied_force', 0.0)

            print(f"    {agent_role}: {status}")
            print(f"      - Stats: Health: {health:.1f} | Energy: {energy:.1f} | Speed: {speed:.2f} | Mass: {mass:.2f}")
            print(f"      - Pos: ({pos[0]:.1f}, {pos[1]:.1f}) | Torque: {torque:.2f}")

            if status == "GRAPPLING":
                print(f"      - Grip Details (Potential: {potential_grip:.1f}, Actual Force: {actual_constraint_force:.1f})")
                print(f"        - Factors: Base: {grip_components.get('base_grip', 0):.1f} | E: {grip_components.get('energy_factor', 0):.2f}x | Str: {grip_components.get('strength_factor', 0):.2f}x | HP: {grip_components.get('health_factor', 0):.2f}x")
                
                # Let's also get the target's counter-torque to see the penalty
                target_torque_penalty = 1.0
                target_agent = all_agents[target.get('id')] # refetch
                if target_agent:
                    grappler_torque = agent.get('applied_torque', 0.0)
                    target_torque = target_agent.get('applied_torque', 0.0)
                    if np.sign(grappler_torque) != np.sign(target_torque) and abs(target_torque) > 0.1:
                        ratio = np.clip(abs(target_torque) / (abs(grappler_torque) + 1e-6), 0.0, 1.0)
                        target_torque_penalty = 1.0 - (env.grapple_torque_escape_strength * ratio)

                print(f"        - Bonuses: Momentum: {momentum_bonus:.2f} | Penalties: Torque: {target_torque_penalty:.2f}x | Fatigue: {grip_components.get('fatigue_factor', 1.0):.2f}x")
            
            elif status == "GRAPPLED":
                # For the grappled, show their breakout attempt force vs the grappler's hold
                grappler_potential_grip = 0.0
                if hasattr(env.physics_manager, '_get_effective_grip_strength'):
                    grappler_potential_grip = env.physics_manager._get_effective_grip_strength(grappler)

                effective_escape_force = 0.0
                if hasattr(env.physics_manager, '_get_effective_escape_strength'):
                    # In this context, 'agent' is the one being grappled (the target)
                    effective_escape_force = env.physics_manager._get_effective_escape_strength(agent)

                print(f"      - Escape Metrics: Desired: {applied_mov_force:.1f} | Effective: {effective_escape_force:.1f} vs Grappler's Grip: {grappler_potential_grip:.1f}")
            
            print(f"      - Forces: Movement (desired): {applied_mov_force:.1f} | Combat Strength: {combat_strength:.1f}")

    print("-" * (len(header) + 2))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def move_towards(agent_pos: np.ndarray, target_pos: np.ndarray, stop_dist: float = 1.0) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    direction = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    if np.linalg.norm(direction) < stop_dist: return np.zeros(2)
    return np.clip(normalize_vector(direction), -1.0, 1.0)

def setup_agent_state(env, agent_idx, pos, team, **kwargs):
    agent = env.agents[agent_idx]
    agent.update({
        'alive': True, 'pos': np.array(pos), 'vel': np.zeros(2),
        'team': team,
        'health': kwargs.get('health', agent['max_health']),
        'energy': kwargs.get('energy', agent['max_energy']),
        'strength': kwargs.get('strength', agent['strength']),
    })
    try:
        p.resetBasePositionAndOrientation(agent['body_id'], [pos[0], pos[1], agent['agent_radius']], [0,0,0,1], physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], kwargs.get('vel', [0,0,0]), [0,0,0], physicsClientId=env.physicsClient)
    except p.error as e:
        if VERBOSE_DEBUG: print(f"Warning: PyBullet error resetting agent {agent_idx}: {e}")

# ============================================================================
# ===                         SCENARIO DEFINITIONS                         ===
# ============================================================================

def setup_tug_of_war(env):
    """Setup: Four pairs of agents grapple and pull apart."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150  # Spread pairs vertically
        center_y = env.height / 2 + offset_y
        
        grappler_idx = i
        target_idx = i + 4
        
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=1.5)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=1.5)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_tug_of_war(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]
        
        # Grappler initiates and pulls left
        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.0)
                actions[idx_grappler]['pickup'] = 1
            else:
                actions[idx_grappler]['movement'] = np.array([-1.0, 0.0])
        
        # Target pulls right ONLY IF grappled
        if target['alive']:
            if target.get('is_grappled'):
                actions[idx_target]['movement'] = np.array([1.0, 0.0])
            else:
                actions[idx_target]['movement'] = np.zeros(2)
            
    return actions

def setup_energy_drain_break(env):
    """Setup: Four pairs with strong grapplers that have draining energy."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=2.0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=1.0)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_energy_drain_break(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]
        
        if grappler['alive']:
            if grappler.get('is_grappling'):
                grappler['energy'] = max(0, grappler['energy'] - 2.0) # Fast drain
            
            if not grappler.get('is_grappling'):
                actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.5)
                actions[idx_grappler]['pickup'] = 1
            else:
                actions[idx_grappler]['movement'] = np.array([-1.0, 0.0])

        if target['alive']:
            actions[idx_target]['movement'] = np.array([1.0, 0.0])
            
    return actions

def setup_momentum_grapple(env):
    """Setup: Four pairs of stationary grapplers and fast-moving targets."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.5, center_y], team=0)
        setup_agent_state(env, target_idx, [20, center_y], team=1, vel=[BEE_SPEED * 1.5, 0, 0])
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_momentum_grapple(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]
        
        if grappler['alive'] and not grappler.get('is_grappling'):
            dist = np.linalg.norm(np.array(target['pos']) - np.array(grappler['pos']))
            if dist < AGENT_RADIUS * 8 and grappler['pos'][0] < target['pos'][0]:
                actions[idx_grappler]['pickup'] = 1
                
    return actions

def setup_release_test(env):
    """Setup: Four pairs of strong grapplers, weak targets to test manual release."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=2.0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=0.7)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_release_test(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]
        
        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS*1.5)
                actions[idx_grappler]['pickup'] = 1
            elif step < 100:
                actions[idx_grappler]['movement'] = np.array([-1.0, 0.0]) # Hold for a bit
            else:
                actions[idx_grappler]['pickup'] = 2 # Release
                
        if target['alive']:
            actions[idx_target]['movement'] = np.array([1.0, 0.0])
            
    return actions

def setup_dominant_grapple(env):
    """Setup: Four pairs of very strong grapplers against weak targets."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=2.5)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=0.5)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_dominant_grapple(env, active_pairs, step):
    """Control: Both agents pull away, testing if the strong one can hold indefinitely."""
    return control_tug_of_war(env, active_pairs, step) # Same control logic

def setup_torque_struggle(env):
    """Setup: Four pairs of agents with equal strength to test the torque mechanic."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=1.5)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=1.5)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_torque_struggle(env, active_pairs, step):
    """Control: Grappler tries to spin, target tries to counter-spin."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]

        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS*1.5)
                actions[idx_grappler]['pickup'] = 1
            else:
                actions[idx_grappler]['movement'] = np.array([0.0, 1.0]) # Use y-axis for torque
        
        if target['alive'] and target.get('is_grappled'):
            actions[idx_target]['movement'] = np.array([0.0, -1.0]) # Counter-torque

    return actions

def setup_breakout_test(env):
    """Setup: A standard grapple where the target actively tries to break free."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], team=0, strength=1.5)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], team=1, strength=1.5)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_breakout_test(env, active_pairs, step):
    """Control: Target uses the 'pickup: 2' action to try and break the grapple."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_grappler, idx_target = pair[0], pair[1]
        grappler, target = env.agents[idx_grappler], env.agents[idx_target]

        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.0)
                actions[idx_grappler]['pickup'] = 1
            else:
                actions[idx_grappler]['movement'] = np.array([-1.0, 0.0]) # Pull

        if target['alive']:
            actions[idx_target]['movement'] = np.array([1.0, 0.0]) # Pull back
            if target.get('is_grappled'):
                actions[idx_target]['pickup'] = 2 # Attempt to break free

    return actions

# ============================================================================
# ===                     CORE SIMULATION RUNNER                           ===
# ============================================================================
def run_scenario(env, scenario_def: dict, current_params: dict):
    name, setup_fn, control_fn = scenario_def['name'], scenario_def['setup_fn'], scenario_def['control_fn']
    if VERBOSE_DEBUG: print(f"  -- Running scenario: {name}")

    env.reset(seed=SCENARIO_BASE_SEED + sum(ord(c) for c in name))
    active_pairs = setup_fn(env)
    
    # Initialize results for each pair
    results = [
        {
            "pair_num": i + 1,
            "scenario_name": name, "grapple_success_step": -1, "grapple_break_step": -1,
            "final_outcome": "Timeout", "grip_strength_history": [], "target_speed_history": []
        } for i in range(len(active_pairs))
    ]
    
    for step in range(MAX_STEPS_PER_SCENARIO):
        actions = control_fn(env, active_pairs, step)
        
        # Manually add the applied movement force to the agent's dict for debugging
        for i in range(len(env.agents)):
            # The 'movement' is a normalized vector, we need to scale it by the actual force applied
            # This is an approximation, but gives us a good idea of the forces involved
            mov_vec = actions[i].get('movement', np.zeros(2))
            # The internal force calculation is complex, so we use a simplified proxy
            # based on the environment's scale factor
            force_magnitude = np.linalg.norm(mov_vec) * env.movement_force_scale * 50 # Heuristic scale
            env.agents[i]['debug_applied_force'] = force_magnitude

        env.step(actions)

        if RENDER_MODE:
            env.render()
            time.sleep(STEP_DELAY)

        if step % 25 == 0 or step == MAX_STEPS_PER_SCENARIO - 1:
            print_debug_to_console(
                {"scenario_name": name, "steps_completed": step},
                env.agents,
                active_pairs,
                env
            )

        # Update results for each pair
        for i, pair in enumerate(active_pairs):
            grappler, target = env.agents[pair[0]], env.agents[pair[1]]
            res = results[i]

            if res["grapple_success_step"] == -1 and grappler.get('is_grappling'):
                res["grapple_success_step"] = step
            
            if grappler.get('is_grappling'):
                res['grip_strength_history'].append(grappler.get('grapple_last_set_force', 0.0))
            if target.get('is_grappled'):
                speed = np.linalg.norm(target.get('vel', np.zeros(2)))
                res['target_speed_history'].append(speed)

            if res["grapple_success_step"] != -1 and res["grapple_break_step"] == -1 and not grappler.get('is_grappling'):
                res["grapple_break_step"] = step
                res["final_outcome"] = "GripBreak"
            
            if not target['alive']:
                res["final_outcome"] = "TargetKilled"

        # End scenario if all pairs are done
        if all(r["final_outcome"] != "Timeout" for r in results):
            break

    # Final outcome check for pairs that timed out
    for res in results:
        if res["final_outcome"] == "Timeout":
            if res["grapple_success_step"] != -1 and res["grapple_break_step"] == -1:
                res["final_outcome"] = "Hold"
    
    # Calculate summary stats for each pair's result
    for res in results:
        grip_hist = res.pop('grip_strength_history', [])
        speed_hist = res.pop('target_speed_history', [])
        res['avg_grip'] = np.mean(grip_hist) if grip_hist else 0
        res['max_grip'] = np.max(grip_hist) if grip_hist else 0
        res['avg_target_speed'] = np.mean(speed_hist) if speed_hist else 0

    return results

# ==============================================================================
# ===                      MAIN EXECUTION BLOCK                              ===
# ==============================================================================

if __name__ == "__main__":
    if RENDER_MODE and not pygame.get_init():
        pygame.init(); pygame.display.init()

    # Define the parameter sweep
    # Best combo so far: 'agent_interaction_force_scale': 0.35, 'pb_agent_constraint_max_force': 12000, 'grapple_momentum_bonus_scale': 0.0, 'grapple_torque_escape_strength': 0.6, 'grapple_fatigue_rate': 0.015
    parameter_sweep = {
        # Keep this low to reduce physics instability
        'agent_interaction_force_scale': [0.35],
        # Vary this to hit our target "Base Grip" range of ~2000-4000
        'pb_agent_constraint_max_force': [13000],
        'grapple_momentum_bonus_scale': [0.25],
        # Let's make the torque struggle more impactful
        'grapple_torque_escape_strength': [0.4, 0.6],
        # Let's test different struggle durations
        'grapple_fatigue_rate': [0.015, 0.025],
        'grapple_crush_damage_rate': [0.1],
        'grapple_momentum_decay': [0.95],
    }

    base_env_config = {
        'num_teams': 2, 'num_agents_per_team': 4, # 4 pairs of agents
        'num_resources': 0, 'num_obstacles': 0,
        'render_mode': RENDER_MODE,
        'movement_force_scale': 15.0, 'pb_agent_linear_damping': 0.11,
        'pb_agent_lateral_friction': 0.5, 'pb_agent_angular_damping': 0.4,
        'bee_speed': 200.0,
    }

    scenarios_to_run = [
        {"name": "Tug of War", "setup_fn": setup_tug_of_war, "control_fn": control_tug_of_war},
        {"name": "Energy Drain Break", "setup_fn": setup_energy_drain_break, "control_fn": control_energy_drain_break},
        {"name": "Momentum Grapple", "setup_fn": setup_momentum_grapple, "control_fn": control_momentum_grapple},
        {"name": "Manual Release", "setup_fn": setup_release_test, "control_fn": control_release_test},
        {"name": "Dominant Grapple", "setup_fn": setup_dominant_grapple, "control_fn": control_dominant_grapple},
        {"name": "Torque Struggle", "setup_fn": setup_torque_struggle, "control_fn": control_torque_struggle},
        {"name": "Active Breakout", "setup_fn": setup_breakout_test, "control_fn": control_breakout_test},
    ]

    all_results = []
    env = None
    try:
        sweep_keys = list(parameter_sweep.keys())
        param_combinations = list(product(*(parameter_sweep[key] for key in sweep_keys)))

        for i, combo in enumerate(param_combinations):
            current_params = dict(zip(sweep_keys, combo))
            print(f"\n{'='*20} Running Sweep {i+1}/{len(param_combinations)} with params: {current_params} {'='*20}")
            
            env_config = {**base_env_config, **current_params}
            
            if env: env.close()
            env = Swarm2DEnv(**env_config)

            for scenario_def in scenarios_to_run:
                run_results_list = run_scenario(env, scenario_def, current_params)
                
                for run_result in run_results_list:
                    # Add params to result for easy analysis
                    summary = {'params': current_params, **run_result}
                    all_results.append(summary)
                    
                    # Detailed print
                    outcome = summary['final_outcome']
                    success_step = summary['grapple_success_step']
                    break_step = summary['grapple_break_step']
                    duration = -1
                    if success_step != -1:
                        duration = (break_step - success_step) if break_step != -1 else (MAX_STEPS_PER_SCENARIO - success_step)

                    print(f"    -> Pair {summary['pair_num']} Result: {outcome} | Success: step {success_step} | Break: step {break_step} | Duration: {duration}")
                
            if RENDER_MODE: time.sleep(PAUSE_BETWEEN_SCENARIOS_SEC)

    except p.error as e:
        print(f"\nPyBullet error caught: {e}")
        print("This was likely caused by closing the render window. Shutting down gracefully.")
    except (Exception, KeyboardInterrupt) as e:
        print(f"\nError or interrupt received: {e}")
        traceback.print_exc()
    finally:
        if env: env.close()
        if RENDER_MODE and pygame.get_init(): pygame.quit()

    if all_results:
        df = pd.DataFrame(all_results)
        params_df = pd.json_normalize(df['params'])
        results_df = df.drop(columns=['params'])
        final_df = pd.concat([results_df, params_df], axis=1)

        print("\n\n" + "="*20 + " FINAL SWEEP SUMMARY (PHASE 3) " + "="*20)
        print(final_df.to_string())
        
        # Save to CSV
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        summary_path = f"tune_phase3_summary_{timestamp}.csv"
        final_df.to_csv(summary_path, index=False, float_format="%.2f")
        print(f"\nFull summary saved to: {summary_path}")

        # Example of finding "good" parameters for Tug of War
        tug_of_war_results = final_df[final_df['scenario_name'] == 'Tug of War'].copy()
        # Good outcome: grapple succeeds and holds for a long time
        successful_holds = tug_of_war_results[tug_of_war_results['final_outcome'] == 'Hold'].copy()
        if not successful_holds.empty:
            successful_holds['duration'] = MAX_STEPS_PER_SCENARIO - successful_holds['grapple_success_step']
            print("\n--- Best Parameters for 'Tug of War' (Longest Hold) ---")
            print(successful_holds.sort_values(by='duration', ascending=False).head(5).to_string())
        else:
            print("\nNo successful 'Hold' outcomes for 'Tug of War'.")

    print("\nPhase 3 tuning script finished.")
