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

# --- Python Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # --- MODIFIED: Import the tuned constants from the environment ---
    from src.env.swarm2denvGNN import (
        Swarm2DEnv, AGENT_RADIUS, BEE_SPEED, 
    )
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    sys.exit(1)

# ==============================================================================
# ===                 TESTING FOCUS & CONTROL CONSTANTS                      ===
# ==============================================================================
MAX_STEPS_PER_SCENARIO = 500
RENDER_MODE = True
STEP_DELAY = 0.015 # Slow down for better visual inspection
PAUSE_BETWEEN_SCENARIOS_SEC = 3.0
SCENARIO_BASE_SEED = random.randint(0, 100000)

# --- MODIFIED: The test now uses the tuned parameters from the env file ---
DEFAULT_ENV_CONFIG = {
    'num_teams': 2, 'num_agents_per_team': 2,
    'num_resources': 1, 'num_obstacles': 0,
    'width': 600, 'height': 400, 'pb_agent_mass': 1.0, 'movement_force_scale': 42.0,
    'pb_agent_linear_damping': 0.9,
    'interaction_force_scale': 2.0,
    'pb_constraint_max_force': 2000.0,
}

# ============================================================================
# ===                         HELPER FUNCTIONS                             ===
# ============================================================================

def render_debug_text(env, scenario_data: dict, grappler: dict, target: dict):
    """Renders real-time debug text onto the Pygame window."""
    if not RENDER_MODE or not hasattr(env, 'font') or not env.font: return
    
    font = env.font
    y_offset = 10
    text_lines = [
        f"Scenario: {scenario_data['scenario_name']}",
        f"Step: {scenario_data['steps_completed']}"
    ]

    # --- Grappler Info ---
    if grappler:
        status = "DEAD" if not grappler.get('alive') else "GRAPPLING" if grappler.get('is_grappling') else "IDLE"
        energy_str = "0.0" if not grappler.get('alive') else f"{grappler['energy']:.1f}"
        
        potential_grip = env._get_effective_grip_strength(grappler)
        grip_val = grappler.get('grapple_last_set_force', 0.0)
        actual_grip_str = f"{grip_val:.1f}" if grappler.get('is_grappling') else "N/A"
        
        # <<< NEW: Add torque and momentum to debug text >>>
        torque_str = f"{grappler.get('applied_torque', 0.0):.1f}"
        momentum_str = f"{grappler.get('grapple_momentum_bonus', 0.0):.1f}"

        if grappler.get('is_grappling'):
            scenario_data["grip_strength_history"].append(grip_val)
        
        text_lines.extend([
            f"--- Grappler (ID {grappler['id']}) ---",
            f"Status: {status}",
            f"Energy: {energy_str}",
            f"Potential Grip (Calc): {potential_grip:.1f}",
            f"Actual Grip (Set Force): {actual_grip_str}",
            f"Momentum Bonus: {momentum_str}",
            f"Applied Torque: {torque_str}"
        ])

    # --- Target Info ---
    if target:
        status = "DEAD" if not target.get('alive') else "GRAPPLED" if target.get('is_grappled') else "FREE"
        speed = np.linalg.norm(target.get('vel')) if target.get('vel') is not None else 0.0
        eff_strength = env._get_effective_combat_strength(target)
        
        # <<< NEW: Add target's counter-torque to debug text >>>
        counter_torque_str = f"{target.get('applied_torque', 0.0):.1f}"

        text_lines.extend([
            f"--- Target (ID {target['id']}) ---",
            f"Status: {status}",
            f"Speed: {speed:.2f}",
            f"Effective Combat Strength: {eff_strength:.2f}",
            f"Applied Counter-Torque: {counter_torque_str}"
        ])
        if target.get('is_grappled'):
            scenario_data["target_speed_history"].append(speed)

    # Render all lines
    for line in text_lines:
        text_surface = font.render(line, True, (255, 255, 255))
        env.screen.blit(text_surface, (10, y_offset))
        y_offset += 18

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def move_towards(agent_pos: np.ndarray, target_pos: np.ndarray, stop_dist: float = 1.0) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    direction = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    if np.linalg.norm(direction) < stop_dist: return np.zeros(2)
    return np.clip(normalize_vector(direction), -1.0, 1.0)

def setup_agent_state(env, agent_idx, pos, **kwargs):
    agent = env.agents[agent_idx]
    agent.update({
        'alive': True, 'pos': np.array(pos), 'vel': np.zeros(2),
        'health': kwargs.get('target_health', agent['max_health']),
        'energy': kwargs.get('target_energy', agent['max_energy']),
        'strength': kwargs.get('target_strength', agent['strength']),
    })
    try:
        p.resetBasePositionAndOrientation(agent['body_id'], [pos[0], pos[1], agent['agent_radius']], [0,0,0,1], physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], [0,0,0], [0,0,0], physicsClientId=env.physicsClient)
    except p.error as e:
        print(f"Warning: PyBullet error resetting agent {agent_idx}: {e}")


# ============================================================================
# ===                         SCENARIO DEFINITIONS                         ===
# ============================================================================

def setup_energy_drain_break(env):
    """Setup: A strong grappler with rapidly draining energy."""
    center_y = env.height / 2
    # Grappler is strong but starts with normal energy
    setup_agent_state(env, 0, [env.width * 0.4, center_y], target_strength=2.0)
    # Target is weaker but will outlast the grappler
    setup_agent_state(env, 1, [env.width * 0.6, center_y], target_strength=1.0)
    return [0, 1]

def control_energy_drain_break(env, active_indices, step):
    """Control: Grappler holds on, losing energy until the grip fails."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    idx_grappler, idx_target = active_indices[0], active_indices[1]
    grappler, target = env.agents[idx_grappler], env.agents[idx_target]
    
    if grappler['alive']:
        # Manually drain grappler's energy to simulate a long struggle
        # Drain faster after the grapple is established to speed up the test
        if grappler.get('is_grappling'):
            grappler['energy'] = max(0, grappler['energy'] - 1.0)
        
        if not grappler.get('is_grappling'):
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.5)
            actions[idx_grappler]['pickup'] = 1
        else:
            # Just hold on, don't move
            actions[idx_grappler]['movement'] = np.zeros(2)
            actions[idx_grappler]['pickup'] = 0

    if target['alive']:
        # Target consistently pulls away, testing the weakening grip
        actions[idx_target]['movement'] = np.array([1.0, 0.0])
        # If grappled, also try to break free with the 'release' action
        if target.get('is_grappled'):
            actions[idx_target]['pickup'] = 2
        
    return actions


def setup_torque_struggle(env):
    """Setup: Two agents of equal strength to test the torque mechanic."""
    center_y = env.height / 2
    setup_agent_state(env, 0, [env.width * 0.4, center_y], target_strength=1.5)
    setup_agent_state(env, 1, [env.width * 0.6, center_y], target_strength=1.5)
    return [0, 1]

def control_torque_struggle(env, active_indices, step):
    """Control: Grappler tries to spin, target tries to counter-spin."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    idx_grappler, idx_target = active_indices[0], active_indices[1]
    grappler, target = env.agents[idx_grappler], env.agents[idx_target]

    if grappler['alive']:
        if not grappler.get('is_grappling'):
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS*1.5)
            actions[idx_grappler]['pickup'] = 1
        else:
            # Grappler applies max positive torque (spin counter-clockwise)
            actions[idx_grappler]['movement'] = np.array([1.0, 0.0]) # Use x-axis for torque
    
    if target['alive'] and target.get('is_grappled'):
        # Target applies max negative torque to resist
        actions[idx_target]['movement'] = np.array([-1.0, 0.0])

    return actions

def setup_momentum_grapple(env):
    """Setup: A stationary grappler and a fast-moving target."""
    center_y = env.height / 2
    setup_agent_state(env, 0, [env.width * 0.5, center_y]) # Stationary grappler
    setup_agent_state(env, 1, [20, center_y]) # Target starts at the left edge
    # Give the target an initial high velocity
    p.resetBaseVelocity(env.agents[1]['body_id'], [BEE_SPEED * 1.5, 0, 0], [0,0,0], physicsClientId=env.physicsClient)
    return [0, 1]


def control_momentum_grapple(env, active_indices, step):
    """Control: Grappler tries to time the grapple as the target flies by."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    idx_grappler, idx_target = active_indices[0], active_indices[1]
    grappler, target = env.agents[idx_grappler], env.agents[idx_target]
    
    # Grappler waits until the target is very close, then attempts to grapple
    # It will keep trying until it succeeds or the target is too far away.
    if grappler['alive'] and not grappler.get('is_grappling'):
        dist = np.linalg.norm(np.array(target['pos']) - np.array(grappler['pos']))
        # A wider attempt window
        if dist < AGENT_RADIUS * 8 and grappler['pos'][0] < target['pos'][0]:
            actions[idx_grappler]['pickup'] = 1

    # Target no longer moves on its own; its initial velocity carries it.
    # This makes the test more deterministic.
    
    return actions

def setup_tug_of_war(env):
    center_y = env.height / 2
    setup_agent_state(env, 0, [env.width * 0.4, center_y], target_strength=1.2)
    setup_agent_state(env, 1, [env.width * 0.6, center_y], target_strength=1.3)
    return [0, 1]

def control_tug_of_war(env, active_indices, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    idx_grappler, idx_target = active_indices[0], active_indices[1]
    grappler, target = env.agents[idx_grappler], env.agents[idx_target]
    grappler_anchor, target_anchor = [50, env.height/2], [env.width-50, env.height/2]
    if grappler['alive']:
        if not grappler.get('is_grappling'):
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.5)
            actions[idx_grappler]['pickup'] = 1
        else:
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], grappler_anchor, stop_dist=0)
            actions[idx_grappler]['pickup'] = 0
    if target['alive']:
        actions[idx_target]['movement'] = move_towards(target['pos'], target_anchor, stop_dist=0)
    return actions

def setup_dominant_grapple(env):
    center_y = env.height / 2
    setup_agent_state(env, 0, [env.width * 0.4, center_y], target_strength=2.0)
    setup_agent_state(env, 1, [env.width * 0.6, center_y], target_strength=0.7)
    return [0, 1]

def control_dominant_grapple(env, active_indices, step):
    return control_tug_of_war(env, active_indices, step)

def setup_release_test(env):
    return setup_dominant_grapple(env)

def control_release_test(env, active_indices, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    idx_grappler, idx_target = active_indices[0], active_indices[1]
    grappler, target = env.agents[idx_grappler], env.agents[idx_target]
    if grappler['alive']:
        if not grappler.get('is_grappling'):
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS*1.5)
            actions[idx_grappler]['pickup'] = 1
        elif step < 200:
            actions[idx_grappler]['movement'] = move_towards(grappler['pos'], [50, env.height/2], stop_dist=0)
            actions[idx_grappler]['pickup'] = 0
        else:
            actions[idx_grappler]['pickup'] = 2
    if target['alive']:
        actions[idx_target]['movement'] = move_towards(target['pos'], [env.width - 50, env.height/2], stop_dist=0)
    return actions

# ============================================================================
# ===                     CORE SIMULATION RUNNER                           ===
# ============================================================================
def run_scenario(scenario_def: dict):
    env = None
    try:
        name, setup_fn, control_fn, num_agents, params_override = scenario_def.values()
        print(f"\n{'='*15} Starting Scenario: {name} {'='*15}")
        
        current_env_params = {**DEFAULT_ENV_CONFIG, **(params_override or {})}
        num_teams = current_env_params.get('num_teams', 2)
        agents_per_team = int(np.ceil(num_agents / num_teams)) if num_teams > 0 else num_agents
        current_env_params['num_agents_per_team'] = agents_per_team

        env = Swarm2DEnv(**current_env_params, max_steps=MAX_STEPS_PER_SCENARIO + 50, render_mode=RENDER_MODE)
        env.reset(seed=SCENARIO_BASE_SEED + sum(ord(c) for c in name))
        
        active_indices = setup_fn(env)
        scenario_data = defaultdict(lambda: -1, {
            "scenario_name": name, "steps_completed": 0, "agents_involved_ids": [env.agents[i]['id'] for i in active_indices],
            "grip_strength_history": [], "target_speed_history": []
        })
        
        grappler_obj, target_obj = env.agents[active_indices[0]], env.agents[active_indices[1]]

        for step in range(MAX_STEPS_PER_SCENARIO):
            if RENDER_MODE:
                env.render()
                render_debug_text(env, scenario_data, grappler_obj, target_obj)
                pygame.display.flip()
                time.sleep(STEP_DELAY)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: raise KeyboardInterrupt

            actions = control_fn(env, active_indices, step)
            _, _, terminated, truncated, _ = env.step(actions)
            
            scenario_data["steps_completed"] += 1
            if scenario_data["grapple_success_step"] == -1 and grappler_obj.get('is_grappling'):
                scenario_data["grapple_success_step"] = step
            if scenario_data["grapple_success_step"] != -1 and scenario_data["grapple_break_step"] == -1 and not grappler_obj.get('is_grappling'):
                scenario_data["grapple_break_step"] = step
            if terminated or truncated: break
            
        print(f"  --- Scenario Summary: {name} ---")
        summary_data = scenario_data.copy() # Start with the base data

        if summary_data["grapple_success_step"] != -1:
            break_step = summary_data['grapple_break_step']
            break_str = f"Step {break_step}" if break_step != -1 else "Held until end"
            duration = (break_step - summary_data['grapple_success_step']) if break_step != -1 else (summary_data['steps_completed'] - summary_data['grapple_success_step'])
            
            # <<< NEW: Calculate summary stats for CSV >>>
            grip_hist = summary_data.pop('grip_strength_history', []) # Remove list from dict
            speed_hist = summary_data.pop('target_speed_history', []) # Remove list from dict
            
            summary_data['avg_grip'] = np.mean(grip_hist) if grip_hist else 0
            summary_data['min_grip'] = np.min(grip_hist) if grip_hist else 0
            summary_data['max_grip'] = np.max(grip_hist) if grip_hist else 0
            summary_data['avg_target_speed'] = np.mean(speed_hist) if speed_hist else 0
            summary_data['grapple_duration'] = duration
            
            print(f"  Grapple Initiated: Step {summary_data['grapple_success_step']}")
            print(f"  Grapple Broken/Released: {break_str} (Duration: {duration} steps)")
            print(f"  Avg. Grip Strength (maxForce): {summary_data['avg_grip']:.2f}")
            print(f"  Avg. Target Speed While Grappled: {summary_data['avg_target_speed']:.2f}")
        else:
            print("  Grapple was not successfully initiated.")
        
        # summary_data now contains only scalar values, safe for DataFrame
        return summary_data, env
        print(f"  --- Scenario Summary: {name} ---")
        if scenario_data["grapple_success_step"] != -1:
            break_step = scenario_data['grapple_break_step']
            break_str = f"Step {break_step}" if break_step != -1 else "Held until end"
            duration = (break_step - scenario_data['grapple_success_step']) if break_step != -1 else (scenario_data['steps_completed'] - scenario_data['grapple_success_step'])
            avg_grip = np.mean(scenario_data['grip_strength_history']) if scenario_data['grip_strength_history'] else 0
            avg_target_speed = np.mean(scenario_data['target_speed_history']) if scenario_data['target_speed_history'] else 0
            print(f"  Grapple Initiated: Step {scenario_data['grapple_success_step']}")
            print(f"  Grapple Broken/Released: {break_str} (Duration: {duration} steps)")
            print(f"  Avg. Grip Strength (maxForce): {avg_grip:.2f}")
            print(f"  Avg. Target Speed While Grappled: {avg_target_speed:.2f}")
        else:
            print("  Grapple was not successfully initiated.")
        return scenario_data, env
    except Exception as e:
        print(f"!!!!!! An error occurred in scenario '{name}': {e} !!!!!!")
        traceback.print_exc()
        return None, env

# ============================================================================
# ===                         MAIN EXECUTION BLOCK                         ===
# ============================================================================

if __name__ == "__main__":
    all_scenario_results = []
    
    # --- EXPANDED SCENARIO SUITE ---
    scenarios_to_run = [
        {
            "name": "Struggle & Break (Tug-of-War)", 
            "setup_fn": setup_tug_of_war, 
            "control_fn": control_tug_of_war, 
            "num_agents": 2, "params_override": None
        },
        {
            "name": "Dominant Grapple (Strong vs Weak)", 
            "setup_fn": setup_dominant_grapple, 
            "control_fn": control_dominant_grapple, 
            "num_agents": 2, "params_override": None
        },
        {
            "name": "Grapple & Manual Release Test", 
            "setup_fn": setup_release_test, 
            "control_fn": control_release_test, 
            "num_agents": 2, "params_override": None
        },
        {
            "name": "NEW: Energy Drain Break Test", 
            "setup_fn": setup_energy_drain_break, 
            "control_fn": control_energy_drain_break, 
            "num_agents": 2, "params_override": None
        },
        {
            "name": "NEW: Torque Struggle Test", 
            "setup_fn": setup_torque_struggle, 
            "control_fn": control_torque_struggle, 
            "num_agents": 2, "params_override": None
        },
        {
            "name": "NEW: Momentum Bonus Grapple Test", 
            "setup_fn": setup_momentum_grapple, 
            "control_fn": control_momentum_grapple, 
            "num_agents": 2, "params_override": None
        },
    ]

    # --- The rest of the execution logic is correct and robust ---
    try:
        if RENDER_MODE: pygame.init()
        for scenario_def in scenarios_to_run:
            result, env_instance = run_scenario(scenario_def)
            if env_instance: env_instance.close()
            if result is None: 
                print("!!!!!! Scenario failed to run, stopping tests. !!!!!!")
                break
            all_scenario_results.append(result)
            if RENDER_MODE and 'pygame' in sys.modules and pygame.get_init():
                print(f"Pausing for {PAUSE_BETWEEN_SCENARIOS_SEC}s...")
                time.sleep(PAUSE_BETWEEN_SCENARIOS_SEC)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    finally:
        if RENDER_MODE and 'pygame' in sys.modules and pygame.get_init():
            pygame.quit()


    if all_scenario_results:
        # Create a DataFrame from the results, handling potential missing keys
        df = pd.DataFrame.from_records(all_scenario_results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        summary_path = f"grapple_test_summary_{timestamp}.csv"
        df.to_csv(summary_path, index=False, float_format="%.2f")
        print(f"\nGrapple test results saved to: {summary_path}")

    print("\nTest Script finished.")