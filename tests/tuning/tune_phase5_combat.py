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
from collections import defaultdict
import math

# --- Python Path (ensure Swarm2DEnv is found) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import AGENT_RADIUS, COMBAT_RADIUS, BEE_SPEED
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# ===                      TUNING & CONTROL CONSTANTS                        ===
# ==============================================================================
# --- Master Switches ---
RENDER_SIMULATIONS = True
VERBOSE_DEBUG = True # Master switch for detailed step-by-step console logs

# --- Timing ---
MAX_STEPS_PER_SCENARIO = 150 # Reduced for faster, more decisive combat
STEP_DELAY = 0.01
PAUSE_BETWEEN_SCENARIOS_SEC = 1.0

# --- Debugging ---
DEBUG_PRINT_FREQ = 25 # How often to print debug info for non-critical loops

# ==============================================================================
# ===                         PARAMETER SWEEP                                ===
# ==============================================================================

# FOCUSED PARAMETER SWEEP - STAGE 3: TUNING THE COUNTER-GRAPPLE
# GOAL: Balance the new counter-grip mechanic. A struggling target should be able to
# significantly shorten the grapple's duration, but not break free instantly.
parameter_sweep = {
    # STEP 1: HOW EFFECTIVE IS COUNTERING?
    # This is now our primary variable. Higher values make escaping easier.
    'grappled_agent_counter_grip_scale': [0.3, 0.5], 
    
    # STEP 2: RE-TUNE FATIGUE AROUND THE COUNTER MECHANIC
    # With an active counter, fatigue can be lower, as it's no longer the ONLY escape vector.
    # We're still aiming for a ~75-100 step duration IN A FIGHT, not from passive waiting.
    'grapple_fatigue_rate': [0.02, 0.035], 
    
    # --- DAMAGE REMAINS LOW for this tuning stage ---
    # We must ensure the duration feels right before we re-introduce lethality.
    'grapple_crush_damage_rate': [1.0], 
    'grapple_struggle_damage_rate': [0.5],

    # --- KEPT CONSTANT ---
    'pb_agent_constraint_max_force': [10000],
    'grapple_torque_escape_strength': [0.6], 
    'grapple_momentum_bonus_scale': [0.1], 
    'agent_interaction_force_scale': [0.35],
    'grapple_momentum_decay': [0.95],
    'grapple_crit_chance': [0.05],
    'grapple_crit_multiplier': [3.0],
    'grapple_rear_crit_bonus_multiplier': [2.5],
}

# --- Base Environment Config (assumed from previous phases) ---
# These are the best values from previous tuning phases
base_env_config = {
    'movement_force_scale': 15.0, 'pb_agent_linear_damping': 0.11,
    'pb_agent_lateral_friction': 0.5, 'pb_agent_angular_damping': 0.4,
    'resource_base_mass': 0.075, 'resource_interaction_force_scale': 1.2,
    'pb_resource_constraint_max_force': 3000, 'pb_res_friction_dynamic': 0.25,
    'pb_res_damping_dynamic': 0.25, 'pb_agent_constraint_max_force': 3000, 'bee_speed': 200.0,
    'num_teams': 2, 'num_agents_per_team': 8, # Max needed for scenarios
    'num_resources': 5, 'num_obstacles': 2,
    'render_mode': RENDER_SIMULATIONS,
}

# ==============================================================================
# ===                      AGENT POLICIES & HELPERS                          ===
# ==============================================================================

def get_agent_status(agent, grappler_map):
    """Gets a string representation of the agent's current high-level status."""
    if agent.get('is_grappling'):
        return "GRAPPLING"
    elif agent.get('is_grappled'):
        return "GRAPPLED"
    elif agent.get('has_resource'):
        return "CARRYING"
    else:
        return "IDLE/MOVING"

def print_debug_to_console(env, agent_indices):
    """Prints a detailed debug status for a list of specified agent indices."""
    if not agent_indices:
        return

    grappler_map = {a.get('grappled_agent_id'): a for a in env.agents if a and a.get('is_grappling')}
    
    header = f"--- Agents: {', '.join(map(str, agent_indices))} ---"
    print("-" * (len(header) + 2))

    for agent_idx in agent_indices:
        agent = env.agents[agent_idx]
        if not agent or not agent.get('alive'):
            print(f"  Agent {agent_idx}: DEAD")
            continue

        status = get_agent_status(agent, grappler_map)
        grappler = grappler_map.get(agent_idx) if status == "GRAPPLED" else None
        
        # Base agent info
        print(f"  Agent {agent_idx} | Status: {status:<12} | Pos: ({agent['pos'][0]:.1f}, {agent['pos'][1]:.1f}) | Vel: ({agent['vel'][0]:.1f}, {agent['vel'][1]:.1f})")
        print(f"    - Stats: Health: {agent.get('health', 0):.1f} | Energy: {agent.get('energy', 0):.1f} | Speed: {np.linalg.norm(agent['vel']):.2f} | Mass: {agent.get('mass', 0):.2f}")

        # Combat / Grapple specific info
        damage_dealt = agent.get('damage_dealt_this_step', 0.0)
        applied_mov_force = agent.get('debug_applied_force', 0.0)
        torque = agent.get('debug_torque', 0.0)

        if status == "GRAPPLING":
            # For the grappler, show their side of the equation
            target_agent = env.agents[agent.get('grappled_agent_id')]
            grip_components = env.physics_manager._get_grip_strength_components(agent)
            momentum_bonus = agent.get('grapple_momentum_bonus', 0.0)
            
            grappler_potential_grip = grip_components.get('final_grip', 0.0)
            total_grappler_force = grappler_potential_grip + momentum_bonus
            
            # Show the target's counter-force to calculate the net grip
            target_action = env.actions[target_agent['id']] if hasattr(env, 'actions') and env.actions else {}
            target_counter_grip = 0.0
            if target_action.get("pickup") == 2: # If target is actively breaking free
                target_grip_strength = env.physics_manager._get_effective_grip_strength(target_agent)
                target_counter_grip = target_grip_strength * env.grappled_agent_counter_grip_scale

            net_grip_force = total_grappler_force - target_counter_grip
            actual_constraint_force = agent.get('grapple_last_set_force', net_grip_force) # Use last set force for accuracy
            
            print(f"    - Grip Tug-of-War: Your Force ({total_grappler_force:.1f}) - Target Counter ({target_counter_grip:.1f}) = Net Force ({net_grip_force:.1f})")
            print(f"    - Constraint Force (Potential/Actual): {net_grip_force:.1f} / {actual_constraint_force:.1f}")
            print(f"    - Factors: Base: {grip_components.get('base_grip', 0):.1f} | E: {grip_components.get('energy_factor', 0):.2f}x | HP: {grip_components.get('health_factor', 0):.2f}x")
            print(f"    - Bonuses: Momentum: {momentum_bonus:.2f} | Penalties: Fatigue: {grip_components.get('fatigue_factor', 1.0):.2f}x | Torque: {grip_components.get('torque_penalty', 1.0):.2f}x")
            print(f"    - Damage Dealt (Crush): {damage_dealt:.2f}")

        elif status == "GRAPPLED" and grappler:
            # For the target, show the same tug-of-war from their perspective
            grappler_grip_comps = env.physics_manager._get_grip_strength_components(grappler)
            grappler_potential_grip = grappler_grip_comps.get('final_grip', 0.0)
            grappler_momentum_bonus = grappler.get('grapple_momentum_bonus', 0.0)
            total_grappler_force = grappler_potential_grip + grappler_momentum_bonus

            my_counter_grip = 0.0
            agent_action = env.actions[agent['id']] if hasattr(env, 'actions') and env.actions else {}
            if agent_action.get("pickup") == 2: # If I am actively breaking free
                my_grip_strength = env.physics_manager._get_effective_grip_strength(agent)
                my_counter_grip = my_grip_strength * env.grappled_agent_counter_grip_scale
            
            net_grip_force = total_grappler_force - my_counter_grip
            actual_constraint_force = grappler.get('grapple_last_set_force', net_grip_force)

            print(f"    - Grip Tug-of-War: Grappler Force ({total_grappler_force:.1f}) - Your Counter ({my_counter_grip:.1f}) = Net Force ({net_grip_force:.1f})")
            print(f"    - Constraint Force (Potential/Actual): {net_grip_force:.1f} / {actual_constraint_force:.1f}")
            print(f"    - Damage Dealt (Struggle): {damage_dealt:.2f}")
        
        print(f"    - Forces: Movement (desired): {applied_mov_force:.1f} | Torque: {torque:.2f}")

    print("-" * (len(header) + 2))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalizes a numpy vector, handling the zero-vector case."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def get_nearest_entity(env, agent_idx, entity_list):
    """Generic helper to find the nearest entity (agent or resource)."""
    agent = env.agents[agent_idx]
    if not agent['alive'] or agent.get('pos') is None: return None
    
    nearest_entity, min_dist_sq = None, float('inf')
    
    for entity in entity_list:
        # Duck-typing check for required keys
        if not entity or not entity.get('alive', True) or entity.get('pos') is None:
            continue
        # Skip self for agent checks
        if 'id' in entity and entity['id'] == agent['id']:
            continue
        # Skip dead agents if entity_list is agents
        if 'team' in entity and entity['team'] == agent['team']:
            continue
        
        dist_sq = np.sum((agent['pos'] - entity['pos'])**2)
        if dist_sq < min_dist_sq:
            min_dist_sq, nearest_entity = dist_sq, entity
            
    return nearest_entity

def get_nearest_enemy(env, agent_idx):
    """Finds the nearest alive enemy agent."""
    return get_nearest_entity(env, agent_idx, [a for a in env.agents if a['team'] != env.agents[agent_idx]['team']])


# --- Simple Action Policies (from combat_test.py) ---
def policy_idle(env, agent_idx, target): return {'movement': np.zeros(2), 'pickup': 0}

def policy_chase_target(env, agent_idx, target):
    agent = env.agents[agent_idx]
    if not agent['alive']:
        return policy_idle(env, agent_idx, target)
    
    # --- NEW: Reflexive escape behavior ---
    if agent.get('is_grappled'):
        # If grappled, prioritize breaking free.
        return {'movement': np.array([1.0, 0.0]), 'pickup': 2}

    if target is None or not target['alive']:
        return policy_idle(env, agent_idx, target)

    direction = target['pos'] - agent['pos']
    dist = np.linalg.norm(direction)
    pickup_action = 1 if dist < (COMBAT_RADIUS * 1.0) else 0 # More aggressive pickup
    movement = normalize_vector(direction)
    return {'movement': movement, 'pickup': pickup_action}

def policy_flee_from_target(env, agent_idx, target):
    agent = env.agents[agent_idx]
    if not agent['alive']:
        return policy_idle(env, agent_idx, target)

    # --- NEW: Reflexive escape behavior ---
    if agent.get('is_grappled'):
        return {'movement': np.array([1.0, 0.0]), 'pickup': 2}

    if target is None:
        return policy_idle(env, agent_idx, target)
        
    flee_direction = agent['pos'] - target['pos']
    movement = normalize_vector(flee_direction)
    return {'movement': movement, 'pickup': 0}

def policy_grapple_and_spin(env, agent_idx, target):
    """Chases, attempts to grapple, and then spins to apply torque."""
    agent = env.agents[agent_idx]
    if agent.get('is_grappling'):
        # Already grappling, so apply max torque
        return {'movement': np.array([1.0, 0.0]), 'pickup': 0}
    else:
        # Not grappling, chase the target to initiate
        return policy_chase_target(env, agent_idx, target)

def policy_try_to_break_grapple(env, agent_idx, target):
    """If grappled, attempts to break free by rotating and using the release action."""
    agent = env.agents[agent_idx]
    if agent.get('is_grappled'):
        # Counter-rotate and spam release
        return {'movement': np.array([-1.0, 0.0]), 'pickup': 2}
    else:
        # Not grappled, just stay idle or move slightly
        return policy_idle(env, agent_idx, target)


# --- Scripted Grapple Scenario Functions (from combat_testWgrapple.py) ---

def setup_agent_state(env, agent_idx, pos, **kwargs):
    agent = env.agents[agent_idx]
    vel = kwargs.get('vel', np.zeros(3)) # Default to 3D zero vector for pybullet
    
    agent.update({
        'alive': True, 'pos': np.array(pos), 'vel': np.array(vel[:2]), # Store 2D velocity
        'health': kwargs.get('health', agent['max_health']),
        'energy': kwargs.get('energy', agent['max_energy']),
        'strength': kwargs.get('strength', agent['strength']),
        'team': kwargs.get('team', agent.get('team', 0)),
    })
    try:
        p.resetBasePositionAndOrientation(agent['body_id'], [pos[0], pos[1], agent['agent_radius']], [0,0,0,1], physicsClientId=env.physicsClient)
        # Use the provided velocity, ensuring it's a 3D vector for pybullet
        p.resetBaseVelocity(agent['body_id'], [vel[0], vel[1], 0], [0,0,0], physicsClientId=env.physicsClient)
    except p.error as e:
        print(f"Warning: PyBullet error resetting agent {agent_idx}: {e}")

def move_towards(agent_pos: np.ndarray, target_pos: np.ndarray, stop_dist: float = 1.0) -> np.ndarray:
    if agent_pos is None or target_pos is None: return np.zeros(2)
    direction = np.array(target_pos[:2]) - np.array(agent_pos[:2])
    if np.linalg.norm(direction) < stop_dist: return np.zeros(2)
    return np.clip(normalize_vector(direction), -1.0, 1.0)

# Scenario: Tug of War
def setup_tug_of_war(env):
    """Setup: Four pairs of agents grapple and pull apart."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], strength=1.5, team=0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], strength=1.5, team=1)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_tug_of_war(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_g, idx_t = pair[0], pair[1]
        grappler, target = env.agents[idx_g], env.agents[idx_t]
        
        # Use the midpoint Y to ensure a straight pull
        center_y = (grappler['pos'][1] + target['pos'][1]) / 2.0
        g_anchor, t_anchor = [50, center_y], [env.width - 50, center_y]
        
        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_g]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.5)
                actions[idx_g]['pickup'] = 1
            else:
                actions[idx_g]['movement'] = move_towards(grappler['pos'], g_anchor, stop_dist=0)
        
        if target['alive']:
            actions[idx_t]['movement'] = move_towards(target['pos'], t_anchor, stop_dist=0)
    return actions

# Scenario: Torque Struggle
def setup_torque_struggle(env):
    """Setup: Four pairs of agents to test torque mechanics."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], strength=1.5, team=0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], strength=1.5, team=1)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_torque_struggle(env, active_pairs, step):
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_g, idx_t = pair[0], pair[1]
        grappler, target = env.agents[idx_g], env.agents[idx_t]

        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_g]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS*1.5)
                actions[idx_g]['pickup'] = 1
            else:
                actions[idx_g]['movement'] = np.array([0.0, 1.0]) # Spin
        
        if target['alive'] and target.get('is_grappled'):
            actions[idx_t]['movement'] = np.array([0.0, -1.0]) # Resist
    return actions

# Scenario: Momentum Grapple
def setup_momentum_grapple(env):
    """Setup: Four pairs of stationary grapplers and fast-moving targets."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.5, center_y], team=0)
        setup_agent_state(env, target_idx, [20, center_y], vel=[BEE_SPEED * 1.5, 0, 0], team=1)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_momentum_grapple(env, active_pairs, step):
    """Control: Grapplers try to time the grapple as targets fly by."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_g, idx_t = pair[0], pair[1]
        grappler, target = env.agents[idx_g], env.agents[idx_t]
        
        if grappler['alive'] and not grappler.get('is_grappling'):
            dist_sq = np.sum((target['pos'] - grappler['pos'])**2)
            # CORRECTED LOGIC: Attempt grapple when target is close and approaching, not after it has passed.
            if dist_sq < (AGENT_RADIUS * 10)**2 and target['pos'][0] < grappler['pos'][0]:
                actions[idx_g]['pickup'] = 1
    return actions

# Scenario: Energy Drain Break
def setup_energy_drain_break(env):
    """Setup: Four pairs of strong grapplers with draining energy."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], strength=2.0, team=0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], strength=1.0, team=1)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_energy_drain_break(env, active_pairs, step):
    """Control: Grapplers hold on, losing energy until the grip fails."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_g, idx_t = pair[0], pair[1]
        grappler, target = env.agents[idx_g], env.agents[idx_t]
        
        if grappler['alive']:
            if grappler.get('is_grappling'):
                grappler['energy'] = max(0, grappler['energy'] - 2.5) # Faster drain
            
            if not grappler.get('is_grappling'):
                actions[idx_g]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.5)
                actions[idx_g]['pickup'] = 1
            else:
                actions[idx_g]['movement'] = np.zeros(2) # Just hold
        
        if target['alive']:
            actions[idx_t]['movement'] = np.array([1.0, 0.0])
            if target.get('is_grappled'):
                actions[idx_t]['pickup'] = 2
    return actions

# Scenario: Active Breakout
def setup_active_breakout(env):
    """Setup: A standard grapple where the target actively tries to break free."""
    active_indices = []
    for i in range(4):
        offset_y = (i - 1.5) * 150
        center_y = env.height / 2 + offset_y
        grappler_idx, target_idx = i, i + 4
        setup_agent_state(env, grappler_idx, [env.width * 0.4, center_y], strength=1.5, team=0)
        setup_agent_state(env, target_idx, [env.width * 0.6, center_y], strength=1.5, team=1)
        active_indices.append([grappler_idx, target_idx])
    return active_indices

def control_active_breakout(env, active_pairs, step):
    """Control: Target uses the 'pickup: 2' action to try and break the grapple."""
    actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
    for pair in active_pairs:
        idx_g, idx_t = pair[0], pair[1]
        grappler, target = env.agents[idx_g], env.agents[idx_t]

        if grappler['alive']:
            if not grappler.get('is_grappling'):
                actions[idx_g]['movement'] = move_towards(grappler['pos'], target['pos'], stop_dist=AGENT_RADIUS * 1.0)
                actions[idx_g]['pickup'] = 1
            else:
                actions[idx_g]['movement'] = np.array([-1.0, 0.0]) # Pull

        if target['alive']:
            actions[idx_t]['movement'] = np.array([1.0, 0.0]) # Pull back
            if target.get('is_grappled'):
                actions[idx_t]['pickup'] = 2 # Attempt to break free

    return actions


# ==============================================================================
# ===                         SCENARIO RUNNERS                               ===
# ==============================================================================

def run_policy_based_scenario(name, env_params, policy_map, setup_fn, steps=150, debug_interval=25):
    """
    Sets up and runs a single scenario based on agent policies.
    Returns a dictionary of aggregated results.
    """
    print(f"    - Running Policy Scenario: {name}")
    env = Swarm2DEnv(**env_params, max_steps=steps + 50)
    
    # --- Setup ---
    agent_configs = setup_fn(env)
    active_indices = []
    
    # Deactivate all agents first
    for agent in env.agents:
        agent['alive'] = False
        p.resetBasePositionAndOrientation(agent['body_id'], [10000, 10000, -100], [0,0,0,1], physicsClientId=env.physicsClient)

    # Configure the active agents
    for config in agent_configs:
        agent_idx = config['id']
        active_indices.append(agent_idx)
        setup_agent_state(env, agent_idx, config['pos'], team=config['team'])
        env.agents[agent_idx]['policy'] = config['policy']
        env.agents[agent_idx]['target_id'] = config['target_id']
        # env.agents[agent_idx]['team'] = config['team'] # setup_agent_state handles this

    initial_health = {i: env.agents[i]['health'] for i in active_indices}
    
    # --- Construct pairs for debugging ---
    debug_pairs = []
    for config in agent_configs:
        if config.get('target_id') is not None:
            # Create a sorted pair to avoid duplicates like [0,1] and [1,0]
            pair = sorted([config['id'], config['target_id']])
            if pair not in debug_pairs:
                debug_pairs.append(pair)

    # --- Run Simulation ---
    for step in range(steps):
        actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]

        # Add debug info to agent dicts before policies are called
        for i in range(len(env.agents)):
            mov_vec = actions[i].get('movement', np.zeros(2))
            force_magnitude = np.linalg.norm(mov_vec) * env_params['movement_force_scale'] * 50
            env.agents[i]['debug_applied_force'] = force_magnitude
            env.agents[i]['damage_dealt_this_step'] = 0.0 # Reset damage counter

        for agent_idx in active_indices:
            agent = env.agents[agent_idx]
            if not agent['alive']: continue
            
            # Allow for scenarios where there's no specific target
            target_agent = None
            if agent.get('target_id') is not None:
                target_agent = env.agents[agent['target_id']]

            actions[agent_idx] = agent['policy'](env, agent_idx, target_agent)

        _, _, terminated, truncated, _ = env.step(actions)
        
        # --- DEBUG: Print agent states periodically ---
        if debug_interval > 0 and (step % debug_interval == 0 or step == steps - 1):
            print(f"\n--- Step: {step:<4} | Scenario: {name} ---")
            
            # Create a flat list of all unique agent indices involved in the scenario
            all_involved_indices = sorted(list(set(idx for pair in debug_pairs for idx in pair)))
            print_debug_to_console(env, all_involved_indices)

        if RENDER_SIMULATIONS: env.render(); time.sleep(STEP_DELAY)
        if terminated or truncated: break

    # --- Collect Results ---
    final_health = {i: env.agents[i]['health'] for i in active_indices}
    total_damage = {i: initial_health[i] - final_health[i] for i in active_indices}
    
    team_0_alive = sum(1 for i in active_indices if env.agents[i]['team'] == 0 and env.agents[i]['alive'])
    team_1_alive = sum(1 for i in active_indices if env.agents[i]['team'] == 1 and env.agents[i]['alive'])
    
    winner = "Draw"
    if team_0_alive > 0 and team_1_alive == 0: winner = "Team0"
    elif team_1_alive > 0 and team_0_alive == 0: winner = "Team1"

    result_key_prefix = f"policy_{name}"
    return {
        f"{result_key_prefix}_winner": winner,
        f"{result_key_prefix}_team0_alive": team_0_alive,
        f"{result_key_prefix}_team1_alive": team_1_alive,
    }


def run_scripted_grapple_scenario(env, scenario_def, seed):
    """
    Sets up and runs a single scripted grapple scenario.
    Returns a dictionary of aggregated results.
    """
    name = scenario_def['name']
    print(f"    - Running Grapple Scenario: {name}")
    env.reset(seed=seed)

    # --- Setup ---
    active_pairs = scenario_def['setup_fn'](env)
    
    # Initialize results for each pair
    pair_results = [
        {
            "pair_num": i + 1, "scenario_name": name,
            "grapple_success_step": -1, "grapple_break_step": -1,
            "final_outcome": "Timeout", "details": []
        } for i in range(len(active_pairs))
    ]

    # --- Run Simulation ---
    for step in range(MAX_STEPS_PER_SCENARIO):
        actions = scenario_def['control_fn'](env, active_pairs, step)
        
        # Add debug info to agent dicts
        for i in range(len(env.agents)):
            mov_vec = actions[i].get('movement', np.zeros(2))
            force_magnitude = np.linalg.norm(mov_vec) * env.movement_force_scale * 50
            env.agents[i]['debug_applied_force'] = force_magnitude
            env.agents[i]['damage_dealt_this_step'] = 0.0 # Reset damage counter

        _, _, terminated, truncated, infos = env.step(actions)

        # Update and log state for each pair
        for i, pair in enumerate(active_pairs):
            grappler, target = env.agents[pair[0]], env.agents[pair[1]]
            res = pair_results[i]

            # Log detailed physics state at this step
            log_entry = {
                'step': step,
                'grappler_health': grappler['health'], 'grappler_energy': grappler['energy'],
                'grappler_is_grappling': grappler.get('is_grappling', False),
                'target_health': target['health'], 'target_energy': target['energy'],
                'target_is_grappled': target.get('is_grappled', False),
                'applied_grip_force': grappler.get('grapple_last_set_force', 0.0),
                'momentum_bonus': grappler.get('grapple_momentum_bonus', 0.0),
                'applied_torque': grappler.get('applied_torque', 0.0),
                'target_counter_torque': target.get('applied_torque', 0.0),
                'damage_dealt': grappler.get('damage_dealt_this_step', 0.0)
            }
            res['details'].append(log_entry)

            if res["grapple_success_step"] == -1 and grappler.get('is_grappling'):
                res["grapple_success_step"] = step
            if res["grapple_success_step"] != -1 and res["grapple_break_step"] == -1 and not grappler.get('is_grappling'):
                res["grapple_break_step"] = step
                if res['final_outcome'] == 'Timeout': res['final_outcome'] = "GripBreak"
            
            if not target['alive'] and res['final_outcome'] == 'Timeout':
                res['final_outcome'] = 'TargetKilled'
            if not grappler['alive'] and res['final_outcome'] == 'Timeout':
                res['final_outcome'] = 'GrapplerKilled'

        if VERBOSE_DEBUG and (step % DEBUG_PRINT_FREQ == 0 or step == MAX_STEPS_PER_SCENARIO - 1):
            print_debug_to_console(env, [pair[0] for pair in active_pairs] + [pair[1] for pair in active_pairs])

        if RENDER_SIMULATIONS: 
            env.render()
            time.sleep(STEP_DELAY)
        
        if terminated or truncated or all(r['final_outcome'] != 'Timeout' for r in pair_results):
            break

    # --- Collect Aggregate Results ---
    final_results = {}
    print(f"      Scenario '{name}' Summary:")
    for i, res in enumerate(pair_results):
        # Final outcome check
        if res["final_outcome"] == "Timeout" and res["grapple_success_step"] != -1:
            res["final_outcome"] = "Hold"
            
        duration = 0
        if res['grapple_success_step'] != -1:
            end_step = res['grapple_break_step'] if res['grapple_break_step'] != -1 else step
            duration = end_step - res['grapple_success_step']

        print(f"        - Pair {i+1}: Outcome: {res['final_outcome']:<15} | Duration: {duration:<4} steps")
        
        avg_grip = np.mean([s['applied_grip_force'] for s in res['details'] if s['grappler_is_grappling']]) if duration > 0 else 0
        
        # Use a unique key for each pair's results
        prefix = f"grapple_{name}_pair{i+1}"
        final_results[f"{prefix}_outcome"] = res['final_outcome']
        final_results[f"{prefix}_duration"] = duration
        final_results[f"{prefix}_avg_grip"] = avg_grip
        final_results[f"{prefix}_details"] = res['details'] # Attach detailed log

    return final_results


def run_hive_assault_scenario(env, seed):
    """
    Runs the full hive assault and capture scenario.
    """
    print(f"    - Running Hive Assault Scenario")
    env.reset(seed=seed)
    
    # --- Setup ---
    target_hive_pos = env.hives[1]["pos"]
    attacker_state = "ATTACKING"
    hive_destroyed_step = -1
    last_owner = 1
    capture_successful = False
    
    step_log = []

    # --- Run Simulation ---
    for step in range(MAX_STEPS_PER_SCENARIO):
        actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env.agents))]
        for agent in env.agents:
            if not agent['alive']: continue
            if agent['team'] == 1: continue # Defenders are idle
            
            # Attacker (Team 0) logic
            hive1 = env.hives.get(1, {})
            if attacker_state == "ATTACKING":
                if hive1.get('state') == 'destroyed':
                    attacker_state = "CAPTURING"
                    hive_destroyed_step = step
                    if VERBOSE_DEBUG: print(f"    [Hive Assault Step {step}] Hive destroyed, moving to capture.")
                actions[agent['id']]['movement'] = move_towards(agent['pos'], target_hive_pos, stop_dist=AGENT_RADIUS * 2)
            
            elif attacker_state == "CAPTURING":
                if hive1.get('owner') == 0:
                    capture_successful = True
                    if VERBOSE_DEBUG: print(f"    [Hive Assault Step {step}] Hive capture successful!")
                    break

        if capture_successful: 
            # Log one final state
            hive1 = env.hives.get(1, {})
            step_log.append({
                'step': step, 'hive1_health': hive1.get('health', 0), 'hive1_food': hive1.get('food_store', 0),
                'hive1_state': hive1.get('state', 'unknown'), 'hive1_owner': hive1.get('owner', -1)
            })
            break

        _, _, terminated, truncated, _ = env.step(actions)

        # Log hive state at this step
        hive1 = env.hives.get(1, {})
        step_log.append({
            'step': step, 'hive1_health': hive1.get('health', 0), 'hive1_food': hive1.get('food_store', 0),
            'hive1_state': hive1.get('state', 'unknown'), 'hive1_owner': hive1.get('owner', -1)
        })

        if RENDER_SIMULATIONS: env.render(); time.sleep(STEP_DELAY)
        if terminated or truncated: break

    # --- Collect Results ---
    results = {
        "hive_assault_captured": capture_successful,
        "hive_assault_destroyed_step": hive_destroyed_step
    }
    results["hive_assault_details"] = step_log
    return results

# ==============================================================================
# ===                         SCENARIO DEFINITIONS                           ===
# ==============================================================================
# This list defines the more complex, policy-driven scenarios from combat_test.py
POLICY_SCENARIOS = [
    {
        "name": "1v1_Symmetric_Chase",
        "description": "Tests baseline combat damage.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 300], "policy": policy_chase_target, "target_id": 1},
            {"id": 1, "team": 1, "pos": [400, 300], "policy": policy_chase_target, "target_id": 0},
        ]
    },
    {
        "name": "2v1_Imbalance",
        "description": "Tests multi-agent combat dynamics.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 250], "policy": policy_chase_target, "target_id": 2},
            {"id": 1, "team": 0, "pos": [200, 350], "policy": policy_chase_target, "target_id": 2},
            {"id": 2, "team": 1, "pos": [400, 300], "policy": policy_chase_target, "target_id": 0},
        ]
    },
    {
        "name": "Flank_Rear_Attack",
        "description": "Tests flank bonus. Agent 0 (chaser) should do more damage.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [250, 300], "policy": policy_chase_target, "target_id": 1},
            {"id": 1, "team": 1, "pos": [350, 300], "policy": policy_flee_from_target,  "target_id": 0},
        ]
    },
    {
        "name": "Grapple_and_Spin_vs_Idle",
        "description": "Tests grapple initiation and torque rewards against a non-resisting target.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [250, 300], "policy": policy_grapple_and_spin, "target_id": 1},
            {"id": 1, "team": 1, "pos": [350, 300], "policy": policy_idle,             "target_id": None},
        ]
    },
    {
        "name": "Grapple_Spin_vs_Break",
        "description": "Tests grapple breaking against an active spinner.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [280, 300], "policy": policy_grapple_and_spin, "target_id": 1},
            {"id": 1, "team": 1, "pos": [320, 300], "policy": policy_try_to_break_grapple, "target_id": None},
        ]
    },
]

# This list defines the more granular, scripted physics tests from combat_testWgrapple.py
GRAPPLE_SCENARIOS = [
    {
        "name": "Struggle_Tug_of_War", 
        "setup_fn": setup_tug_of_war, 
        "control_fn": control_tug_of_war, 
    },
    {
        "name": "Torque_Struggle", 
        "setup_fn": setup_torque_struggle, 
        "control_fn": control_torque_struggle, 
    },
    {
        "name": "Momentum_Grapple",
        "setup_fn": setup_momentum_grapple,
        "control_fn": control_momentum_grapple,
    },
    {
        "name": "Energy_Drain_Break",
        "setup_fn": setup_energy_drain_break,
        "control_fn": control_energy_drain_break,
    },
    {
        "name": "Active_Breakout",
        "setup_fn": setup_active_breakout,
        "control_fn": control_active_breakout,
    },
]

# ==============================================================================
# ===                         MAIN EXECUTION BLOCK                           ===
# ==============================================================================

if __name__ == "__main__":
    if RENDER_SIMULATIONS and not pygame.get_init():
        pygame.init()
        pygame.display.init()

    all_results = []
    
    sweep_keys = list(parameter_sweep.keys())
    param_combinations = list(product(*(parameter_sweep[key] for key in sweep_keys)))
    
    base_seed = random.randint(0, 100000)
    print(f"Master seed for this run: {base_seed}")

    print(f"Starting combat tuning sweep with {len(param_combinations)} parameter combinations.")

    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(sweep_keys, combo))
        print(f"\n{'='*30} Starting Sweep {i+1}/{len(param_combinations)} {'='*30}")
        print(f"Parameters: {json.dumps(current_params, indent=2)}")

        # Create a single environment instance for this parameter set
        env_config = {**base_env_config, **current_params}
        env = Swarm2DEnv(**env_config, max_steps=MAX_STEPS_PER_SCENARIO + 50)
        
        run_summary = {'params': current_params}

        try:
            # ===========================================================
            # ==  HERE WE WILL CALL EACH OF THE SCENARIO RUNNER FUNCTIONS ==
            # ===========================================================
            policy_results = {}
            print("\n--- Running Policy-Based Scenarios ---")
            for sc_def in POLICY_SCENARIOS:
                policy_results.update(run_policy_based_scenario(sc_def['name'], env_config, policy_map={}, setup_fn=sc_def['setup_fn'], steps=MAX_STEPS_PER_SCENARIO, debug_interval=DEBUG_PRINT_FREQ))
                if RENDER_SIMULATIONS: time.sleep(PAUSE_BETWEEN_SCENARIOS_SEC / 2)
            run_summary.update(policy_results)
            
            grapple_results = {}
            print("\n--- Running Scripted Grapple Scenarios ---")
            for sc_def in GRAPPLE_SCENARIOS:
                grapple_results.update(run_scripted_grapple_scenario(env, sc_def, base_seed + len(grapple_results)))
                if RENDER_SIMULATIONS: time.sleep(PAUSE_BETWEEN_SCENARIOS_SEC / 2)
            run_summary.update(grapple_results)

            print("\n--- Running Hive Assault Scenario ---")
            hive_results = run_hive_assault_scenario(env, base_seed + 99)
            run_summary.update(hive_results)

            all_results.append(run_summary)

        except Exception as e:
            print(f"!!!!!! ERROR during sweep for params {current_params} !!!!!!")
            traceback.print_exc()
        finally:
            if env:
                env.close()
            if RENDER_SIMULATIONS:
                # Brief pause to allow window to close cleanly
                time.sleep(0.1)
        
        print(f"\n--- Finished Sweep {i+1}/{len(param_combinations)} ---")


    # --- Final Analysis ---
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n\n" + "="*30 + " FINAL COMBAT SWEEP SUMMARY " + "="*30)
        
        # Separate detailed logs from main results for clarity in summary
        detail_columns = [col for col in df.columns if col.endswith('_details')]
        summary_df = df.drop(columns=detail_columns)
        
        params_df = pd.json_normalize(summary_df['params'])
        results_df = summary_df.drop(columns=['params'])
        final_summary_df = pd.concat([results_df, params_df], axis=1)

        # We will add scoring logic here later based on the results
        
        print(final_summary_df.to_string())
        
        # Save to CSV
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = "tuning_results"
        os.makedirs(output_dir, exist_ok=True)
        
        summary_path = os.path.join(output_dir, f"combat_summary_{timestamp}.csv")
        final_summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Save the detailed logs to a JSON file
        detailed_log_path = os.path.join(output_dir, f"combat_details_{timestamp}.json")
        with open(detailed_log_path, 'w') as f:
            # Convert DataFrame to list of dicts for JSON serialization
            json.dump(df.to_dict(orient='records'), f, indent=2)
        print(f"Detailed logs saved to {detailed_log_path}")

    if pygame.get_init():
        pygame.quit()
    
    print("\nCombat tuning script finished.")
