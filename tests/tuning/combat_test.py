#!/usr/bin/env python3
import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import json
from collections import defaultdict

# --- Python Path (ensure Swarm2DEnv is found) ---
try:
    # Ensure the path to the project root (BubbleFight) is in sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.env.swarm2denvGNN import (
        Swarm2DEnv, AGENT_RADIUS, COMBAT_RADIUS,
        AGENT_MAX_HEALTH, AGENT_MAX_ENERGY, AGENT_BASE_STRENGTH,
        REWARD_COMPONENT_KEYS
    )
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    print("Please ensure the script is run from the 'tests/combat/' directory or the project root is in your PYTHONPATH.")
    sys.exit(1)

# ==============================================================================
# ===                 TESTING FOCUS & CONTROL CONSTANTS                      ===
# ==============================================================================
MAX_STEPS_PER_SCENARIO = 750  # Approx 25 seconds at 30 FPS
RENDER_MODE = True
STEP_DELAY = 0.01  # For visual observation (if RENDER_MODE is True)
PAUSE_BETWEEN_SCENARIOS_SEC = 2.5
SCENARIO_BASE_SEED = random.randint(0, 100000)

# Base environment parameters for combat focused tests
BASE_ENV_PARAMS = {
    'num_teams': 2,
    'num_agents_per_team': 5, # Create enough agents to be assigned by scenarios
    'num_resources': 2,     # Few resources to reduce clutter
    'num_obstacles': 1,     # Few obstacles
    'width': 600,           # Smaller arena for faster encounters
    'height': 600,
    'pb_agent_mass': 1.0,
    'movement_force_scale': 60.0,
    'pb_agent_linear_damping': 0.6,
    'render_mode': RENDER_MODE,
}

# ============================================================================
# ===               AGENT CONTROL POLICIES & HELPERS                       ===
# ============================================================================
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalizes a numpy vector, handling the zero-vector case."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)

def get_nearest_enemy(env, agent_idx):
    """Finds the nearest alive enemy agent."""
    agent = env.agents[agent_idx]
    if not agent['alive'] or agent.get('pos') is None: return None
    
    nearest_enemy = None
    min_dist_sq = float('inf')
    
    for other_agent in env.agents:
        if other_agent['alive'] and other_agent['team'] != agent['team'] and other_agent.get('pos') is not None:
            dist_sq = np.sum((agent['pos'] - other_agent['pos'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_enemy = other_agent
    return nearest_enemy

# --- Action Policies ---
def policy_idle(env, agent_idx, target): return {'movement': np.zeros(2), 'pickup': 0}

def get_nearest_resource(env, agent_idx):
    """Finds the nearest available resource."""
    agent = env.agents[agent_idx]
    if not agent['alive'] or agent.get('pos') is None: return None
    
    nearest_res, min_dist_sq = None, float('inf')
    for res in env.resources:
        if res and not res.get('delivered') and res.get('pos') is not None and not res.get('carriers'):
            dist_sq = np.sum((agent['pos'] - res['pos'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq, nearest_res = dist_sq, res
    return nearest_res

def policy_chase_and_collect(env, agent_idx, target):
    """
    (V2) Primary: Chase/attack the target. Secondary: Find nearest loot and collect it.
    Fixes the pickup distance check to prevent attaching from too far.
    """
    agent = env.agents[agent_idx]
    if not agent['alive'] or agent.get('pos') is None:
        return policy_idle(env, agent_idx, target)

    # Primary objective: Attack alive target
    if target and target['alive'] and target.get('pos') is not None:
        direction = target['pos'] - agent['pos']
        dist = np.linalg.norm(direction)
        # Attempt to grapple/attack if within combat radius
        pickup_action = 1 if dist < (COMBAT_RADIUS * 0.9) else 0
        movement = normalize_vector(direction)
        return {'movement': movement, 'pickup': pickup_action}
    
    # Secondary objective: Target is dead, collect loot
    else:
        resource_target = get_nearest_resource(env, agent_idx)
        if resource_target and resource_target.get('pos') is not None:
            direction = resource_target['pos'] - agent['pos']
            dist = np.linalg.norm(direction)
            
            # BUG FIX: Only request pickup if the agent is nearly touching the resource.
            required_dist_for_pickup = agent['agent_radius'] + resource_target['radius_pb']
            pickup_action = 1 if dist < required_dist_for_pickup + 2.0 else 0
            
            movement = normalize_vector(direction)
            return {'movement': movement, 'pickup': pickup_action}
        
    # No targets, go idle
    return policy_idle(env, agent_idx, target)

def policy_flee(env, agent_idx, target):
    """Agent flees from the nearest enemy."""
    agent = env.agents[agent_idx]
    nearest_enemy = get_nearest_enemy(env, agent_idx)
    if not agent['alive'] or agent.get('pos') is None or nearest_enemy is None:
        return policy_idle(env, agent_idx, target)
        
    flee_direction = agent['pos'] - nearest_enemy['pos']
    movement = normalize_vector(flee_direction)
    return {'movement': movement, 'pickup': 0} # Never try to pickup when fleeing

def policy_grapple_and_spin(env, agent_idx, target):
    """Chases, attempts to grapple, and then spins to apply torque."""
    agent = env.agents[agent_idx]
    if agent.get('is_grappling'):
        # Already grappling, so apply max torque by moving 'right'
        return {'movement': np.array([1.0, 0.0]), 'pickup': 0}
    else:
        # Not grappling, chase the target to initiate
        return policy_chase_and_collect(env, agent_idx, target)

def policy_try_to_break_grapple(env, agent_idx, target):
    """If grappled, attempts to break free by rotating and using the release action."""
    agent = env.agents[agent_idx]
    if agent.get('is_grappled'):
        # Counter-rotate and spam release
        return {'movement': np.array([-1.0, 0.0]), 'pickup': 2}
    else:
        # Not grappled, just stay idle or move slightly
        return policy_idle(env, agent_idx, target)

# ============================================================================
# ===                   SCENARIO & SIMULATION RUNNER                       ===
# ============================================================================

def spawn_cooperative_resource(env, pos, size=10.0):
    """
    Directly spawns a large, cooperative resource at a specific location for testing.
    Removes the first available normal resource to make space in the env list.
    """
    # Find a resource to replace in the environment's list
    for i, res in enumerate(env.resources):
        if res and not res.get('delivered'):
            # Remove the old resource from PyBullet and the list
            if res.get('body_id') is not None:
                try: p.removeBody(res['body_id'], physicsClientId=env.physicsClient)
                except p.error: pass
            
            # Use the environment's internal spawner to create the new one correctly
            # We use _spawn_resource_at_location to bypass random placement
            print(f"  - Spawning COOP resource at {pos} with size {size}")
            new_res = env._spawn_resource_at_location(target_pos=np.array(pos), size=size, cooperative=True)
            
            if new_res:
                env.resources[i] = new_res # Replace the old resource
                return new_res
            else:
                print("  - WARNING: Failed to spawn cooperative resource.")
                return None
    
    print("  - WARNING: No existing resource found to replace for coop spawn.")
    return None

def policy_coop_collect(env, agent_idx, target_resource):
    """
    (V2) A policy for agents to converge on a resource and carry it to their hive.
    Fixes the pickup distance check.
    """
    agent = env.agents[agent_idx]
    if not agent['alive'] or agent.get('pos') is None:
        return policy_idle(env, agent_idx, target_resource)

    # If already carrying this resource, move towards the hive
    if agent.get('has_resource') and agent.get('resource_obj') == target_resource:
        hive = env.hives.get(agent['team'])
        if hive and hive.get('pos') is not None:
            direction = hive['pos'] - agent['pos']
            return {'movement': normalize_vector(direction), 'pickup': 0} # 0 since already carrying
        else:
            return policy_idle(env, agent_idx, target_resource)

    # If not carrying, move towards the target resource
    if target_resource and not target_resource.get('delivered') and target_resource.get('pos') is not None:
        direction = target_resource['pos'] - agent['pos']
        dist = np.linalg.norm(direction)
        
        # BUG FIX: Only request pickup if the agent is nearly touching the resource.
        required_dist_for_pickup = agent['agent_radius'] + target_resource['radius_pb']
        pickup_action = 1 if dist < required_dist_for_pickup + 2.0 else 0

        movement = normalize_vector(direction)
        return {'movement': movement, 'pickup': pickup_action}

    # No target, go idle
    return policy_idle(env, agent_idx, target_resource)

def setup_scenario_agents(env, agent_configs):
    """
    (V2) Configures agents based on a list of configuration dictionaries.
    Stores target IDs instead of stale objects for robust lookup.
    """
    active_indices = []
    # Create a map for agent-to-agent targeting
    agent_map = {f"T{ac['team']}_{ac.get('id', i)}": i for i, ac in enumerate(agent_configs)}

    for agent in env.agents:
        agent['alive'] = False
        p.resetBasePositionAndOrientation(agent['body_id'], [10000, 10000, -100], [0,0,0,1], physicsClientId=env.physicsClient)
    
    for i, config in enumerate(agent_configs):
        # Ensure we don't try to configure more agents than exist in the env
        if i >= len(env.agents):
            print(f"Warning: Scenario wants to configure agent {i}, but only {len(env.agents)} exist. Stopping setup.")
            break
            
        agent_idx = i
        active_indices.append(agent_idx)
        agent = env.agents[agent_idx]

        # Reset agent state
        agent['team'] = config['team']
        agent['alive'] = True
        agent['health'] = config.get('health', agent['max_health'])
        agent['energy'] = config.get('energy', agent['max_energy'])
        
        pos = config['pos']
        p.resetBasePositionAndOrientation(agent['body_id'], [pos[0], pos[1], agent['agent_radius']], [0,0,0,1], physicsClientId=env.physicsClient)
        p.resetBaseVelocity(agent['body_id'], config.get('vel', [0,0,0]), [0,0,0], physicsClientId=env.physicsClient)
        
        agent['policy'] = config['policy']
        
        # --- ROBUST TARGETING ---
        # Store the ID of the target, not the object itself.
        target_name = config.get('target')
        if isinstance(target_name, str) and target_name.startswith('res_id_'):
            agent['target_idx'] = None
            agent['target_resource_id'] = target_name # Store resource ID
        elif isinstance(target_name, str) and target_name.startswith('T'):
            agent['target_idx'] = agent_map.get(target_name)
            agent['target_resource_id'] = None # No resource target
        else:
            agent['target_idx'] = None
            agent['target_resource_id'] = None
            
        print(f"  - Configuring Agent {agent['id']} (Index {agent_idx}): Team {agent['team']}, Policy '{config['policy'].__name__}'")

    return active_indices

def run_scenario(env_instance, scenario_def):
    """
    (V3) Runs a single scenario with robust target fetching and detailed cooperative metrics,
    inspired by the advanced physics tuning script.
    """
    name = scenario_def['name']
    print(f"\n{'='*20} Starting Scenario: {name} {'='*20}")
    print(f"  Description: {scenario_def['description']}")
    
    coop_target_res_id = None
    if 'resource_spawns' in scenario_def:
        for res_spawn in scenario_def['resource_spawns']:
            # Replace placeholder ID in agent configs with the real ID from the spawner
            placeholder_id = res_spawn.get('id')
            spawned_res = spawn_cooperative_resource(env_instance, res_spawn['pos'], res_spawn['size'])
            if spawned_res:
                real_res_id_str = f"res_id_{spawned_res['id']}"
                if spawned_res.get('cooperative'):
                    coop_target_res_id = real_res_id_str # Track the main coop target
                for agent_conf in scenario_def['agent_configs']:
                    if agent_conf.get('target') == placeholder_id:
                        agent_conf['target'] = real_res_id_str # Replace with real ID string

    active_agent_indices = setup_scenario_agents(env_instance, scenario_def['agent_configs'])
    for _ in range(5): p.stepSimulation(physicsClientId=env_instance.physicsClient)

    agent_ids = [env_instance.agents[i]['id'] for i in active_agent_indices]
    initial_health = {aid: env_instance.agents[i]['health'] for i, aid in zip(active_agent_indices, agent_ids)}

    # --- Enhanced Metrics Tracking ---
    scenario_data = {
        "ids": agent_ids, "health": initial_health.copy(),
        "damage_taken": defaultdict(float),
        "rewards": {aid: defaultdict(float) for aid in agent_ids},
        "attachments": defaultdict(int), "delivered": False, "delivery_step": -1,
        "coop_metrics": { # Nested dict for clarity
            "ever_moved": False, "first_move_step": -1,
            "max_carriers": 0, "final_dist_to_hive": -1.0,
            "total_dist_moved": 0.0,
            "last_pos": None # Will be populated if it's a coop scenario
        }
    }
    
    # Initialize last_pos for coop metrics
    if coop_target_res_id:
        res_obj = next((r for r in env_instance.resources if f"res_id_{r['id']}" == coop_target_res_id), None)
        if res_obj: scenario_data["coop_metrics"]["last_pos"] = np.array(res_obj['pos'])

    for step in range(MAX_STEPS_PER_SCENARIO):
        actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(len(env_instance.agents))]
        
        # --- CORE FIX: LIVE TARGET LOOKUP ---
        for agent_idx in active_agent_indices:
            agent = env_instance.agents[agent_idx]
            if not agent['alive']: continue

            target_for_policy = None
            # Prioritize resource target ID lookup
            target_res_id_str = agent.get('target_resource_id')
            if target_res_id_str:
                # Find the LIVE resource object in the environment on every step
                res_id_to_find = int(target_res_id_str.split('_')[-1])
                target_for_policy = next((res for res in env_instance.resources if res and res.get('id') == res_id_to_find), None)
            else: # Fallback to agent target
                target_agent_idx = agent.get('target_idx')
                if target_agent_idx is not None:
                    target_for_policy = env_instance.agents[target_agent_idx]
            
            # The policy now receives a live, up-to-date object or None
            actions[agent_idx] = agent['policy'](env_instance, agent_idx, target_for_policy)
        
        prev_health = scenario_data['health'].copy()
        obs_list, rewards_list, terminated, truncated, infos = env_instance.step(actions)
        current_health = {env_instance.agents[i]['id']: env_instance.agents[i]['health'] for i in active_agent_indices}
        
        # --- DETAILED METRICS UPDATE ---
        for i, agent_idx in enumerate(active_agent_indices):
            agent_id, agent = agent_ids[i], env_instance.agents[agent_idx]
            scenario_data['damage_taken'][agent_id] += prev_health[agent_id] - current_health[agent_id]
            for r_key, r_val in rewards_list[agent_idx].items():
                scenario_data['rewards'][agent_id][r_key] += float(r_val)
            if agent.get('has_resource') and agent.get('resource_obj'):
                if f"res_id_{agent['resource_obj']['id']}" == agent.get('target_resource_id'):
                    scenario_data['attachments'][agent_id] = 1 # Mark as attached
        
        # Update coop metrics if a coop resource is the target
        if coop_target_res_id:
            res_id_to_track = int(coop_target_res_id.split('_')[-1])
            res_obj_live = next((r for r in env_instance.resources if r and r['id'] == res_id_to_track), None)
            if res_obj_live:
                scenario_data["coop_metrics"]["max_carriers"] = max(scenario_data["coop_metrics"]["max_carriers"], len(res_obj_live.get('carriers', [])))
                current_pos = np.array(res_obj_live['pos'])
                last_pos = scenario_data["coop_metrics"]["last_pos"]
                if last_pos is not None:
                    dist_moved_this_step = np.linalg.norm(current_pos - last_pos)
                    if dist_moved_this_step > 0.01: # Threshold for meaningful movement
                        scenario_data["coop_metrics"]["total_dist_moved"] += dist_moved_this_step
                        if not scenario_data["coop_metrics"]["ever_moved"]:
                            scenario_data["coop_metrics"]["ever_moved"] = True
                            scenario_data["coop_metrics"]["first_move_step"] = step
                scenario_data["coop_metrics"]["last_pos"] = current_pos
                # Update final distance to hive
                hive_pos = env_instance.hives[0]['pos']
                scenario_data["coop_metrics"]["final_dist_to_hive"] = np.linalg.norm(current_pos - hive_pos)

        if infos.get("delivered_resource_ids_this_step") and not scenario_data["delivered"]:
            scenario_data["delivered"] = True
            scenario_data["delivery_step"] = step

        scenario_data['health'] = current_health
        if RENDER_MODE: env_instance.render(); time.sleep(STEP_DELAY)
        if scenario_data["delivered"]:
            print(f"  -> Scenario ended early at step {step}: resource delivered.")
            break
        
    print(f"  --- Scenario Summary: {name} ---")
    if "Cooperative" in name:
        cm = scenario_data['coop_metrics']
        print(f"  Coop Metrics:")
        print(f"    - Resource Delivered: {scenario_data['delivered']} (at step {scenario_data['delivery_step']})")
        print(f"    - Resource Ever Moved: {cm['ever_moved']} (first move at step {cm['first_move_step']})")
        print(f"    - Max Concurrent Carriers: {cm['max_carriers']}")
        print(f"    - Total Distance Moved: {cm['total_dist_moved']:.2f}")
        print(f"    - Final Distance to Hive: {cm['final_dist_to_hive']:.2f}")
    
    for agent_id in agent_ids:
        print(f"    Agent {agent_id}:")
        print(f"      - Final Health: {scenario_data['health'].get(agent_id, 0):.1f}")
        print(f"      - Successfully Attached: {'Yes' if scenario_data['attachments'].get(agent_id) else 'No'}")
        clean_rewards = {k: round(v, 2) for k, v in scenario_data['rewards'].get(agent_id, {}).items() if abs(v) > 1e-4}
        print(f"      - Rewards: {clean_rewards or '{}'}")
    
    return { "name": name, "data": scenario_data }
# ============================================================================
# ===                         SCENARIO DEFINITIONS                         ===
# ============================================================================
SCENARIOS = [
    {
        "name": "2-Agent Cooperative Carry",
        "description": "Tests if two agents can successfully attach to and carry a large resource.",
        "resource_spawns": [
            # Use a placeholder ID that will be replaced. Size is now valid.
            {"id": "placeholder_coop_1", "pos": [300, 300], "size": 9.0}
        ],
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 250], "policy": policy_coop_collect, "target": "placeholder_coop_1"},
            {"id": 1, "team": 0, "pos": [200, 350], "policy": policy_coop_collect, "target": "placeholder_coop_1"},
        ]
    },
    {
        "name": "3-Agent Cooperative Carry",
        "description": "Tests if three agents can successfully attach to and carry a large resource.",
        "resource_spawns": [
            # Use a valid size of 10.0 (max size)
            {"id": "placeholder_coop_2", "pos": [300, 300], "size": 10.0}
        ],
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [150, 300], "policy": policy_coop_collect, "target": "placeholder_coop_2"},
            {"id": 1, "team": 0, "pos": [200, 250], "policy": policy_coop_collect, "target": "placeholder_coop_2"},
            {"id": 2, "team": 0, "pos": [200, 350], "policy": policy_coop_collect, "target": "placeholder_coop_2"},
        ]
    },
    # ... (the rest of your combat scenarios can remain as they were, they target agents by name string like "T1_1")
    {
        "name": "1v1 Symmetric Chase",
        "description": "Tests baseline combat damage. Two identical agents charge each other, winner collects loot.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 300], "policy": policy_chase_and_collect, "target": "T1_1"},
            {"id": 1, "team": 1, "pos": [400, 300], "policy": policy_chase_and_collect, "target": "T0_0"},
        ]
    },
    {
        "name": "Strong vs Weak",
        "description": "Tests strength modifier. A high-strength agent fights a low-strength one.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 300], "policy": policy_chase_and_collect, "target": "T1_1", "strength": 1.5},
            {"id": 1, "team": 1, "pos": [400, 300], "policy": policy_chase_and_collect, "target": "T0_0", "strength": 0.5},
        ]
    },
    {
        "name": "Flank / Rear Attack",
        "description": "Tests flank bonus. Agent 0 (chaser) should do more damage to Agent 1 (fleeing).",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [250, 300], "policy": policy_chase_and_collect, "target": "T1_1"},
            {"id": 1, "team": 1, "pos": [350, 300], "policy": policy_flee,  "target": "T0_0"},
        ]
    },
    {
        "name": "Grapple and Spin",
        "description": "Tests grapple initiation and torque rewards. A0 tries to grapple and spin A1.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [250, 300], "policy": policy_grapple_and_spin, "target": "T1_1"},
            {"id": 1, "team": 1, "pos": [350, 300], "policy": policy_idle,             "target": None},
        ]
    },
    {
        "name": "Grapple Break",
        "description": "Tests grapple breaking. A0 grapples A1, and A1 actively tries to break free.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [280, 300], "policy": policy_grapple_and_spin, "target": "T1_1"},
            {"id": 1, "team": 1, "pos": [320, 300], "policy": policy_try_to_break_grapple, "target": None},
        ]
    },
    {
        "name": "2v1 Imbalance",
        "description": "Tests multi-agent combat dynamics.",
        "agent_configs": [
            {"id": 0, "team": 0, "pos": [200, 250], "policy": policy_chase_and_collect, "target": "T1_2"},
            {"id": 1, "team": 0, "pos": [200, 350], "policy": policy_chase_and_collect, "target": "T1_2"},
            {"id": 2, "team": 1, "pos": [400, 300], "policy": policy_chase_and_collect, "target": "T0_0"},
        ]
    },
]
# ============================================================================
# ===                         MAIN EXECUTION BLOCK                         ===
# ============================================================================
if __name__ == "__main__":
    all_results = []
    
    # Create a single, persistent environment instance
    env = Swarm2DEnv(**BASE_ENV_PARAMS, max_steps=MAX_STEPS_PER_SCENARIO + 50)
    
    try:
        for i, scenario_def in enumerate(SCENARIOS):
            # Reset the environment to a clean state for the new scenario
            env.reset(seed=SCENARIO_BASE_SEED + i)
            
            result = run_scenario(env, scenario_def)
            all_results.append(result)

            if i < len(SCENARIOS) - 1:
                print(f"\nPausing for {PAUSE_BETWEEN_SCENARIOS_SEC}s...")
                time.sleep(PAUSE_BETWEEN_SCENARIOS_SEC)
    
    except Exception as e:
        print("\n--- AN ERROR OCCURRED DURING SCENARIO EXECUTION ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # --- Save All Results ---
        if all_results:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            summary_path = f"combat_tuning_summary_{timestamp}.csv"
            
            flat_results = []
            for res in all_results:
                if not res: continue
                flat_row = {"scenario": res['name']}
                for agent_id in res['data']['ids']:
                    flat_row[f"agent_{agent_id}_total_dmg_taken"] = res['data']['damage_taken'][agent_id]
                    flat_row[f"agent_{agent_id}_grapple_breaks"] = res['data']['grapple_breaks'][agent_id]
                    flat_row[f"agent_{agent_id}_grapple_success"] = res['data']['grapple_successes'][agent_id]
                    # Add any other summary stats you want per-agent
                flat_results.append(flat_row)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(summary_path, index=False, float_format="%.2f")
            print(f"\nALL combat scenario results summarized in: {summary_path}")
        
        print("\nClosing environment.")
        env.close()
        # Pygame is handled by the env's close method
        print("\nCombat Test Script finished.")