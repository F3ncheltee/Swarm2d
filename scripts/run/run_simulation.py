"""
A command-line script for running a headless or rendered Swarm2d simulation.

This script provides a simple way to run the Swarm2DEnv with a specified policy
without needing the full GUI. It is useful for quick tests, demonstrations, and
debugging.

Example Usage:
    # Run with the heuristic policy and rendering enabled
    python run_simulation.py --policy heuristic

    # Run with the random policy and rendering disabled
    python run_simulation.py --policy random --no-render
"""
import numpy as np
import torch
import argparse
import time
import pygame
import importlib
import inspect
import os

# --- Dynamic Policy Loading ---
# This allows the script to use any policy located in the 'policies' directory without having to manually import each one.
try:
    from Swarm2d.env.env import Swarm2DEnv
    # Get the path to the policies directory
    policies_dir = os.path.join(os.path.dirname(__file__), 'policies')
except ImportError:
    # Fallback for running from the project root
    from env.env import Swarm2DEnv
    policies_dir = os.path.join(os.path.dirname(__file__), 'Swarm2d', 'policies')

def discover_policies(policy_directory):
    """Dynamically finds and imports policy classes from a given directory."""
    policy_classes = {}
    for item in os.scandir(policy_directory):
        if item.is_dir() and not item.name.startswith('__'):
            try:
                module_path = f"Swarm2d.policies.{item.name}.{item.name}"
                module = importlib.import_module(module_path)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.lower().replace("_", "") == item.name.lower().replace("_", "") + "policy":
                        policy_classes[item.name.lower()] = obj
            except (ImportError, AttributeError):
                print(f"Warning: Could not load policy from directory '{item.name}'.")
    return policy_classes

# Discover available policies at startup
# AVAILABLE_POLICIES = discover_policies(policies_dir)
# Manual import for simplicity in this script
from Swarm2d.policies.randomPolicy.random_policy import RandomPolicy
from Swarm2d.policies.heuristicPolicy.heuristic_policy import HeuristicPolicy

AVAILABLE_POLICIES = {
    'random': RandomPolicy,
    'heuristic': HeuristicPolicy
}
# --- End Dynamic Policy Loading ---


def run_simulation(policy_name: str, render: bool):
    """
    Initializes and runs the Swarm2D simulation with a specified policy.

    Args:
        policy_name (str): The name of the policy to use (e.g., 'random', 'heuristic').
        render (bool): If True, the simulation will be rendered with Pygame.
    """
    print("--- Initializing Swarm2D Environment ---")
    # A detailed and tuned configuration for a standard simulation run.
    env_config = {
        'num_teams': 6,
        'num_agents_per_team': 20,
        'num_resources': 100,
        'num_obstacles': 10,
        'max_steps': 750,
        'render_mode': "human" if render else "headless",
        'debug': False,
        'use_gpu_occlusion_in_env': False,
        'use_pybullet_raycasting': True,
        'movement_force_scale': 15.0,
        'pb_agent_linear_damping': 0.11,
        'pb_agent_lateral_friction': 0.5,
        'pb_agent_angular_damping': 0.4,
        'resource_base_mass': 0.075,
        'resource_interaction_force_scale': 1.2,
        'pb_resource_constraint_max_force': 3000,
        'pb_res_friction_dynamic': 0.25,
        'pb_res_damping_dynamic': 0.25,
        'bee_speed': 200.0,
        'resource_mass_scale_factor': 1.4,
        'pb_coop_resource_constraint_max_force': 10000,
        'grappled_agent_counter_grip_scale': 0.3,
        'grapple_fatigue_rate': 0.02,
        'grapple_crush_damage_rate': 1.0,
        'grapple_struggle_damage_rate': 0.5,
        'pb_agent_constraint_max_force': 10000,
        'grapple_torque_escape_strength': 0.6,
        'grapple_momentum_bonus_scale': 0.1,
        'agent_interaction_force_scale': 0.35,
        'grapple_momentum_decay': 0.95,
        'grapple_crit_chance': 0.05,
        'grapple_crit_multiplier': 3.0,
        'grapple_rear_crit_bonus_multiplier': 2.5,
    }

    env = Swarm2DEnv(**env_config)

    # --- Policy Initialization ---
    policy_class = AVAILABLE_POLICIES.get(policy_name.lower())
    if not policy_class:
        raise ValueError(f"Unknown policy: '{policy_name}'. Available policies: {list(AVAILABLE_POLICIES.keys())}")
    
    # The policy needs to be instantiated for each agent.
    policies = [policy_class(env.action_space) for _ in range(env.num_agents)]
    
    print(f"--- Running simulation with '{policy_name}' policy ---")

    obs_list, info = env.reset()

    for step in range(env.max_steps):
        actions = []
        for i in range(env.num_agents):
            agent_obs = obs_list[i]
            # The 'act' method is expected to be part of the policy's public API.
            action = policies[i].act(agent_obs)
            actions.append(action)

        obs_list, rewards, terminated, truncated, infos = env.step(actions)

        if render:
            env.render()

        if terminated or truncated:
            print(f"Episode finished at step {step}. Terminated={terminated}, Truncated={truncated}")
            break

    # After the simulation loop, keep the window open for inspection if rendering.
    if render:
        print("\nSimulation finished. Close the Pygame window to exit.")
        running = True
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                env.render()
                time.sleep(0.1)
            except (pygame.error, AttributeError):
                # This can happen if the Pygame window is closed unexpectedly.
                running = False


    env.close()
    print("--- Simulation Closed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Swarm2D simulation with a specific policy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=list(AVAILABLE_POLICIES.keys()),
        help="The policy to use for the simulation.\n"
             "Available choices: %(choices)s"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="If set, the simulation will run in headless mode without visualization."
    )
    args = parser.parse_args()

    run_simulation(policy_name=args.policy, render=not args.no_render)
