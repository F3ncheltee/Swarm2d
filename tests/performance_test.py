import time
import numpy as np
import argparse
import cProfile
import pstats

try:
    from env.env import Swarm2DEnv
except ImportError:
    from Swarm2d.env.env import Swarm2DEnv

def run_performance_test(config, profile=False):
    """
    Runs a performance test on the Swarm2DEnv.

    Args:
        config (dict): Environment configuration.
        profile (bool): Whether to run the cProfile profiler.
    """
    print("--- Starting Performance Test ---")
    print(f"Configuration:")
    print(f"  - Teams: {config['num_teams']}")
    print(f"  - Agents per Team: {config['num_agents_per_team']}")
    print(f"  - Total Agents: {config['num_teams'] * config['num_agents_per_team']}")
    print(f"  - Simulation Steps: {config['max_steps']}")
    print(f"  - Render Mode: '{config['render_mode']}' (should be None for accurate SPS)")
    print("-" * 30)

    env = Swarm2DEnv(**config)
    env.reset()

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    start_time = time.time()

    for step in range(config['max_steps']):
        # Generate random actions for all agents
        actions = [{'movement': np.random.uniform(-1, 1, size=2),
                    'pickup': np.random.randint(0, 3)} for _ in range(env.num_agents)]
        
        obs, rewards, terminated, truncated, infos = env.step(actions)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{config['max_steps']}...")

        if terminated or truncated:
            print("Episode ended prematurely.")
            break
            
    end_time = time.time()

    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        print("\n--- Profiler Results (Top 15 cumulative time) ---")
        stats.print_stats(15)


    total_time = end_time - start_time
    sps = config['max_steps'] / total_time if total_time > 0 else float('inf')

    print("\n--- Performance Test Results ---")
    print(f"Total agents: {env.num_agents}")
    print(f"Total steps executed: {config['max_steps']}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Steps Per Second (SPS): {sps:.2f}")
    print("-" * 30)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance test for the Swarm2D Environment.")
    parser.add_argument("--teams", type=int, default=6, help="Number of teams.")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents per team.")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps to run.")
    parser.add_argument("--render", action="store_true", help="Enable rendering. Note: This will significantly impact SPS.")
    parser.add_argument("--profile", action="store_true", help="Run cProfile to identify performance bottlenecks.")

    args = parser.parse_args()

    # Use a baseline configuration and override with command-line arguments
    # MODIFIED: Copied the full, tuned configuration from test_env.py to ensure stable physics.
    env_config = {
        # Base simulation setup
        'num_teams': args.teams,
        'num_agents_per_team': args.agents,
        'num_resources': 100,
        'num_obstacles': 10,
        'max_steps': args.steps,
        'render_mode': 'human' if args.render else None,
        'debug': False,
        'use_gpu_occlusion_in_env': False,
        'use_pybullet_raycasting': True,

        # Agent Base Parameters
        'agent_radius': 3.0,
        'agent_base_strength': 1.0,
        'agent_max_energy': 100.0,
        'agent_max_health': 100.0,
        'sensing_range_fraction': 0.05,
        'recency_normalization_period': 250.0,

        # Tuned Physics & Combat Parameters from test_env.py
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
        'grapple_torque_scale': 25.0,
        'grapple_momentum_bonus_scale': 0.1,
        'agent_interaction_force_scale': 0.35,
        'grapple_momentum_decay': 0.95,
        'grapple_crit_chance': 0.05,
        'grapple_crit_multiplier': 3.0,
        'grapple_rear_crit_bonus_multiplier': 2.5,
        
        # ADDED: This was the missing key causing the hang.
        'agent_randomization_factors': {
            'bee_speed': {'base': 200.0, 'rand': 0.2},
            'agent_radius': {'base': 3.0, 'rand': 0.2},
            'agent_base_strength': {'base': 1.0, 'rand': 0.2},
            'agent_max_energy': {'base': 100.0, 'rand': 0.2},
            'agent_max_health': {'base': 100.0, 'rand': 0.2},
            'sensing_range_fraction': {'base': 0.05, 'rand': 0.2}
        }
    }
    
    run_performance_test(env_config, profile=args.profile)
