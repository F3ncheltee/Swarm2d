import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict

from constants import REWARD_COMPONENT_KEYS

# --- Default Configuration Function ---
def get_default_config() -> Dict:
    """
    Returns the default configuration for the simulation environment and GUI.
    It's defined here to avoid circular imports and keep configs in one place.
    """
    return get_updated_config()

def get_updated_config():
    """
    Returns the default configuration for the simulation environment and GUI.
    It's defined here to avoid circular imports and keep configs in one place.
    """
    # This dictionary defines the structure of the GUI and the default values.
    # The environment itself should have its own robust defaults.
    config = {
        "General": {
            "help_text": "Core simulation settings.",
            "num_teams": {"value": 2, "type": "int", "range": [1, 10], "help_text": "Number of teams."},
            "num_agents_per_team": {"value": 10, "type": "int", "range": [1, 50], "help_text": "Agents per team.", "overridable": True},
            "num_resources": {"value": 100, "type": "int", "range": [0, 500], "help_text": "Total initial resources."},
            "num_obstacles": {"value": 20, "type": "int", "range": [0, 100], "help_text": "Total obstacles."},
            "max_steps": {"value": 5000, "type": "int", "range": [100, 10000], "help_text": "Max steps per episode."},
            "Arena": {
                "help_text": "Arena dimensions and properties.",
                "width": {"value": 1000, "type": "int", "range": [200, 5000], "help_text": "Width of the arena."},
                "height": {"value": 1000, "type": "int", "range": [200, 5000], "help_text": "Height of the arena."}
            }
        },
        "Rendering & Debug": {
            "help_text": "Settings for visualization and debugging.",
            "render_pybullet_gui": {"value": True, "type": "bool", "help_text": "Enable PyBullet's graphical interface."},
            "render_pygame_window": {"value": True, "type": "bool", "help_text": "Enable the 2D Pygame window."},
            "debug": {"value": False, "type": "bool", "help_text": "Enable debug prints and visualizations."},
            "enable_profiling": {"value": False, "type": "bool", "help_text": "Enable step-by-step performance profiling."}
        },
        "Physics": {
            "help_text": "Parameters governing the physics simulation.",
            "Agent Physics": {
                "help_text": "Agent movement and interaction physics.",
                "movement_force_scale": {"value": 15.0, "type": "float", "range": [0, 50], "help_text": "Force applied for agent movement."},
                "pb_agent_mass": {"value": 1.0, "type": "float", "range": [0.1, 10.0], "help_text": "Base mass for a standard agent."},
                "pb_agent_linear_damping": {"value": 0.11, "type": "float", "range": [0, 1], "help_text": "Linear damping for agents."},
                "pb_agent_angular_damping": {"value": 0.4, "type": "float", "range": [0, 1], "help_text": "Angular damping for agents."},
                "pb_agent_lateral_friction": {"value": 0.5, "type": "float", "range": [0, 2], "help_text": "Friction between agents and surfaces."},
                "pb_agent_spinning_friction": {"value": 0.03, "type": "float", "range": [0, 1], "help_text": "Spinning friction for agents."},
                "pb_agent_rolling_friction": {"value": 0.01, "type": "float", "range": [0, 1], "help_text": "Rolling friction for agents."},
                "pb_agent_restitution": {"value": 0.5, "type": "float", "range": [0, 1], "help_text": "Bounciness of agents."}
            },
            "Resource Physics": {
                "help_text": "Resource movement and interaction physics.",
                "resource_base_mass": {"value": 0.075, "type": "float", "range": [0.01, 1], "help_text": "Base mass for resources."},
                "resource_interaction_force_scale": {"value": 1.2, "type": "float", "range": [0, 5], "help_text": "Force scale for carrying resources."},
                "pb_resource_constraint_max_force": {"value": 3000, "type": "int", "range": [100, 10000], "help_text": "Max force for resource constraints."},
                "pb_res_friction_static": {"value": 0.35, "type": "float", "range": [0, 2], "help_text": "Static friction for free resources."},
                "pb_res_friction_dynamic": {"value": 0.25, "type": "float", "range": [0, 2], "help_text": "Dynamic friction for carried resources."},
                "pb_res_damping_static": {"value": 0.35, "type": "float", "range": [0, 2], "help_text": "Damping for free resources."},
                "pb_res_damping_dynamic": {"value": 0.25, "type": "float", "range": [0, 2], "help_text": "Damping for carried resources."},
                "pb_res_restitution": {"value": 0.4, "type": "float", "range": [0, 1], "help_text": "Bounciness of resources."},
                "resource_push_force_scale": {"value": 0.0, "type": "float", "range": [0, 10], "help_text": "Force applied when agents push loose resources."}
            },
            "Cooperative Resource Physics": {
                "help_text": "Settings specific to cooperative resources.",
                "resource_mass_scale_factor": {"value": 1.4, "type": "float", "range": [0, 5], "help_text": "Mass scaling factor for coop resources."},
                "pb_coop_resource_constraint_max_force": {"value": 10000, "type": "int", "range": [100, 20000], "help_text": "Max constraint force for coop resources."}
            },
            "Grapple Physics": {
                "help_text": "Settings for agent grappling interactions.",
                "agent_interaction_force_scale": {"value": 0.35, "type": "float", "range": [0, 2], "help_text": "Force scale for agent interactions."},
                "pb_agent_constraint_max_force": {"value": 13000, "type": "int", "range": [100, 20000], "help_text": "Max force for grapple constraints."},
                "grapple_torque_scale": {"value": 25.0, "type": "float", "range": [0, 100], "help_text": "How much movement input translates to twisting torque."},
                "grapple_torque_escape_strength": {"value": 0.6, "type": "float", "range": [0, 2], "help_text": "How much opposing torque weakens a grapple."}
            },
            "Standard Combat Modifiers": {
                "help_text": "Parameters that modify standard (non-grapple) combat damage.",
                "agent_base_damage": {"value": 0.05, "type": "float", "range": [0, 1], "help_text": "Base damage before any modifiers."},
                "agent_strength_damage_mod": {"value": 0.5, "type": "float", "range": [0, 2], "help_text": "Influence of agent strength on damage."},
                "agent_energy_damage_mod": {"value": 0.5, "type": "float", "range": [0, 2], "help_text": "Influence of agent energy on damage."},
                "agent_size_damage_mod": {"value": 0.2, "type": "float", "range": [0, 2], "help_text": "Influence of agent size on damage."},
                "agent_damage_stochasticity": {"value": 0.1, "type": "float", "range": [0, 1], "help_text": "Random damage variance (+/-)."},
                "attacker_facing_threshold": {"value": 0.3, "type": "float", "range": [-1, 1], "help_text": "Dot product; lower is a wider attack cone."},
                "attacker_not_facing_damage_multiplier": {"value": 0.3, "type": "float", "range": [0, 1], "help_text": "Damage multiplier if attacker is not facing target."},
                "attacker_stationary_damage_multiplier": {"value": 0.6, "type": "float", "range": [0, 1], "help_text": "Damage multiplier if attacker is stationary."},
                "attacker_speed_bonus_threshold_ratio": {"value": 0.6, "type": "float", "range": [0, 1], "help_text": "Fraction of max speed needed for speed bonus."},
                "max_attacker_speed_damage_mod": {"value": 0.2, "type": "float", "range": [0, 1], "help_text": "Max damage bonus % from speed."},
                "defender_strength_mitigation_mod": {"value": 0.15, "type": "float", "range": [0, 1], "help_text": "How much defender strength reduces damage."},
                "defender_mass_mitigation_mod": {"value": 0.05, "type": "float", "range": [0, 1], "help_text": "How much defender mass reduces damage."},
                "min_damage_after_mitigation": {"value": 0.1, "type": "float", "range": [0, 1], "help_text": "Minimum % of damage taken after mitigation."},
                "flank_rear_attack_bonus_multiplier": {"value": 1.25, "type": "float", "range": [1, 5], "help_text": "Damage bonus for attacking from behind."},
                "flank_rear_defender_facing_threshold": {"value": -0.2, "type": "float", "range": [-1, 1], "help_text": "Dot product for what counts as a rear attack."},
                "agent_slowed_duration": {"value": 5, "type": "int", "range": [0, 100], "help_text": "How many steps an agent is slowed for after being hit."},
                "agent_slowed_factor": {"value": 0.5, "type": "float", "range": [0, 1], "help_text": "Speed multiplier when slowed (lower is slower)."}
            },
            "Hive Combat Modifiers": {
                "help_text": "Parameters that modify hive combat outcomes.",
                "hive_core_min_size": {"value": 8.0, "type": "float", "range": [1, 20], "help_text": "Minimum size of a dropped hive core."},
                "hive_core_max_size": {"value": 15.0, "type": "float", "range": [1, 30], "help_text": "Maximum size of a dropped hive core."},
                "hive_core_food_to_size_ratio": {"value": 0.1, "type": "float", "range": [0, 1], "help_text": "Ratio of hive food to core size on destruction."},
                "hive_damage_points_per_bleed_chunk": {"value": 5.0, "type": "float", "range": [1, 20], "help_text": "Damage needed to cause a resource chunk to drop."},
                "hive_bleed_resource_size": {"value": 3.0, "type": "float", "range": [1, 10], "help_text": "Size of resource chunks dropped from a damaged hive."}
            },
            "Grapple Combat Modifiers": {
                "help_text": "Parameters that modify grappling outcomes and damage.",
                "grappled_agent_counter_grip_scale": {"value": 0.3, "type": "float", "range": [0, 2], "help_text": "How effectively a grappled agent can counter-grip."},
                "grapple_fatigue_rate": {"value": 0.02, "type": "float", "range": [0, 0.1], "help_text": "Rate of fatigue when grappling."},
                "grapple_crush_damage_rate": {"value": 1.0, "type": "float", "range": [0, 5], "help_text": "Passive damage per step to a grappled agent."},
                "grapple_struggle_damage_rate": {"value": 0.5, "type": "float", "range": [0, 5], "help_text": "Damage grappled agent deals back to grappler."},
                "grapple_momentum_bonus_scale": {"value": 0.1, "type": "float", "range": [0, 1], "help_text": "How much target momentum adds to initial grip strength."},
                "grapple_momentum_decay": {"value": 0.95, "type": "float", "range": [0.8, 1], "help_text": "Decay rate for grapple momentum bonus."},
                "grapple_crit_chance": {"value": 0.05, "type": "float", "range": [0, 1], "help_text": "Chance for a critical hit in a grapple."},
                "grapple_crit_multiplier": {"value": 3.0, "type": "float", "range": [1, 10], "help_text": "Damage multiplier for critical hits."},
                "grapple_rear_crit_bonus_multiplier": {"value": 2.5, "type": "float", "range": [1, 5], "help_text": "Bonus multiplier for rear critical hits."}
            }
        },
        "Spawning": {
            "help_text": "Parameters controlling the initial placement of entities.",
            "Hive Spawning": {
                "help_text": "Settings for hive placement.",
                "hive_min_distance": {"value": 120.0, "type": "float", "range": [50, 500], "help_text": "Minimum distance between hives."},
                "hive_spawn_jitter": {"value": 50.0, "type": "float", "range": [0, 200], "help_text": "Random jitter applied to hive spawn locations."},
                "hive_spawn_radius_factor": {"value": 0.35, "type": "float", "range": [0.1, 0.8], "help_text": "Factor of arena radius for hive spawning circle."}
            },
            "Resource Spawning": {
                "help_text": "Settings for resource placement and types.",
                "resource_min_size": {"value": 3.0, "type": "float", "range": [1, 20], "help_text": "Minimum conceptual size of a resource."},
                "resource_max_size": {"value": 10.0, "type": "float", "range": [1, 20], "help_text": "Maximum conceptual size of a resource."},
                "resource_min_radius_pb": {"value": 3.0, "type": "float", "range": [1, 20], "help_text": "Physical radius of min size resource."},
                "resource_max_radius_pb": {"value": 12.0, "type": "float", "range": [1, 20], "help_text": "Physical radius of max size resource."},
                "resource_hive_buffer": {"value": 15.0, "type": "float", "range": [0, 100], "help_text": "Minimum distance between resources and hives."},
                "resource_obstacle_buffer": {"value": 10.0, "type": "float", "range": [0, 50], "help_text": "Minimum distance between resources and obstacles."},
                "coop_resource_probability": {"value": 0.3, "type": "float", "range": [0, 1], "help_text": "Probability of a spawned resource being cooperative."}
            },
            "Obstacle Spawning": {
                "help_text": "Settings for obstacle placement.",
                "obstacle_radius_min": {"value": 10.0, "type": "float", "range": [5, 50], "help_text": "Minimum radius of an obstacle."},
                "obstacle_radius_max": {"value": 50.0, "type": "float", "range": [5, 100], "help_text": "Maximum radius of an obstacle."},
                "obstacle_hive_buffer": {"value": 100.0, "type": "float", "range": [0, 300], "help_text": "Minimum distance between obstacles and hives."}
            },
            "Agent Spawning": {
                "help_text": "Settings for agent placement and attribute randomization.",
                "agent_spawn_radius": {"value": 50.0, "type": "float", "range": [10, 150], "help_text": "Radius around the hive where agents can spawn."},
                "agent_mass_strength_influence": {"value": 0.5, "type": "float", "range": [0, 1], "help_text": "Influence of agent strength on its mass."},
                "agent_mass_min_factor": {"value": 0.8, "type": "float", "range": [0.5, 1.5], "help_text": "Minimum mass factor for agents."},
                "agent_mass_max_factor": {"value": 1.2, "type": "float", "range": [0.5, 2.0], "help_text": "Maximum mass factor for agents."}
            }
        },
        "Constants": {
            "help_text": "Core gameplay constants and agent attributes.",
            "overridable": True, 
            "Agent Attributes": {
                "help_text": "Base attributes for agents, can be randomized.",
                "overridable": True,
                "bee_speed": {"value": {"base": 200.0, "rand": 0.2}, "type": "randomizable", "range": [50, 400], "value_type": "float", "overridable": True, "help_text": "Agent's base maximum speed."},
                "agent_radius": {"value": {"base": 3.0, "rand": 0.2}, "type": "randomizable", "range": [2, 10], "value_type": "float", "overridable": True, "help_text": "Agent's physical radius."},
                "agent_base_strength": {"value": {"base": 1.0, "rand": 0.2}, "type": "randomizable", "range": [0.5, 5], "value_type": "float", "overridable": True, "help_text": "Agent's base strength for combat and interactions."},
                "agent_max_energy": {"value": {"base": 100.0, "rand": 0.2}, "type": "randomizable", "range": [50, 200], "value_type": "float", "overridable": True, "help_text": "Agent's maximum energy capacity."},
                "agent_max_health": {"value": {"base": 100.0, "rand": 0.2}, "type": "randomizable", "range": [50, 200], "value_type": "float", "overridable": True, "help_text": "Agent's maximum health."}
            },
            "Agent Energy & Health": {
                "help_text": "Parameters for energy consumption and health recovery.",
                "overridable": True,
                "energy_movement_cost": {"value": 0.00001, "type": "float", "range": [0, 0.001], "overridable": True, "help_text": "Energy cost proportional to movement force."},
                "energy_base_cost": {"value": 0.00001, "type": "float", "range": [0, 0.001], "overridable": True, "help_text": "Base energy cost per step."},
                "energy_grapple_cost_multiplier": {"value": 5.0, "type": "float", "range": [0, 20], "overridable": True, "help_text": "Multiplier for base energy cost while grappling."},
                "energy_recharge_rate": {"value": 0.5, "type": "float", "range": [0, 5], "overridable": True, "help_text": "Energy recharged per step near the hive."},
                "recharge_distance_threshold": {"value": 50, "type": "int", "range": [0, 200], "overridable": True, "help_text": "Distance from hive to start recharging."},
                "agent_health_recovery_rate": {"value": 0.1, "type": "float", "range": [0, 2], "overridable": True, "help_text": "Health recovered per step near the hive."},
                "food_cost_per_agent_health_point": {"value": 0.5, "type": "float", "range": [0, 2], "overridable": True, "help_text": "Hive food needed to restore 1 health point."},
                "food_cost_per_agent_energy_point": {"value": 0.25, "type": "float", "range": [0, 2], "overridable": True, "help_text": "Hive food needed to restore 1 energy point."}
            },
            "Hive Attributes": {
                "help_text": "Base attributes for hives.",
                "overridable": True,
                "hive_max_health": {"value": 100.0, "type": "float", "range": [50, 500], "overridable": True, "help_text": "Maximum health of a hive."},
                "hive_max_food": {"value": 100.0, "type": "float", "range": [50, 500], "overridable": True, "help_text": "Maximum food a hive can store."},
                "hive_delivery_radius": {"value": 42.0, "type": "float", "range": [10, 100], "overridable": True, "help_text": "Radius for resource delivery."},
                "hive_attack_radius": {"value": 42.0, "type": "float", "range": [10, 100], "overridable": True, "help_text": "Radius for hive combat engagements."},
                "hive_damage_factor": {"value": 0.2, "type": "float", "range": [0, 2], "overridable": True, "help_text": "Multiplier for damage dealt to hives."},
                "hive_decay_rate": {"value": 0.02, "type": "float", "range": [0, 1], "overridable": True, "help_text": "Passive health decay rate for hives."},
                "hive_health_decay_if_empty": {"value": 0.03, "type": "float", "range": [0, 1], "overridable": True, "help_text": "Additional health decay if hive has no agents."}
            },
            "Game Rules": {
                "help_text": "General game rules like respawn timers.",
                "respawn_cooldown": {"value": 5, "type": "int", "range": [0, 100], "help_text": "Steps an agent must wait before respawning."},
                "combat_radius": {"value": 9.0, "type": "float", "range": [1, 50], "help_text": "Range at which combat can be initiated."},
                "teammate_death_vicinity_radius_factor": {"value": 1.0, "type": "float", "range": [0, 5], "help_text": "Multiplier of obs_radius for teammate death penalty."}
            }
        },
        "Observations": {
            "help_text": "Settings related to agent observations and memory.",
        "Observation Settings": {
            "help_text": "Toggle different observation components.",
                "sensing_range_fraction": {"value": {"base": 0.05, "rand": 0.2}, "type": "randomizable", "range": [0.01, 0.5], "value_type": "float", "overridable": True, "help_text": "Fraction of arena diagonal for sensing range."},
                "generate_memory_map": {"value": True, "type": "bool", "help_text": "Enable the generation of the persistent 2D grid memory map."},
                "generate_memory_graph": {"value": True, "type": "bool", "help_text": "Enable the generation of the unified GNN memory graph."}
            },
            "Line-of-Sight & Occlusion": {
                "help_text": "Parameters for determining visibility.",
                "use_pybullet_raycasting": {"value": True, "type": "bool", "help_text": "Use PyBullet's raycasting for visibility (more accurate)."},
                "use_gpu_occlusion_in_env": {"value": False, "type": "bool", "help_text": "Use GPU field sampling for visibility (faster but less accurate)."},
                "occlusion_field_resolution": {"value": 64, "type": "int", "range": [16, 256], "help_text": "Resolution of the GPU occlusion field."},
                "num_los_sample_points": {"value": 3, "type": "int", "range": [1, 10], "help_text": "Number of points to check along a line of sight for GPU method."},
                "los_occlusion_threshold": {"value": 0.7, "type": "float", "range": [0, 2], "help_text": "Threshold for blocking line of sight in GPU method."},
                "los_grid_cell_size": {"value": 5.0, "type": "float", "range": [1, 50], "help_text": "Cell size for the CPU occlusion grid fallback."}
            },
            "Graph Memory & Clustering": {
                "help_text": "Settings for the GNN-based persistent memory.",
                "use_batched_memory": {"value": True, "type": "bool", "help_text": "Use the high-performance batched memory manager for all agents."},
                "mid_periphery_scale": {"value": 2.5, "type": "float", "range": [1, 10], "help_text": "Scale of agent's obs_radius to define the 'mid' memory periphery."},
                "mid_cluster_cell_size": {"value": 33.0, "type": "float", "range": [5, 100], "help_text": "Voxel size for clustering mid-periphery nodes."},
                "far_cluster_cell_size": {"value": 100.0, "type": "float", "range": [10, 300], "help_text": "Voxel size for clustering far-periphery nodes."},
                "mem_connection_radius": {"value": 150.0, "type": "float", "range": [10, 500], "help_text": "Radius to connect memory nodes in the final graph."},
                "adaptive_clustering_max_neighbors": {"value": 32, "type": "int", "range": [4, 128], "help_text": "Max neighbors before clustering distant graph nodes."}
            }
        },
        "Rewards": _generate_rewards_config(),
        "Team Settings": {
            "help_text": "Configuration for individual teams.",
            "teams": [
                {"name": "Team 0", "color": [255, 0, 0, 255]},
                {"name": "Team 1", "color": [0, 0, 255, 255]},
                {"name": "Team 2", "color": [0, 255, 0, 255]},
                {"name": "Team 3", "color": [255, 255, 0, 255]},
                {"name": "Team 4", "color": [0, 255, 255, 255]},
                {"name": "Team 5", "color": [255, 0, 255, 255]},
                {"name": "Team 6", "color": [255, 165, 0, 255]},
                {"name": "Team 7", "color": [128, 0, 128, 255]},
                {"name": "Team 8", "color": [0, 128, 128, 255]},
                {"name": "Team 9", "color": [128, 128, 0, 255]}
            ]
        }
    }
    return config

def _generate_rewards_config():
    """Dynamically generates the rewards configuration from REWARD_COMPONENT_KEYS."""
    rewards_config = {
        "help_text": "Configuration for reward components. All rewards can be overridden per-team.",
        "overridable": True,
    }

    # Categorize rewards based on keywords
    categorized_rewards = {
        "Resource Rewards": {},
        "Combat Rewards": {},
        "Grapple Rewards": {},
        "Hive Rewards": {},
        "Discovery & Exploration": {},
        "Penalties": {},
        "Other": {} # Fallback category
    }

    for key in REWARD_COMPONENT_KEYS:
        # Define default values for rewards, assuming most are positive except for penalties
        default_value = 1.0
        reward_range = [-5.0, 5.0] # More reasonable default range for multipliers
        if "lose" in key or "penalty" in key or "death" in key or "controlled" in key:
            default_value = 1.0 # The negative sign is in the base value, so multiplier is positive
        
        # Simple categorization
        category = "Other"
        if "attach" in key or "progress" in key or "delivery" in key:
            category = "Resource Rewards"
        elif "combat" in key:
            category = "Combat Rewards"
        elif "grapple" in key or "torque" in key:
            category = "Grapple Rewards"
        elif "hive" in key:
            category = "Hive Rewards"
        elif "found" in key or "exploration" in key:
            category = "Discovery & Exploration"
        elif "death" in key or "lost" in key:
            category = "Penalties"
            
        categorized_rewards[category][key] = {
            "value": default_value, 
            "type": "reward", 
            "range": reward_range, 
            "overridable": True, 
            "help_text": f"Reward for {key.replace('r_', '').replace('_', ' ')}."
        }

    # Structure it for the GUI
    for category_name, rewards in categorized_rewards.items():
        if rewards: # Only add categories that have rewards
            rewards_config[category_name] = {
                "overridable": True,
                **rewards
            }
            
    return rewards_config
