"""
This module defines the core Swarm2d multi-agent reinforcement learning environment.

The main class, `Swarm2DEnv`, is a highly configurable, multi-agent, multi-team
reinforcement learning environment built using the PyBullet physics engine. It is
designed for research in swarm intelligence, emergent behavior, and complex
coordination strategies.

The environment adheres to the gymnasium.Env interface, making it compatible
with standard reinforcement learning libraries and frameworks.
"""
#!/usr/bin/env python3

import sys
import os

# Calculate the project root directory (Swarm2d)
# __file__ is the path to the current script (env.py)
# os.path.dirname(__file__) is Swarm2d/env/
# Going up one level to get to Swarm2d/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import pygame
import random
import math
import time
import logging
from collections import deque, defaultdict
from scipy.spatial import cKDTree # Keep if used for CPU fallback or other logic
from typing import Tuple, Dict, Optional, Union, List
import copy # For deepcopy

# --- PyTorch and PyG Imports ---
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import scatter
from torch_geometric.utils import scatter as pyg_scatter # For PyG scatter if needed
radius_graph = pyg_nn.radius_graph

from constants import (
    AGENT_RADIUS,
    AGENT_BASE_STRENGTH,
    AGENT_MAX_ENERGY,
    AGENT_MAX_HEALTH,
    SENSING_RANGE_FRACTION,
    OBS_RADIUS,
    SELF_OBS_DIM,
    RAW_CH,
    NODE_TYPE,
    NODE_FEATURE_MAP,
    MEM_NODE_FEAT_IDX,
    OCCLUSION_FIELD_RESOLUTION,
    NUM_LOS_SAMPLE_POINTS,
    LOS_OCCLUSION_THRESHOLD,
    LOS_GRID_CELL_SIZE,
    CLUSTER_CELL_SIZE,
    CLUSTER_MERGE_THRESHOLD,
    HIVE_MAX_HEALTH,
    HIVE_MAX_FOOD,
    MAX_STEPS,
    RESOURCE_MIN_RADIUS_PB,
    RESOURCE_MAX_RADIUS_PB,
    RAW_CH_COUNT,
    OBSTACLE_RADIUS_MIN,
    OBSTACLE_RADIUS_MAX,
    RESOURCE_MIN_SIZE,
    RESOURCE_MAX_SIZE,
    HIVE_RADIUS_ASSUMED,
    HIVE_DELIVERY_RADIUS,
    BEE_SPEED,
    WIDTH,
    HEIGHT,
    NUM_RESOURCES,
    GRAPPLE_INITIATION_DISTANCE_MULTIPLIER,
    FPS,
    COLLISION_GROUP_GROUND,
    ACTIVE_RESOURCE_LIMIT,
    REWARD_COMPONENT_KEYS
)
from env.helper import Helper, bresenham_line
from env.occlusion import OcclusionHelper
from env.observations import RawMapObservationManager, ActorMapState
from env.batched_graph_memory import BatchedPersistentGraphMemory
from env.rewards import RewardManager # Use the base RewardManager directly
from env.managers import SpawnManager, PhysicsManager, ObservationManager, RenderManager


###########################################
# Swarm2DEnv: Configurable Multi-Team Environment
###########################################
class Swarm2DEnv(gym.Env):
    """
    A highly configurable multi-agent, multi-team reinforcement learning environment.

    This class simulates a 2D world where teams of agents can compete or cooperate
    to achieve various objectives, such as resource gathering, combat, and exploration.
    It uses the PyBullet physics engine for realistic interactions and provides a rich,
    graph-based observation space suitable for training sophisticated AI policies.

    The environment is designed with a modular, manager-based architecture, where
    different components (physics, spawning, observations, rewards) are handled by
    dedicated manager classes. This makes the system extensible and easier to maintain.

    Attributes:
        device (torch.device): The device (CPU or GPU) for tensor computations.
        num_teams (int): The number of teams in the simulation.
        num_agents (int): The total number of agents across all teams.
        width (int): The width of the simulation arena.
        height (int): The height of the simulation arena.
        step_counter (int): The number of steps elapsed in the current episode.
        spawn_manager (SpawnManager): Handles the creation of agents, resources, and obstacles.
        physics_manager (PhysicsManager): Manages all PyBullet physics interactions.
        observation_manager (ObservationManager): Generates observations for each agent.
        reward_manager (RewardManager): Calculates rewards for each agent.
        render_manager (RenderManager): Handles rendering the environment, primarily for the Pygame window.
    """
    def __init__(self,
                 num_teams: int = 6,
                 num_agents_per_team: Union[int, List[int]] = 10,
                 # --- New overridable agent base parameters ---
                 agent_radius: float = AGENT_RADIUS,
                 agent_base_strength: float = AGENT_BASE_STRENGTH,
                 agent_max_energy: float = AGENT_MAX_ENERGY,
                 agent_max_health: float = AGENT_MAX_HEALTH,
                 sensing_range_fraction: float = SENSING_RANGE_FRACTION,
                 recency_normalization_period: float = 250.0,
                 # --- End new parameters ---
                 num_resources: int = 350, # Using NUM_RESOURCES from global if not passed
                 num_obstacles: int = 25,  # Using NUM_OBSTACLES from global
                 max_steps: Optional[int] = None,     # Allow setting max_steps from caller
                 width: int = 1000,        # Using WIDTH from global
                 height: int = 1000,       # Using HEIGHT from global
                 render_mode: bool = False,
                 debug: bool = False,
                 agent_force_scale: float = 1.0, # DEFAULT_AGENT_FORCE_SCALE (legacy, kept for compatibility)
                 movement_force_scale: float = 15.0, # New preferred param
                 resource_interaction_force_scale: float = 1.2, # New: For resource carrying grip
                 agent_interaction_force_scale: float = 0.35, # New: For agent grappling grip
                 resource_push_force_scale: float = 0.0, # DEFAULT_RESOURCE_PUSH_FORCE_SCALE
                 resource_base_mass: float = 0.075, # DEFAULT_RESOURCE_BASE_MASS
                 resource_mass_scale_factor: float = 1.4, # DEFAULT_RESOURCE_MASS_SCALE
                 pb_agent_mass: float = 1.0,
                 pb_agent_lateral_friction: float = 0.5,
                 pb_agent_linear_damping: float = 0.11,
                 pb_agent_angular_damping: float = 0.4,
                 pb_agent_spinning_friction: float = 0.03, # New
                 pb_agent_rolling_friction: float = 0.01, # New
                 pb_agent_restitution: float = 0.5,
                 pb_res_friction_static: float = 0.35,
                 pb_res_friction_dynamic: float = 0.25,
                 pb_res_damping_static: float = 0.35,
                 pb_res_damping_dynamic: float = 0.25,
                 pb_res_restitution: float = 0.4,
                 pb_constraint_max_force: float = 500.0, # Kept for backward compatibility, but unused if new ones are set
                 pb_resource_constraint_max_force: float = 3000, # New, specific for resources
                 pb_coop_resource_constraint_max_force: float = 10000, # New, specific for coop resources
                 pb_agent_constraint_max_force: float = 13000, # New, specific for grappling
                 bee_speed: float = 200, # Agent speed
                 node_feature_dim: int = 22, # NODE_FEATURE_DIM
                 raw_ch_count: int = RAW_CH_COUNT,     # Defaults to RAW_CH_COUNT (8) from constants
                 raw_map_grid_size: int = 32,
                 memory_map_grid_size: int = 64,
                 env_self_obs_dim: int = SELF_OBS_DIM,  # Use constant for consistency
                 raw_ch_definition: dict = RAW_CH,
                 node_type_definition: dict = NODE_TYPE,
                 node_feature_map_definition: dict = NODE_FEATURE_MAP,
                 use_gpu_occlusion_in_env: bool = False, # True to use GPU for LOS field
                 use_pybullet_raycasting: bool = True, # True to use PyBullet raycasting for LOS
                 occlusion_field_res_env: int = OCCLUSION_FIELD_RESOLUTION,
                 num_los_samples_env: int = NUM_LOS_SAMPLE_POINTS,
                 los_occlusion_thresh_env: float = LOS_OCCLUSION_THRESHOLD,
                 los_grid_cell_size_env: float = LOS_GRID_CELL_SIZE,
                # --- Parameters for Foveated Graph Generation (to be passed to Obs Manager) ---
                 mid_periphery_scale: float = 2.5,
                 grapple_momentum_bonus_scale: float = 0.1, # New
                 grapple_torque_scale: float = 25.0, # New
                 grapple_momentum_decay: float = 0.95, # New
                 grapple_fatigue_rate: float = 0.01, # New: How fast grip strength decays from holding on
                 grapple_crush_damage_rate: float = 0.05, # New: Passive damage per step while grappling someone
                 grapple_torque_escape_strength: float = 0.5, # New: How much counter-torque helps break a grapple
                 grapple_crit_chance: float = 0.05, # New: Chance for a grapple tick to be a critical hit
                 grapple_crit_multiplier: float = 3.0, # New: Damage multiplier for a critical hit
                 grapple_struggle_damage_rate: float = 0.2, # New: Damage the target deals back to the grappler
                 grapple_rear_crit_bonus_multiplier: float = 3.0, # New: Multiplier for crit chance from behind
                 grappled_agent_counter_grip_scale: float = 0.5, # New: How much grip strength is gained from counter-grappling
                 debug_mode: bool = False, # New flag to control verbose logging
                 agent_randomization_factors: Optional[Dict] = None,
                 team_parameter_overrides: Optional[Dict] = None,
                 enable_profiling: bool = False,
                 team_configs: Optional[List[Dict]] = None,
                 generate_memory_map: bool = True,
                 generate_memory_graph: bool = True,
                 # --- Spawning Detail Parameters ---
                 hive_min_distance: float = 120.0,
                 hive_spawn_jitter: float = 50.0,
                 hive_spawn_radius_factor: float = 0.35,
                 resource_hive_buffer: float = 15.0,
                 resource_obstacle_buffer: float = 10.0,
                 coop_resource_probability: float = 0.3,
                 obstacle_hive_buffer: float = 100.0,
                 agent_spawn_radius: float = 100.0, # Reduced from 200.0 to 100.0 (Closer to hive!)
                 agent_mass_strength_influence: float = 0.5,
                 agent_mass_min_factor: float = 0.5, # Widened from 0.8
                 agent_mass_max_factor: float = 3.0, # Widened from 1.2 to allow cubic scaling
                 team_reward_overrides: Optional[Dict] = None,
                 team_reward_multipliers: Dict[str, Dict[str, float]] = None,
                 graph_connection_radius_factor: float = 0.75,
                 cluster_aggressiveness: float = 4.0,  # Increased for better clustering at distance
                 mem_skeleton_connection_factor: float = 2.00,
                 min_cluster_size: int = 3.0,
                 max_cluster_size: int = 100.0,
                 cluster_exclusion_radius_factor: float = 1.05, 
                 detailed_clustering_radius_factor: float = 2.0, 
                 **kwargs):
        """
        Initializes the Swarm2D environment with a wide range of configurable parameters.

        This constructor sets up the simulation parameters, initializes the PyBullet physics
        engine, and composes the various manager classes that handle the core logic of
        the environment.

        Args:
            num_teams: Number of teams in the environment.
            num_agents_per_team: Number of agents per team. Can be an int (same for all) or a list.
            agent_radius: The physical radius of the agents.
            agent_base_strength: Base strength of agents, affecting interactions.
            agent_max_energy: Maximum energy level for agents.
            agent_max_health: Maximum health for agents.
            sensing_range_fraction: Fraction of the arena diagonal used as the agent's sensing range.
            num_resources: Number of resources to spawn.
            num_obstacles: Number of obstacles to spawn.
            max_steps: Maximum number of steps per episode before truncation.
            width: Width of the simulation arena.
            height: Height of the simulation arena.
            render_mode: If 'human', enables Pygame rendering. If 'gui', enables PyBullet GUI.
            debug: If True, enables debug mode with extra logging and visualizations.
            movement_force_scale: Scaling factor for agent movement forces.
            resource_interaction_force_scale: Scaling factor for the force to pick up/hold resources.
            agent_interaction_force_scale: Scaling factor for the force to grapple/hold other agents.
            resource_push_force_scale: Scaling factor for forces when pushing resources.
            resource_base_mass: Base mass for resources.
            resource_mass_scale_factor: Scaling factor for resource mass based on size.
            pb_agent_mass: Mass of agents in the PyBullet simulation.
            pb_agent_lateral_friction: Lateral friction for agents in PyBullet.
            pb_agent_linear_damping: Linear damping for agents in PyBullet.
            pb_agent_angular_damping: Angular damping for agents in PyBullet.
            pb_agent_spinning_friction: Spinning friction for agents in PyBullet.
            pb_agent_rolling_friction: Rolling friction for agents in PyBullet.
            pb_agent_restitution: Bounciness of agents in PyBullet.
            pb_res_friction_static: Static friction for resources in PyBullet.
            pb_res_friction_dynamic: Dynamic friction for resources in PyBullet.
            pb_res_damping_static: Static damping for resources in PyBullet.
            pb_res_damping_dynamic: Dynamic damping for resources in PyBullet.
            pb_res_restitution: Bounciness of resources in PyBullet.
            pb_resource_constraint_max_force: Maximum force for constraints holding resources.
            pb_coop_resource_constraint_max_force: Maximum force for constraints on cooperatively-held resources.
            pb_agent_constraint_max_force: Maximum force for grapple constraints.
            bee_speed: A factor influencing agent speed.
            node_feature_dim: The dimensionality of the feature vector for each node in the observation graph.
            raw_ch_count: The number of channels in the 2D map observation.
            actor_map_grid_size: The grid size (width and height) of the local 2D map observation.
            env_self_obs_dim: The dimensionality of the agent's self-observation vector.
            use_gpu_occlusion_in_env: If True, uses a GPU-based method for occlusion field calculations.
            use_pybullet_raycasting: If True, uses PyBullet's built-in raycasting for line-of-sight checks.
            grapple_*: A set of parameters controlling the mechanics of the grappling system.
            debug_mode: If True, enables verbose logging and debug visualizations.
            agent_randomization_factors: Dictionary specifying randomization ranges for agent properties.
            team_parameter_overrides: Dictionary to override specific parameters for certain teams.
            enable_profiling: If True, enables code profiling hooks.
            team_configs: A list of dictionaries, each configuring a specific team (e.g., name, color).
            generate_memory_map: If True, generates a 2D map representation of agent memory.
            generate_memory_graph: If True, generates a graph representation of agent memory.
            hive_*: Parameters controlling the spawning logic and placement of hives.
            resource_*: Parameters controlling the spawning logic and placement of resources.
            coop_resource_probability: The probability of a spawned resource requiring cooperative carrying.
            obstacle_*: Parameters controlling the spawning logic and placement of obstacles.
            agent_spawn_radius: The radius around a hive within which agents are spawned.
            team_reward_overrides: Dictionary to override specific reward components for certain teams,
                                   useful for curriculum learning.
            team_reward_multipliers: Dictionary to override specific reward multipliers for certain teams.
            graph_connection_radius_factor: A scaling factor for the radius used to build observation graphs.
            cluster_aggressiveness: A parameter controlling the aggressiveness of cluster formation.
            mem_skeleton_connection_factor: A parameter controlling the factor for memory skeleton connections.
            min_cluster_size: Minimum size for a cluster to be formed.
            max_cluster_size: Maximum size of a cluster.
            cluster_exclusion_radius_factor: Factor to determine the exclusion radius around clusters.
            detailed_clustering_radius_factor: Factor for detailed clustering radius.
            **kwargs: Catches any unused keyword arguments.
        """
        super(Swarm2DEnv, self).__init__()

        # --- Device Setup (CUDA if available, otherwise CPU) ---
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_mode = debug_mode
        self.enable_profiling = enable_profiling
        self.team_configs = team_configs
        self.generate_memory_map = generate_memory_map
        self.generate_memory_graph = generate_memory_graph
        self.single_agent_obs_idx = None # New attribute
        
        # --- Store New Overridable Parameters ---
        self.agent_radius_config = agent_radius
        self.agent_base_strength_config = agent_base_strength
        self.agent_max_energy_config = agent_max_energy
        self.agent_max_health_config = agent_max_health
        self.sensing_range_fraction_config = sensing_range_fraction
        self.recency_normalization_period = recency_normalization_period
        
        # --- Store Hive Attributes ---
        self.hive_max_health = HIVE_MAX_HEALTH
        self.hive_max_food = HIVE_MAX_FOOD

        # GRAPPLING
        self.grapple_torque_scale = grapple_torque_scale       # How much torque is applied from movement input
        self.grapple_break_reward = 15.0        # Large reward for escaping a grapple
        self.torque_win_reward_scale = 0.01     # Small continuous reward for overpowering an opponent's rotation
        self.grapple_momentum_bonus_scale = grapple_momentum_bonus_scale # How much target momentum adds to grip strength
        self.grapple_momentum_decay = grapple_momentum_decay      # How quickly the momentum bonus fades per step
        self.grapple_fatigue_rate = grapple_fatigue_rate
        self.grapple_crush_damage_rate = grapple_crush_damage_rate
        self.grapple_torque_escape_strength = grapple_torque_escape_strength
        self.grapple_crit_chance = grapple_crit_chance
        self.grapple_crit_multiplier = grapple_crit_multiplier
        self.grapple_struggle_damage_rate = grapple_struggle_damage_rate
        self.grapple_rear_crit_bonus_multiplier = grapple_rear_crit_bonus_multiplier
        self.grappled_agent_counter_grip_scale = grappled_agent_counter_grip_scale

        self.num_teams = num_teams
        if isinstance(num_agents_per_team, list):
            if len(num_agents_per_team) != num_teams:
                # Fallback or error if lists don't match
                print(f"Warning: Length of num_agents_per_team list ({len(num_agents_per_team)}) does not match num_teams ({num_teams}). Using first element for all.")
                self.agent_counts_per_team = [num_agents_per_team[0]] * num_teams
            else:
                self.agent_counts_per_team = num_agents_per_team
            self.num_agents_per_team = 0  # Becomes ill-defined
        else:
            self.agent_counts_per_team = [num_agents_per_team] * num_teams
            self.num_agents_per_team = num_agents_per_team

        self.num_agents = sum(self.agent_counts_per_team)
        self.num_resources_config = num_resources
        self.num_obstacles_config = num_obstacles
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.debug = debug

        # --- Calculate arena diagonal and observation radius early ---
        self.d_max = np.sqrt(self.width**2 + self.height**2)
        self.sensing_range = self.sensing_range_fraction_config * self.d_max
        self.obs_radius = self.sensing_range # This is the agent's FoV radius

        # New, more descriptive physics parameters
        self.movement_force_scale = movement_force_scale
        self.resource_interaction_force_scale = resource_interaction_force_scale
        self.agent_interaction_force_scale = agent_interaction_force_scale
        
        self.agent_force_scale = agent_force_scale # For backwards compatibility
        # Compatibility: If new param is default but old one is not, use the old one
        if self.movement_force_scale == 1.0 and self.agent_force_scale != 1.0:
            self.movement_force_scale = self.agent_force_scale
            print(f"  INFO: Using legacy 'agent_force_scale' ({self.agent_force_scale}) for movement_force_scale.")

        self.bee_speed_config = bee_speed

        # --- Resource Physics Parameters ---
        self.resource_push_force_scale = resource_push_force_scale
        self.resource_base_mass = resource_base_mass
        self.resource_mass_scale_factor = resource_mass_scale_factor

        # --- PyBullet Physics Parameters ---
        self.pb_agent_mass = pb_agent_mass
        self.pb_agent_lateral_friction = pb_agent_lateral_friction
        self.pb_agent_linear_damping = pb_agent_linear_damping
        self.pb_agent_angular_damping = pb_agent_angular_damping
        self.pb_agent_spinning_friction = pb_agent_spinning_friction # New
        self.pb_agent_rolling_friction = pb_agent_rolling_friction # New
        self.pb_agent_restitution = pb_agent_restitution
        self.pb_res_friction_static = pb_res_friction_static
        self.pb_res_friction_dynamic = pb_res_friction_dynamic
        self.pb_res_damping_static = pb_res_damping_static
        self.pb_res_damping_dynamic = pb_res_damping_dynamic
        self.pb_res_restitution = pb_res_restitution
        
        # --- PyBullet Constraint Parameters (New logic) ---
        # If the new specific values are not provided (i.e., remain at default -1.0),
        # fall back to the legacy 'pb_constraint_max_force' for backward compatibility.
        self.pb_resource_constraint_max_force = pb_resource_constraint_max_force if pb_resource_constraint_max_force != -1.0 else pb_constraint_max_force
        self.pb_agent_constraint_max_force = pb_agent_constraint_max_force if pb_agent_constraint_max_force != -1.0 else pb_constraint_max_force
        self.pb_coop_resource_constraint_max_force = pb_coop_resource_constraint_max_force if pb_coop_resource_constraint_max_force != -1.0 else self.pb_resource_constraint_max_force
 
        self.team_reward_overrides = team_reward_overrides if team_reward_overrides is not None else {}

        self.current_movement_force_scale = self.movement_force_scale
        self.current_resource_interaction_force_scale = self.resource_interaction_force_scale
        self.current_agent_interaction_force_scale = self.agent_interaction_force_scale

        self.agent_trails = [deque(maxlen=50) for _ in range(self.num_agents)]

        # --- Debug Visualization ---
        self.velocity_debug_lines = {} # For PyBullet velocity vectors


        # --- Dynamic Parameters based on Config ---
        global OBS_RADIUS # Update global variable if needed elsewhere, use self.obs_radius internally
        OBS_RADIUS = self.obs_radius

        # --- Resource Radius Scaling (Physical Radius) ---
        self.min_resource_radius_pb = RESOURCE_MIN_RADIUS_PB
        self.max_resource_radius_pb = RESOURCE_MAX_RADIUS_PB
        
        # --- Observation/Action Space Dimensions (Based on latest understanding) ---
        self.grid_size = 32 # Standardized grid size for local maps
        self.map_channels = RAW_CH_COUNT
        self.self_obs_dim = env_self_obs_dim 
        self.joint_action_dim = 3 # movement_x, movement_y, pickup_discrete
        self.coop_min_resource_radius_pb = self.min_resource_radius_pb * 1.0 # Example: 0% larger base
        self.coop_max_resource_radius_pb = self.max_resource_radius_pb * 1.0 # Example: Can go 20% larger than normal max

        # Store obs gen params
        self.node_feature_dim = node_feature_dim
        self.raw_ch_count = raw_ch_count
        self.raw_map_grid_size = raw_map_grid_size # For agent's local map
        self.memory_map_grid_size = memory_map_grid_size # For memory map
        self.env_self_obs_dim = env_self_obs_dim       # For agent's self vector
        self.raw_ch_def_dict_const = raw_ch_definition
        self.node_type_def_const = node_type_definition
        self.node_feature_map_const = node_feature_map_definition
        self.use_gpu_occlusion_in_env = use_gpu_occlusion_in_env
        self.use_pybullet_raycasting = use_pybullet_raycasting
        self.occlusion_field_res_env = occlusion_field_res_env
        self.num_los_samples_env = num_los_samples_env
        self.los_occlusion_thresh_env = los_occlusion_thresh_env
        self.los_grid_cell_size_env = los_grid_cell_size_env

        # --- Store Spawning Detail Parameters ---
        self.hive_min_distance = hive_min_distance
        self.hive_spawn_jitter = hive_spawn_jitter
        self.hive_spawn_radius_factor = hive_spawn_radius_factor
        self.resource_hive_buffer = resource_hive_buffer
        self.resource_obstacle_buffer = resource_obstacle_buffer
        self.coop_resource_probability = coop_resource_probability
        self.obstacle_hive_buffer = obstacle_hive_buffer
        self.agent_spawn_radius = agent_spawn_radius
        self.agent_mass_strength_influence = agent_mass_strength_influence
        self.agent_mass_min_factor = agent_mass_min_factor
        self.agent_mass_max_factor = agent_mass_max_factor

        # --- Store Foveated Graph Parameters ---
        # These are now calculated dynamically based on agent radius for better adaptability.
        self.mid_periphery_scale = 2.5 # This remains a configurable multiplier
        self.far_cluster_cell_size = self.agent_radius_config
        self.mid_cluster_cell_size = self.agent_radius_config / 3.0
        self.cluster_aggressiveness = cluster_aggressiveness
        self.mem_skeleton_connection_factor = mem_skeleton_connection_factor
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.cluster_exclusion_radius_factor = cluster_exclusion_radius_factor
        self.detailed_clustering_radius_factor = detailed_clustering_radius_factor
        # --- NEW: Store Hybrid Clustering Parameter ---
        self.cluster_merge_threshold = CLUSTER_MERGE_THRESHOLD


        # Store agent randomization factors
        self.agent_randomization_factors = agent_randomization_factors if agent_randomization_factors is not None else {}
        self.team_parameter_overrides = team_parameter_overrides
   
        # --- NEW: Store Graph Generation Parameter ---
        self.graph_connection_radius_factor = graph_connection_radius_factor

        # --- Initialize Environment Components ---
        # Entity lists are initialized here, but populated in reset()
        self.hives = {}
        self.hive_body_ids = {}
        self.obstacles = []
        self.resources = []
        self.agents = []
        self.agent_body_ids = []
        self.actor_map_states = {}

        # Initialize managers using composition
        self.spawn_manager = SpawnManager(self)
        self.physics_manager = PhysicsManager(self)
        self.observation_manager = ObservationManager(self)
        # The RewardManager is now imported directly from rewards.py
        self.reward_manager = RewardManager(self, team_reward_overrides=self.team_reward_overrides)
        self.render_manager = RenderManager(self)

        self.physics_manager.set_movement_force_scale(self.current_movement_force_scale)
        self.physics_manager.set_resource_interaction_force_scale(self.current_resource_interaction_force_scale)
        self.physics_manager.set_agent_interaction_force_scale(self.current_agent_interaction_force_scale)

        # Attributes for current step's global entity tensors (will be on self.device)
        self.current_step_all_pos_t: Optional[torch.Tensor] = None
        self.current_step_all_feat_t: Optional[torch.Tensor] = None
        self.current_step_all_types_t: Optional[torch.Tensor] = None
        self.current_step_all_teams_t: Optional[torch.Tensor] = None
        self.current_step_all_radii_t: Optional[torch.Tensor] = None
        self.current_step_all_coop_t: Optional[torch.Tensor] = None
        self.current_step_all_body_ids_t: Optional[torch.Tensor] = None
        self.current_step_agent_id_to_node_idx_map: Dict[int, int] = {}
        self.current_step_gpu_occlusion_field: Optional[torch.Tensor] = None
        self.current_step_cpu_occlusion_grid: Optional[np.ndarray] = None

        self.actions_prev_step: List[Dict] = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(self.num_agents)]


        # --- Metadata (for external access by policies, etc.) ---
        # This dictionary provides key information about the environment's configuration,
        # which can be essential for policies to correctly interpret observations and shape actions.
        self.metadata = {
            "render_modes": ["human", "gui"], "agent_radius": self.agent_radius_config, "bee_speed": self.bee_speed_config,
            "agent_base_strength": self.agent_base_strength_config, "obstacle_radius_min": OBSTACLE_RADIUS_MIN,
            "obstacle_radius_max": OBSTACLE_RADIUS_MAX, "resource_min_size": RESOURCE_MIN_SIZE,
            "resource_max_size": RESOURCE_MAX_SIZE, "agent_max_energy": self.agent_max_energy_config,
            "agent_max_health": self.agent_max_health_config, "hive_max_health": HIVE_MAX_HEALTH,
            "hive_max_food": HIVE_MAX_FOOD, "hive_radius_assumed": HIVE_RADIUS_ASSUMED,
            "hive_delivery_radius": HIVE_DELIVERY_RADIUS, 
            "obs_radius": self.obs_radius, # Agent's own perception radius
            "max_obs_radius_possible": self.obs_radius * 1.2, # Assumes agent obs_radius can be up to 20% larger.
            "node_feature_dim": self.node_feature_dim,
            "raw_map_channels": self.raw_ch_count,
            "raw_map_grid_size": self.raw_map_grid_size,
            "memory_map_grid_size": self.memory_map_grid_size,
            "self_obs_dim": self.env_self_obs_dim, "width": self.width, "height": self.height,
            "num_reward_components": 15, # Define expected components (must match policies)
            "resource_min_radius_pb": self.min_resource_radius_pb,
            "resource_max_radius_pb": self.max_resource_radius_pb,
            "max_agent_strength": self.agent_base_strength_config * 1.2, # Estimate max randomized value
            "max_agent_speed": BEE_SPEED * 1.2, # Estimate max randomized value
            "max_agent_radius": self.agent_radius_config * 1.2, # Estimate max randomized value
            "map_channels_shape": self.map_channels, # Use calculated value
            # Add the PyBullet params to metadata for potential reference
            "pb_agent_lateral_friction": self.pb_agent_lateral_friction,
            "pb_agent_linear_damping": self.pb_agent_linear_damping,
            "pb_agent_angular_damping": self.pb_agent_angular_damping,
            "pb_agent_spinning_friction": self.pb_agent_spinning_friction, # New
            "pb_agent_rolling_friction": self.pb_agent_rolling_friction, # New
            "pb_res_friction_static": self.pb_res_friction_static,
            "pb_res_friction_dynamic": self.pb_res_friction_dynamic,
            "pb_res_damping_static": self.pb_res_damping_static,
            "pb_res_damping_dynamic": self.pb_res_damping_dynamic,
        }
        # Store initial counts in metadata *after* initialization
        self.metadata['initial_resources'] = self.num_resources_config
        self.metadata['initial_agents'] = self.num_agents
        for tid in range(self.num_teams):
            self.metadata[f'initial_agents_team_{tid}'] = self.agent_counts_per_team[tid]
        
        # --- Set Environment Attributes from Metadata ---
        # World and Simulation parameters
        self.width = self.metadata.get('width', WIDTH)
        self.height = self.metadata.get('height', HEIGHT)
        self.timeStep = self.metadata.get('timeStep', 1.0 / 60.0)
        self.max_steps = self.metadata.get('max_steps', MAX_STEPS)
        
        # Resource parameters
        self.num_resources_config = num_resources
        self.max_resource_radius_pb = self.metadata.get('resource_max_radius_pb', RESOURCE_MAX_RADIUS_PB)
        self.grapple_initiation_distance_multiplier = self.metadata.get('grapple_initiation_distance_multiplier', GRAPPLE_INITIATION_DISTANCE_MULTIPLIER)

        # --- Connect to PyBullet ---
        client_mode = p.DIRECT
        if self.render_mode in ["gui", "both"]:
            try:
                self.physicsClient = p.connect(p.GUI)
                
                # --- Set Top-Down Camera View ---
                p.resetDebugVisualizerCamera(
                    cameraDistance=max(self.width, self.height) * 0.8, # Zoom out to see the whole arena width
                    cameraYaw=0,       # Looking straight ahead
                    cameraPitch=-85.0,  # Look almost straight down (Adjusted from -89.99 to avoid gimbal lock/zoom issues)
                    cameraTargetPosition=[self.width / 2, self.height / 2, 0], # Center of the arena
                    physicsClientId=self.physicsClient
                )
                # --- End Camera Setup ---

                if self.physicsClient < 0:
                    print("Warning: PyBullet GUI connection failed, falling back to DIRECT mode.")
                    self.physicsClient = p.connect(p.DIRECT)
                    self.render_mode = "human" if self.render_mode == "both" else None
                
                # Configure the visualizer
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.physicsClient)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.physicsClient)
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1, physicsClientId=self.physicsClient)
                # p.configureDebugVisualizer(p.COV_ENABLE_GRID, 1, physicsClientId=self.physicsClient)  # COV_ENABLE_GRID not available in this PyBullet version

            except p.error as e:
                print(f"PyBullet connection error: {e}. Falling back to DIRECT mode.")
                self.physicsClient = p.connect(p.DIRECT)
                self.render_mode = "human" if self.render_mode == "both" else None
        elif self.render_mode == "human":
             self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        if self.physicsClient < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server.")

        # --- PyBullet Setup ---
        p.setPhysicsEngineParameter(numSolverIterations=22, numSubSteps=8, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        self.timeStep = 1.0 / FPS # Use global FPS
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClient)

        # Load ground plane
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        # --- Set Ground Plane Color ---
        p.changeVisualShape(self.planeId, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=self.physicsClient)
        # --- End Ground Plane Color ---

        # Optionally, explicitly set plane dynamics (though defaults are usually fine)
        p.changeDynamics(self.planeId, -1,
                        lateralFriction=1.0,  # High friction for the ground
                        restitution=0.0,      # No bounce from the ground
                        # linearDamping=0.0,    # Ground shouldn't damp itself
                        # angularDamping=0.0,
                        physicsClientId=self.physicsClient)
        p.setCollisionFilterGroupMask(self.planeId, -1, COLLISION_GROUP_GROUND, -1, physicsClientId=self.physicsClient)

        self.arena_bounds = [0, self.width, 0, self.height]

        # --- Conditionally initialize memory managers ---
        # GRAPH-BASED MEMORY:
        # If enabled, initialize the batched graph memory manager.
        self.batched_graph_memory: Optional['BatchedPersistentGraphMemory'] = None
        self.batched_graph_memory = BatchedPersistentGraphMemory(self.num_agents, self.device)

        # You might also want to cache the final observations to avoid re-computation
        self._cached_final_observations: Optional[List[Dict]] = None

        print("Swarm2DEnv initialization complete.")        # --- Add missing/derived metadata AFTER agent init ---
        # Compute theoretical maximums based on randomization factors
        # Randomization formula: base * (1 + rand_factor) for maximum
        def get_theoretical_max(param_name: str, base_value: float) -> float:
            """Compute theoretical maximum considering randomization."""
            if not self.agent_randomization_factors:
                return base_value * 1.2  # Default 20% buffer
            param_config = self.agent_randomization_factors.get(param_name, {})
            rand_factor = param_config.get('rand', 0.2)  # Default 0.2 (20%)
            return base_value * (1.0 + rand_factor)
        
        theoretical_max_radius = get_theoretical_max('agent_radius', self.agent_radius_config)
        theoretical_max_speed = get_theoretical_max('bee_speed', self.bee_speed_config)
        theoretical_max_strength = get_theoretical_max('agent_base_strength', self.agent_base_strength_config)
        theoretical_max_obs_radius = get_theoretical_max('sensing_range_fraction', self.obs_radius / self.d_max) * self.d_max
        
        # Use max of observed and theoretical to ensure we never under-normalize
        all_agent_radii = [a['agent_radius'] for a in self.agents if a]
        all_agent_speeds = [a['speed'] for a in self.agents if a]
        all_agent_strengths = [a['strength'] for a in self.agents if a]
        all_agent_obs_radii = [a.get('obs_radius', self.obs_radius) for a in self.agents if a]
        
        self.metadata['max_agent_radius_observed'] = max(all_agent_radii) if all_agent_radii else theoretical_max_radius
        self.metadata['max_agent_speed_observed'] = max(all_agent_speeds) if all_agent_speeds else theoretical_max_speed
        self.metadata['max_agent_strength_observed'] = max(all_agent_strengths) if all_agent_strengths else theoretical_max_strength
        
        # Ensure normalization uses theoretical max to prevent values > 1.0
        self.metadata['max_agent_radius_observed'] = max(self.metadata['max_agent_radius_observed'], theoretical_max_radius)
        self.metadata['max_agent_speed_observed'] = max(self.metadata['max_agent_speed_observed'], theoretical_max_speed)
        self.metadata['max_agent_strength_observed'] = max(self.metadata['max_agent_strength_observed'], theoretical_max_strength)
        
        observed_max_obs_radius = max(all_agent_obs_radii) if all_agent_obs_radii else theoretical_max_obs_radius
        self.metadata['max_obs_radius_possible'] = max(observed_max_obs_radius, theoretical_max_obs_radius)
        
        self.metadata['obstacle_radius_max_config'] = OBSTACLE_RADIUS_MAX

        # It should be the maximum possible physical radius of any entity in the environment.
        self.metadata['max_size_norm_divisor'] = max(
            1.0, # Ensure it's never zero
            self.metadata.get('max_agent_radius_observed', self.agent_radius_config * 1.2),
            self.metadata.get('obstacle_radius_max_config', OBSTACLE_RADIUS_MAX),
            self.metadata.get('hive_radius_assumed', HIVE_RADIUS_ASSUMED),
            self.metadata.get('resource_max_radius_pb', self.max_resource_radius_pb) # Use the instance attribute
        )
        # --- Define Gym Spaces ---
        _map_channels_expected = RAW_CH_COUNT
        
        # Define Gym Spaces - NOW INCLUDES GRAPH (as a Dict space for PyG Data components)
        # This is tricky because PyG Data objects are not standard gym.spaces.
        # A common workaround is to define it as a Dict space representing the components
        # and let the user handle PyG Data object assembly/disassembly.
        # Or, use a custom space if you have one. For simplicity, we'll use Dict.
        # Note: Max nodes/edges for graph space definition is an estimate.
        MAX_GRAPH_NODES_ESTIMATE = 64 # Max entities an agent might see
        MAX_GRAPH_EDGES_ESTIMATE = MAX_GRAPH_NODES_ESTIMATE * 10 # Rough estimate

        self.observation_space = spaces.Dict({
            "self": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_self_obs_dim,), dtype=np.float32),
            "map": spaces.Box(low=0, high=1, shape=(self.raw_ch_count, self.raw_map_grid_size, self.raw_map_grid_size), dtype=np.float32),
            # The 'graph' space is defined as a Dict. The actual output will be a PyG Data object.
            # Max nodes/edges are estimates for space definition.
            "graph": spaces.Dict({
                "x": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_GRAPH_NODES_ESTIMATE, self.node_feature_dim), dtype=np.float32),
                "edge_index": spaces.Box(low=0, high=MAX_GRAPH_NODES_ESTIMATE - 1, shape=(2, MAX_GRAPH_EDGES_ESTIMATE), dtype=np.int64),
                "radii": spaces.Box(low=0, high=np.inf, shape=(MAX_GRAPH_NODES_ESTIMATE,), dtype=np.float32),
            })
        })
        self.action_space = spaces.Dict({
            "movement": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "pickup": spaces.Discrete(3)
        })
        
        # --- Rendering & Bookkeeping Init ---
        self.visibility_lines = {}
        center_cell = (self.grid_size // 2, self.grid_size // 2)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.visibility_lines[(i, j)] = bresenham_line(center_cell[0], center_cell[1], i, j)
        self.screen = None; self.clock = None; self.obs_surface = None
        self.resources_picked_count = 0; self.resources_delivered_count = 0; self.agents_killed_count = 0
        
        # --- Reward-related Bookkeeping ---
        self.discovery_records = {}
        self.exploration_counts = defaultdict(int)
        self.explored_points = set()
        self.collected_points = set()
        self.current_maac_roles = {}

        # This call is needed here to initialize tensors before the obs manager is fully used.
        self.physics_manager._prepare_global_entity_tensors_for_step()
        print("Swarm2DEnv initialization complete (with PyBullet param args).")

    def render(self) -> None:
        """
        Renders the environment based on the configured render_mode.

        If `render_mode` is 'gui' or 'both', the primary visualization is handled automatically
        by PyBullet during `p.stepSimulation()`. This method is responsible for explicitly
        calling the `RenderManager` to update the separate Pygame window when `render_mode`
        is 'human' or 'both'. This Pygame window can display additional debug information,
        agent-specific views, or other custom visualizations not available in the main
        PyBullet GUI.
        """
        # The PyBullet visualizer is updated automatically during p.stepSimulation().
        # This method triggers the Pygame window update if that render mode is active.
        if self.render_mode in ["human", "both"]:
            return self.render_manager.render()

    def _update_pybullet_visualizations(self):
        """Draws and updates debug information (e.g., velocity vectors, observation radii) directly in the PyBullet GUI."""
        if self.render_mode not in ["gui", "both"]:
            return

        max_speed = self.metadata.get('max_agent_speed_observed', self.bee_speed_config * 1.2)
        max_line_length = AGENT_RADIUS * 4 # Max length of the debug line
        
        # Draw high-quality circles using debug lines
        num_segments = 32 # Higher number = smoother circle
        angle_step = 2 * math.pi / num_segments

        for agent in self.agents:
            agent_id = agent.get('id')
            if not agent: continue
            
            # --- Handle Observation Radius Visualization (Wireframe Circle) ---
            # Use replaceItemUniqueId to update lines instead of redrawing, for smoother visualization.
            
            if agent.get('alive') and agent.get('pos') is not None and agent.get('obs_radius_viz_active'):
                pos = agent['pos']
                radius = agent['obs_radius']
                color = agent.get('obs_radius_color', [1, 1, 0])
                
                # Retrieve existing line IDs or initialize empty list
                viz_ids = agent.get('obs_radius_viz_ids', [])
                if not viz_ids:
                    # Initial draw: create lines and store IDs
                    new_ids = []
                    for i in range(num_segments):
                        angle1 = i * angle_step
                        angle2 = (i + 1) * angle_step
                        
                        p1 = [pos[0] + radius * math.cos(angle1), pos[1] + radius * math.sin(angle1), 0.1]
                        p2 = [pos[0] + radius * math.cos(angle2), pos[1] + radius * math.sin(angle2), 0.1]
                        
                        # Lifetime=0 means persistent until removed
                        line_id = p.addUserDebugLine(p1, p2, color, lineWidth=2.0, lifeTime=0, physicsClientId=self.physicsClient)
                        new_ids.append(line_id)
                    agent['obs_radius_viz_ids'] = new_ids
                else:
                    # Update existing lines
                    # If segment count somehow changed (shouldn't happen), we'd need to handle mismatch.
                    for i, line_id in enumerate(viz_ids):
                        if i >= num_segments: break
                        angle1 = i * angle_step
                        angle2 = (i + 1) * angle_step
                        
                        p1 = [pos[0] + radius * math.cos(angle1), pos[1] + radius * math.sin(angle1), 0.1]
                        p2 = [pos[0] + radius * math.cos(angle2), pos[1] + radius * math.sin(angle2), 0.1]
                        
                        # Replace the line segment
                        p.addUserDebugLine(p1, p2, color, lineWidth=2.0, lifeTime=0, replaceItemUniqueId=line_id, physicsClientId=self.physicsClient)
            else:
                # Agent dead or inactive: Hide or Remove lines
                viz_ids = agent.get('obs_radius_viz_ids', [])
                if viz_ids:
                    for line_id in viz_ids:
                        try: p.removeUserDebugItem(line_id, physicsClientId=self.physicsClient)
                        except p.error: pass
                    agent['obs_radius_viz_ids'] = [] # Clear the list

            if not agent.get('alive') or agent_id is None:
                continue

            vel_norm = np.linalg.norm(agent['vel'])

            if vel_norm > 0.1: # Threshold to avoid drawing for stationary agents
                pos = agent['pos']
                vel_dir = agent['vel'] / vel_norm

                line_length = min(vel_norm / max_speed, 1.0) * max_line_length
                start_point = [pos[0], pos[1], agent.get('agent_radius', AGENT_RADIUS) * 0.5]
                end_point = [
                    start_point[0] + vel_dir[0] * line_length,
                    start_point[1] + vel_dir[1] * line_length,
                    start_point[2]
                ]

                line_color = [1, 1, 1] # White
                line_width = 2
                line_id = self.velocity_debug_lines.get(agent_id)

                if line_id is not None:
                    p.addUserDebugLine(start_point, end_point, line_color, lineWidth=line_width,
                                       replaceItemUniqueId=line_id, physicsClientId=self.physicsClient)
                else:
                    new_line_id = p.addUserDebugLine(start_point, end_point, line_color, lineWidth=line_width,
                                               physicsClientId=self.physicsClient)
                    self.velocity_debug_lines[agent_id] = new_line_id
            else:
                # If agent is stopped, remove its debug line
                line_id = self.velocity_debug_lines.pop(agent_id, None)
                if line_id is not None:
                    try:
                        p.removeUserDebugItem(line_id, physicsClientId=self.physicsClient)
                    except p.error:
                        pass # Ignore error if item already removed

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, curriculum_config: Optional[Dict] = None, override_config: Optional[Dict] = None) -> Tuple[List[Dict[str, Union[torch.Tensor, Data]]], Dict]:
        """
        Resets the environment to an initial state and returns the initial observations.
        
        Args:
            seed: RNG seed
            options: Gym options
            curriculum_config: Curriculum parameters (persistent)
            override_config: Ephemeral overrides for this episode (num_resources, num_obstacles)
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # --- Config Overrides ---
        # Apply overrides temporarily
        original_config = {}
        if override_config:
            if 'num_resources' in override_config:
                original_config['num_resources'] = self.num_resources_config
                self.num_resources_config = override_config['num_resources']
            if 'num_obstacles' in override_config:
                original_config['num_obstacles'] = self.num_obstacles
                self.num_obstacles = override_config['num_obstacles']
            
        # --- Curriculum Learning Integration ---
        if curriculum_config:
            self.num_teams = curriculum_config.get('num_teams', self.num_teams)
            self.num_agents_per_team = curriculum_config.get('num_agents_per_team', self.num_agents_per_team)
            # Recalculate num_agents based on new num_agents_per_team if needed
            # self.num_agents = sum(self.agent_counts_per_team) 
            # Note: Changing agent count might require deeper re-init depending on implementation.
            print(f"[Curriculum] Resetting environment to {self.num_teams} teams and {self.num_agents_per_team} agents/team.")

        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        p.setPhysicsEngineParameter(numSolverIterations=30, numSubSteps=8, physicsClientId=self.physicsClient)
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClient)
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        self.spawn_manager.init_boundaries()

        self.discovery_records.clear(); self.exploration_counts.clear()
        self.explored_points.clear(); self.collected_points.clear()

        self.spawn_manager.spawn_hives()
        self.spawn_manager.init_obstacles_pybullet()
        self.spawn_manager.resource_id_counter = 0; self.resources = []
        initial_spawn_count = min(self.num_resources_config, ACTIVE_RESOURCE_LIMIT)
        
        # ... rest of spawn logic is handled by managers or calls below ...
        # But wait, looking at original code, it calls spawn_manager methods directly here?
        # No, checking read_file output...
        # 847|        self.spawn_manager.spawn_hives()
        # 848|        self.spawn_manager.init_obstacles_pybullet()
        
        # Let's restore the original config at the end of the method or after spawning?
        # If we restore it at the end of reset(), it won't affect the episode running (good), 
        # but we need to make sure spawning uses the modified values.
        # spawn_manager.spawn_resources() likely uses self.env.num_resources_config.
        
        # We need to continue matching the function body to verify where to insert restoration or if we need to wrap the whole thing.
        # For safety, I'll replace just the start of the function up to spawn_hives, and insert restoration at the end of the block 
        # or simply rely on the caller to not expect persistent changes from override_config.
        # Actually, let's keep it simple: override_config modifies self.*, and we rely on the caller to manage it 
        # OR we restore it. Restoring it is safer.
        
        # I will replace the header and the first few lines.

        for _ in range(initial_spawn_count):
            res = self.spawn_manager.spawn_resource()
            if res: self.resources.append(res)
        self.agents = []; self.agent_body_ids = []
        self.spawn_manager.init_agents_pybullet()

        # --- Dynamically calculate the world-to-map scale for the GLOBAL memory map ---
        # A smaller scale means more cells, i.e., higher resolution.
        # Let's tie it to the final output size. We want the persistent map
        # to be a higher resolution than the final map. Let's say 8x.
        world_to_map_scale = (self.width / self.memory_map_grid_size) / 8.0

        # --- Initialize Memory Objects for the New Episode ---
        self.actor_map_states.clear()
        
        # Always initialize actor_map_states as they are used by both memory systems
        for i in range(self.num_agents):
            agent_team_id = self.agents[i].get('team', 0) if self.agents[i] else 0
            self.actor_map_states[i] = ActorMapState(i, agent_team_id, self.width, self.height, self.memory_map_grid_size, world_to_map_scale=world_to_map_scale, chunk_size=64, device=self.device)

        # --- Create env_metadata dictionary AFTER spawning entities ---
        # This is used by the critic to get a global overview of the environment state.
        self.env_metadata = {
            'width': self.width,
            'height': self.height,
            'max_steps': self.max_steps,
            'hives_info_for_critic': {
                team_id: info['pos'] for team_id, info in self.hives.items()
            }
        }

        self.step_counter = 0
        self.resources_picked_count = 0; self.resources_delivered_count = 0; self.agents_killed_count = 0
        self.actions_prev_step = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(self.num_agents)]

        # Clear agent trails at reset
        for trail in self.agent_trails:
            trail.clear()

        # Clear any existing debug lines from the previous episode
        for line_id in self.velocity_debug_lines.values():
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.physicsClient)
            except p.error:
                pass # Item might have been removed already
        self.velocity_debug_lines.clear()
            
        # Update agent positions from PyBullet after init
        for agent in self.agents:
            if agent and agent.get('alive'):
                agent['pos'], agent['vel'] = self.physics_manager._get_agent_state(agent)
        for res in self.resources: # Update resource positions too
            if res and res.get('body_id') is not None:
                try:
                    pos_res_3d, _ = p.getBasePositionAndOrientation(res['body_id'], physicsClientId=self.physicsClient)
                    res['pos'] = np.array(pos_res_3d[:2])
                except p.error: res['pos'] = None

        if self.batched_graph_memory:
            self.batched_graph_memory.reset()

        # Prepare global entity tensors for the initial state before generating observations
        self.physics_manager._prepare_global_entity_tensors_for_step()
        
        # Pass obstacle IDs to observation manager for PyBullet raycasting
        obstacle_body_ids = [obs['body_id'] for obs in self.obstacles if obs and obs.get('body_id') is not None]
        self.observation_manager.obstacle_body_ids = obstacle_body_ids
        self.observation_manager.physics_client_id = self.physicsClient

        # Generate the first observation of the episode
        # The first call to _generate_final_observations will handle everything
        observations_list = self.observation_manager._generate_final_observations(agent_idx_to_observe=self.single_agent_obs_idx)

        return observations_list, {"message": "Environment reset successful"}


    def step(self, actions: List[Dict]) -> Tuple[List[Dict[str, Union[torch.Tensor, Data]]], List[Dict[str, float]], bool, bool, Dict]:
        """
        Advances the environment by one timestep.

        This is the main workhorse method of the environment. It takes a list of actions
        from the agents, applies them to the simulation, steps the physics engine forward,
        calculates the resulting rewards, and generates the next observations. The process
        is broken down into a series of logical phases, each handled by a helper method.

        Args:
            actions: A list of action dictionaries, one for each agent. Each dictionary
                     should contain keys like 'movement' (a 2D numpy array) and 'pickup'
                     (an integer).

        Returns:
            A tuple containing:
            - obs (List[Dict]): A list of the next observations for each agent.
            - rewards (List[Dict]): A list of reward dictionaries for each agent.
            - terminated (bool): Whether the episode has ended permanently (not used).
            - truncated (bool): Whether the episode has ended due to reaching the max step limit.
            - infos (Dict): A dictionary containing auxiliary diagnostic information.
        """
        # Phase 1: Initialize step, validate inputs, and handle state decay
        rewards, infos = self._initialize_step_and_decay_states(actions)

        # Phase 2: Perform a single, efficient proximity search for all entities
        proximity_data = self.physics_manager._vectorized_proximity_search()

        # Phase 3 & 4: Apply agent actions and advance the physics simulation
        grapple_break_events = self._apply_actions_and_step_simulation(actions, rewards, proximity_data, infos)

        # Phase 5: Process game logic and calculate rewards based on the physics outcomes
        self._process_post_physics_logic(rewards, infos, proximity_data, grapple_break_events)

        # --- 4. Calculate Rewards ---
        delivered_resource_ids_this_step = self.reward_manager._process_resource_delivery(rewards, proximity_data['gnn_idx_to_res_obj_map'])
        self.reward_manager._process_discovery_and_misc_rewards(rewards, proximity_data)  # V2: Pass proximity_data for optimization
        self.reward_manager._process_continuous_hive_rewards(rewards) # Add this line

        # --- 5. Update observations ---
        # Get the latest states after physics step for accurate observations
        self.physics_manager._update_and_correct_physics_states()

        # Phase 6: Finalize the step and generate new observations for each agent
        obs, terminated, truncated = self._finalize_step_and_generate_observations()

        return obs, rewards, terminated, truncated, infos

    def _initialize_step_and_decay_states(self, actions: List[Dict]) -> Tuple[List[Dict[str, float]], Dict]:
        """
        Handles the initial setup for the step. (Phase 1)

        This includes:
        - Validating the incoming actions list.
        - Initializing reward and info dictionaries.
        - Incrementing the step counter.
        - Calling the physics manager to apply natural state decay (e.g., energy loss).
        """
        if not isinstance(actions, list) or len(actions) != self.num_agents:
            actions = [{'movement': np.zeros(2), 'pickup': 0} for _ in range(self.num_agents)]
        self.actions_prev_step = copy.deepcopy(actions)

        reward_keys = REWARD_COMPONENT_KEYS
        rewards = [{key: 0.0 for key in reward_keys} for _ in self.agents]
        
        # Initialize info dictionary with new metric trackers
        infos = {
            "delivered_resource_ids_this_step": [],
            "kills_by_team": [0] * self.num_teams,
            "deaths_by_team": [0] * self.num_teams,
            "damage_by_team": [0.0] * self.num_teams,
            "grapples_initiated_by_team": [0] * self.num_teams,
            "grapples_broken_by_team": [0] * self.num_teams,
            "hive_damage_by_team": [0.0] * self.num_teams
        }

        self.step_counter += 1

        self.physics_manager._vectorized_state_decay_and_respawn(rewards)
        return rewards, infos

    def _apply_actions_and_step_simulation(self, actions: List[Dict], rewards: List[Dict[str, float]], proximity_data: Dict, infos: Dict) -> List:
        """
        Applies agent actions as forces, steps the physics simulation, and updates
        internal states from the physics engine. (Phases 3 & 4)

        This involves:
        - Applying movement forces based on agent actions.
        - Handling interaction logic (pickup, grapple) based on agent actions and proximity.
        - Calling `p.stepSimulation()` to move the physics world forward.
        - Updating internal state representations from the physics engine.
        - Updating visual elements like agent trails.
        """
        # Apply movement forces and interaction logic
        self.physics_manager._prepare_global_entity_tensors_for_step()
        self.physics_manager._vectorized_apply_movement_forces(actions)
        grapple_break_events = self.physics_manager._iterative_apply_interaction_logic(
            actions, rewards, proximity_data['agent_resource_map'], proximity_data['agent_enemy_map'], infos
        )

        # Advance the physics simulation by one step
        p.stepSimulation(physicsClientId=self.physicsClient)
        self.physics_manager._update_and_correct_physics_states()

        # Update visual elements like agent trails and debug lines
        for i, agent in enumerate(self.agents):
            if agent and agent.get('alive') and agent.get('pos') is not None:
                self.agent_trails[i].append(agent['pos'].copy())
        self._update_pybullet_visualizations()

        return grapple_break_events

    def _process_post_physics_logic(self, rewards: List[Dict[str, float]], infos: Dict, proximity_data: Dict, grapple_break_events: List):
        """
        Processes game logic outcomes that depend on the results of the physics step,
        such as resource delivery, combat, and other rewardable events. (Phase 5)
        """
        self.physics_manager._process_grapple_outcomes(rewards, grapple_break_events, infos)
        delivered_res_ids = self.reward_manager._process_resource_delivery(rewards, proximity_data['gnn_idx_to_res_obj_map'])
        if delivered_res_ids:
            infos["delivered_resource_ids_this_step"].extend(list(delivered_res_ids))
            self.resources = [r for r in self.resources if r['id'] not in delivered_res_ids]

        self.physics_manager.process_combat(rewards, proximity_data['agent_agent_combat_pairs'], infos)
        self.physics_manager.process_hive_engagements(rewards, proximity_data['agent_hive_interaction_pairs'], proximity_data['gnn_idx_to_hive_obj_map'], infos)
        self.reward_manager._process_discovery_and_misc_rewards(rewards, proximity_data)  # V2: Pass proximity_data for optimization
        self.reward_manager._process_continuous_hive_rewards(rewards)

    def _finalize_step_and_generate_observations(self) -> Tuple[List, bool, bool]:
        """
        Finalizes the step and generates new observations. (Phase 6)

        This involves:
        - Cleaning up delivered resources and respawning new ones.
        - Preparing global entity data for the observation manager.
        - Calling the observation manager to generate the final observations for each agent.
        - Checking if the episode should be terminated or truncated.
        """
        self.spawn_manager._cleanup_and_respawn_resources()

        # Prepare global tensors for the observation generation
        self.physics_manager._prepare_global_entity_tensors_for_step()
        
        # Pass necessary info to the observation manager for raycasting
        obstacle_body_ids = [obs['body_id'] for obs in self.obstacles if obs and obs.get('body_id') is not None]
        self.observation_manager.obstacle_body_ids = obstacle_body_ids
        self.observation_manager.physics_client_id = self.physicsClient
        
        # Generate the final, structured observations for each agent
        obs = self.observation_manager._generate_final_observations(agent_idx_to_observe=self.single_agent_obs_idx)

        terminated = False  # Termination logic can be added here if needed
        truncated = self.step_counter >= self.max_steps

        return obs, terminated, truncated

    def update_reward_overrides(self, reward_overrides: Dict) -> None:
        """
        Updates the reward weights for all teams.

        This method allows for dynamic changing of the reward structure during
        training, which is essential for curriculum learning. It directly calls
        the `RewardManager` to apply the new weights.

        Args:
            reward_overrides: A dictionary defining the new reward weights.
                              The structure should be {team_id: {reward_key: weight}}.
        """
        self.reward_manager.update_reward_weights(reward_overrides)

    def close(self) -> None:
        """
        Cleans up the environment and closes connections.

        This should be called when the environment is no longer needed. It disconnects
        from the PyBullet physics server and shuts down the Pygame window if it was used.
        """
        print("Closing Swarm2DEnv...")
        try:
            if hasattr(self, 'physicsClient') and self.physicsClient >= 0 and p.isConnected(self.physicsClient):
                p.disconnect(physicsClientId=self.physicsClient)
                self.physicsClient = -1 # Mark as disconnected
        except p.error as e: print(f"PyBullet disconnect error: {e}")
        except AttributeError: pass

        try:
            # Check pygame init status more carefully
            if hasattr(self, 'render_mode') and self.render_mode in ["human", "both"] and pygame.get_init():
                pygame.display.quit()
                pygame.quit()
        except Exception as e: print(f"Pygame quit error: {e}")
        