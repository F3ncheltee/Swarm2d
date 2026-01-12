#!/usr/bin/env python3

import numpy as np
import math
import random
import pybullet as p
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import copy
import time
import json
import logging

from constants import (
    AGENT_BASE_STRENGTH,
    AGENT_RADIUS,
    AGENT_SLOWED_FACTOR,
    COLLISION_GROUP_OBSTACLE,
    COLLISION_GROUP_RESOURCE,
    MEM_NODE_FEATURE_DIM,
    OCC_CH_COUNT,
    D_MAX
)
from env.helper import distance
from env.spawn import HiveSpawn, SpawnHelpers, ResourceSpawn, ObstacleSpawn, AgentSpawn, random_spawn_near_hive
from env.physics import (CombatManager, StatusManager, InteractionManager, 
                                 AttachmentManager, MovementManager, HiveCombatManager, DataManager)
from env.observations import (RawMapObservationManager, ActorMapState)
from env.rewards import RewardManager as RewardGenerator
from env.render import RenderManager as RenderGenerator
from torch_geometric.data import Data, Batch
from env.occlusion import check_los_batched_gpu_sampling, check_los_batched_pybullet, get_entities_in_radius_batched_pybullet
from env.observations import obs_profiler
import torch_geometric.nn as pyg_nn




def _tensor_to_str(tensor: torch.Tensor) -> str:
    """Helper to format a tensor for printing."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return np.array2string(tensor.detach().numpy(), precision=2, separator=', ', suppress_small=True)


class SpawnManager:
    """Manages all spawning functionality for the environment using delegation."""

    def __init__(self, env):
        self.env = env
        # Initialize the spawn classes as mixins
        self.hive_spawn = HiveSpawn()
        self.spawn_helpers = SpawnHelpers()
        self.resource_spawn = ResourceSpawn()
        self.obstacle_spawn = ObstacleSpawn()
        self.agent_spawn = AgentSpawn()

        # Copy environment attributes to spawn classes
        self._copy_env_attributes()

        # Initialize counters
        self.hive_id_counter = 0
        self.obstacle_id_counter = 0
        self.resource_id_counter = 0

        # Pass physics params from env to spawner
        self.agent_spawn.pb_agent_mass = self.env.pb_agent_mass
        self.agent_spawn.pb_agent_lateral_friction = self.env.pb_agent_lateral_friction
        self.agent_spawn.pb_agent_linear_damping = self.env.pb_agent_linear_damping
        self.agent_spawn.pb_agent_angular_damping = self.env.pb_agent_angular_damping
        self.agent_spawn.pb_agent_spinning_friction = self.env.pb_agent_spinning_friction
        self.agent_spawn.pb_agent_rolling_friction = self.env.pb_agent_rolling_friction
        self.agent_spawn.pb_agent_restitution = self.env.pb_agent_restitution

        # Pass resource physics params
        self.resource_spawn.resource_base_mass = self.env.resource_base_mass

    def _copy_env_attributes(self):
        """Copy necessary attributes from env to spawn classes."""
        attrs_to_copy = [
            'num_teams', 'num_agents_per_team', 'num_agents', 'num_resources_config',
            'num_obstacles_config', 'width', 'height', 'd_max', 'metadata', 'physicsClient',
            'render_mode', # <-- ADD THIS
            'debug',
            'agent_counts_per_team',
            # --- New Agent Configs ---
            'agent_radius_config', 'agent_base_strength_config', 'agent_max_energy_config',
            'agent_max_health_config', 'sensing_range_fraction_config',
            # --- End New ---
            'min_resource_radius_pb', 'max_resource_radius_pb', 'coop_min_resource_radius_pb',
            'coop_max_resource_radius_pb', 'resource_base_mass', 'resource_mass_scale_factor',
            'pb_agent_mass', 'pb_agent_linear_damping', 'pb_agent_angular_damping',
            'pb_agent_lateral_friction', 'pb_agent_restitution', 'pb_res_damping_static',
            'pb_res_friction_static', 'pb_res_restitution', 'bee_speed_config', 'obs_radius',
            'hives', 'hive_body_ids', 'obstacles', 'resources', 'agents', 'agent_body_ids',
            'current_step_all_pos_t', 'current_step_all_radii_t', 'device', 'agent_randomization_factors',
            'team_parameter_overrides',
            # --- New Spawning Detail Parameters ---
            'hive_min_distance', 'hive_spawn_jitter', 'hive_spawn_radius_factor',
            'resource_hive_buffer', 'resource_obstacle_buffer', 'coop_resource_probability',
            'obstacle_hive_buffer', 'agent_spawn_radius', 'agent_mass_strength_influence',
            'agent_mass_min_factor', 'agent_mass_max_factor',
            'team_configs' # Add this to ensure it's copied to spawn modules
        ]

        for attr in attrs_to_copy:
            if hasattr(self.env, attr):
                for spawn_class in [self.hive_spawn, self.spawn_helpers, self.resource_spawn,
                                  self.obstacle_spawn, self.agent_spawn]:
                    setattr(spawn_class, attr, getattr(self.env, attr))

    def spawn_hives(self):
        """Delegates to HiveSpawn class."""
        self._copy_env_attributes()
        self.hive_spawn.hive_id_counter = self.hive_id_counter
        self.hive_spawn.hives = self.env.hives
        self.hive_spawn.hive_body_ids = self.env.hive_body_ids

        # Delegate to the specialized class
        self.hive_spawn.spawn_hives()

        self.hive_id_counter = self.hive_spawn.hive_id_counter
        self.env.hives = self.hive_spawn.hives
        self.env.hive_body_ids = self.hive_spawn.hive_body_ids

    def init_obstacles_pybullet(self):
        """Delegates to ObstacleSpawn class."""
        self._copy_env_attributes()
        self.obstacle_spawn.obstacle_id_counter = self.obstacle_id_counter
        self.obstacle_spawn.obstacles = self.env.obstacles

        # Delegate to the specialized class
        self.obstacle_spawn.init_obstacles_pybullet()

        self.obstacle_id_counter = self.obstacle_spawn.obstacle_id_counter
        self.env.obstacles = self.obstacle_spawn.obstacles

    def init_boundaries(self):
        """Delegates to ObstacleSpawn class."""
        self._copy_env_attributes()
        self.obstacle_spawn.init_boundaries()

    def spawn_resource(self):
        """Delegates to ResourceSpawn class."""
        self._copy_env_attributes()
        self.resource_spawn.resource_id_counter = self.resource_id_counter
        self.resource_spawn.resources = self.env.resources

        # Delegate to the specialized class
        result = self.resource_spawn.spawn_resource()

        self.resource_id_counter = self.resource_spawn.resource_id_counter
        self.env.resources = self.resource_spawn.resources

        return result

    def _cleanup_and_respawn_resources(self):
        """Delegates to ResourceSpawn class."""
        self._copy_env_attributes()
        self.resource_spawn.resources = self.env.resources
        self.resource_spawn._cleanup_and_respawn_resources()
        self.env.resources = self.resource_spawn.resources

    def init_agents_pybullet(self):
        """Delegates to AgentSpawn class."""
        self._copy_env_attributes()
        self.agent_spawn.agents = self.env.agents
        self.agent_spawn.agent_body_ids = self.env.agent_body_ids

        # Delegate to the specialized class
        self.agent_spawn.init_agents_pybullet()

        self.env.agents = self.agent_spawn.agents
        self.env.agent_body_ids = self.agent_spawn.agent_body_ids

    def respawn_agent(self, agent):
        """Delegates to AgentSpawn class."""
        self._copy_env_attributes()
        self.agent_spawn.respawn_agent(agent)


class PhysicsManager(CombatManager, StatusManager, InteractionManager,
                     AttachmentManager, MovementManager, HiveCombatManager, DataManager):
    """
    Manages all physics-related functionality for the environment.
    Inherits from specialized physics classes in physics.py.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """
        Dynamically delegate attribute access to the env object.
        This allows methods inherited from physics.py classes to access
        environment attributes (e.g., self.agents) as if they were part
        of the Swarm2DEnv class, preserving the original code structure.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'PhysicsManager' object and its 'env' have no attribute '{name}'")


class ObservationManager(RawMapObservationManager):
    """
    The primary observation manager that orchestrates different types of observations.
    It inherits from specialized managers to compose the final observation.
    """

    def __init__(self, env_ref):
        super(ObservationManager, self).__init__()
        self.env = env_ref
        self.device = self.env.device
        self.debug_mode = self.env.debug_mode
        # Use env.step_counter instead of maintaining our own (it's updated in env.step)

        # --- Correctly store the graph connection factor ---
        self.graph_connection_radius_factor = self.env.graph_connection_radius_factor

        # Constants and definitions used throughout the manager
        self.node_feature_map_const = self.env.node_feature_map_const
        self.node_type_def_const = self.env.node_type_def_const

        # Enable PyTorch optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.min_cluster_size = self.env.min_cluster_size
        self.max_cluster_size = self.env.max_cluster_size
        self.graph_connection_radius_factor = self.env.graph_connection_radius_factor
        self.cluster_aggressiveness = self.env.cluster_aggressiveness
        self.cluster_exclusion_radius_factor = self.env.cluster_exclusion_radius_factor
        self.detailed_clustering_radius_factor = self.env.detailed_clustering_radius_factor # New
        self.mem_skeleton_connection_factor = self.env.mem_skeleton_connection_factor
        self.max_steps = self.env.max_steps
        self.debug = self.env.debug
        self.raw_map_grid_size = env_ref.raw_map_grid_size
        self.memory_map_grid_size = env_ref.memory_map_grid_size


    def _generate_all_observations_batched_internal(self, agent_idx_to_observe: Optional[int] = None) -> Dict[int, Dict[str, Union[torch.Tensor, Data]]]:
        """
        Generates observations for all living agents.
        Correctly instantiates a PyG Batch object to resolve the .to_data_list() AttributeError.
        """
        obs_profiler.start_step()
        obs_output_all_agents: Dict[int, Dict[str, Union[torch.Tensor, Data]]] = {}

        # --- 1. Gather Data for ALIVE Agents ---
        start_time = time.time()

        if agent_idx_to_observe is not None:
            alive_agent_env_indices: List[int] = [agent_idx_to_observe] if self.agents[agent_idx_to_observe] and self.agents[agent_idx_to_observe].get('alive') else []
        else:
            alive_agent_env_indices: List[int] = [idx for idx, agent_data in enumerate(self.agents) if agent_data and agent_data.get('alive')]

        if not alive_agent_env_indices:
            return obs_output_all_agents

        num_alive_agents = len(alive_agent_env_indices)
        agent_ids_alive = [self.agents[i]['id'] for i in alive_agent_env_indices]
        try:
            agent_node_indices_global_batch_t = torch.tensor([self.current_step_agent_id_to_node_idx_map[aid] for aid in agent_ids_alive], device=self.device, dtype=torch.long)
        except KeyError as e:
            # This can happen in the very brief window where an agent dies after proximity search but before obs gen
            print(f"Warning (_generate_all_obs): KeyError looking up agent ID {e}. Agent likely just died. Skipping obs gen for this batch.")
            return obs_output_all_agents

        # --- 2. Extract Batched Data from Global Tensors ---
        agent_pos_batch_t = self.current_step_all_pos_t[agent_node_indices_global_batch_t]
        agent_feat_batch_t = self.current_step_all_feat_t[agent_node_indices_global_batch_t]
        agent_team_ids_batch_t = self.current_step_all_teams_t[agent_node_indices_global_batch_t]
        agent_vel_batch_t = agent_feat_batch_t[:, [self.node_feature_map_const['vel_x_norm'], self.node_feature_map_const['vel_y_norm']]]
        agent_obs_radii_batch_t = agent_feat_batch_t[:, self.node_feature_map_const['obs_radius_norm']] * self.metadata['max_obs_radius_possible']
        obs_profiler.record("data_gathering", time.time() - start_time)

        # --- DEBUG START: Detailed prints for Agent 0 ---
        if self.debug_mode and self.env.step_counter < 5 and 0 in alive_agent_env_indices:
            agent_0_batch_idx = alive_agent_env_indices.index(0)
            print(f"\n[DEBUG @ Step {self.env.step_counter}, Agent 0, ObsManager]")
            agent_0_pos = agent_pos_batch_t[agent_0_batch_idx]
            agent_0_radius = agent_obs_radii_batch_t[agent_0_batch_idx]
            print(f"  - Agent 0 Position: [{agent_0_pos[0]:.1f}, {agent_0_pos[1]:.1f}], Obs Radius: {agent_0_radius:.1f}")
        # --- DEBUG END ---

        # --- 3. SELF-OBSERVATION GENERATION (Fully Vectorized) ---
        start_time = time.time()
        width_norm = float(self.metadata.get('width', self.width))
        height_norm = float(self.metadata.get('height', self.height))
        max_speed_norm = float(self.metadata.get('max_agent_speed_observed', self.bee_speed_config * 1.2))
        max_strength_norm_selfobs = float(self.metadata.get('max_agent_strength_observed', AGENT_BASE_STRENGTH * 1.2))
        max_agent_radius_norm_selfobs = float(self.metadata.get('max_agent_radius_observed', AGENT_RADIUS * 1.2))
        max_obs_radius_norm_selfobs = float(self.metadata.get('max_obs_radius_possible', self.obs_radius * 1.2))
        d_max_norm = np.sqrt(width_norm**2 + height_norm**2)

        self_obs_pos_x_norm = agent_feat_batch_t[:, self.node_feature_map_const['pos_x_norm']]
        self_obs_pos_y_norm = agent_feat_batch_t[:, self.node_feature_map_const['pos_y_norm']]
        self_obs_vel_x_norm = agent_vel_batch_t[:, 0]
        self_obs_vel_y_norm = agent_vel_batch_t[:, 1]
        self_obs_energy_norm = agent_feat_batch_t[:, self.node_feature_map_const['energy_norm']]
        self_obs_health_norm = agent_feat_batch_t[:, self.node_feature_map_const['health_norm']]
        self_obs_carrying_flag = agent_feat_batch_t[:, self.node_feature_map_const['is_carrying']]
        self_obs_agent_id = agent_feat_batch_t[:, self.node_feature_map_const['agent_id']]

        raw_agent_radii_batch = agent_feat_batch_t[:, self.node_feature_map_const['size_norm']] * self.metadata['max_size_norm_divisor']
        # Normalization uses theoretical maximum, so values should never exceed 1.0
        self_obs_agent_radius_norm = raw_agent_radii_batch / max_agent_radius_norm_selfobs
        self_obs_obs_radius_norm = agent_obs_radii_batch_t / max_obs_radius_norm_selfobs

        energy_factor = torch.clamp(self_obs_energy_norm, 0.05, 1.0)
        carrying_multiplier = torch.where(self_obs_carrying_flag > 0.5, 1.0, 1.0)
        slowed_multiplier = torch.where(agent_feat_batch_t[:, self.node_feature_map_const['is_slowed']] > 0.5, AGENT_SLOWED_FACTOR, 1.0)
        # base_speed_norm is already normalized [0, 1], multiply by max to get actual speed
        base_speed = agent_feat_batch_t[:, self.node_feature_map_const['base_speed_norm']] * max_speed_norm
        effective_speed = base_speed * energy_factor * carrying_multiplier * slowed_multiplier
        # All multipliers are <= 1.0, so effective_speed <= base_speed <= max_speed_norm
        self_obs_norm_effective_speed = effective_speed / max_speed_norm

        base_strength = agent_feat_batch_t[:, self.node_feature_map_const['strength_norm']] * max_strength_norm_selfobs
        effective_strength = base_strength * energy_factor
        # energy_factor <= 1.0, so effective_strength <= base_strength <= max_strength_norm_selfobs
        self_obs_norm_effective_strength = effective_strength / max_strength_norm_selfobs

        hive_mask = (self.current_step_all_types_t == self.node_type_def_const['hive'])
        all_hive_pos_t = self.current_step_all_pos_t[hive_mask]
        all_hive_teams_t = self.current_step_all_teams_t[hive_mask]
        all_hive_health_norm_t = self.current_step_all_feat_t[hive_mask, self.node_feature_map_const['health_norm']]

        own_hive_indices = (all_hive_teams_t.unsqueeze(0) == agent_team_ids_batch_t.unsqueeze(1)).nonzero(as_tuple=True)

        rel_hive_x_norm = torch.zeros(num_alive_agents, device=self.device)
        rel_hive_y_norm = torch.zeros(num_alive_agents, device=self.device)
        current_hive_dist_norm = torch.ones(num_alive_agents, device=self.device)
        own_hive_health_norm = torch.zeros(num_alive_agents, device=self.device)

        if own_hive_indices[0].numel() > 0:
            agent_indices, hive_indices = own_hive_indices
            own_hive_pos_batch = all_hive_pos_t[hive_indices]
            rel_hive_vec = own_hive_pos_batch - agent_pos_batch_t[agent_indices]
            
            # --- FIX: USE UNIT VECTORS FOR NAVIGATION ---
            # Previously: rel_hive_vec / d_max_norm (Distance dependent, gets weak when close)
            # Now: Normalized Direction (Unit Vector) + Separate Distance Channel
            hive_dist = torch.linalg.norm(rel_hive_vec, dim=1, keepdim=True).clamp(min=1e-6)
            hive_dir = rel_hive_vec / hive_dist
            
            # Use Direction for indices 6, 7
            rel_hive_x_norm.scatter_(0, agent_indices, hive_dir[:, 0])
            rel_hive_y_norm.scatter_(0, agent_indices, hive_dir[:, 1])
            
            # --- DEBUG HIVE OBSERVATION ---
            if self.debug_mode and 0 in agent_indices and self.env.step_counter % 50 == 0:
                idx_in_batch = (agent_indices == 0).nonzero(as_tuple=True)[0].item()
                h_pos = own_hive_pos_batch[idx_in_batch]
                a_pos = agent_pos_batch_t[agent_indices[idx_in_batch]]
                vec = rel_hive_vec[idx_in_batch]
                nx = rel_hive_x_norm[0].item()
                ny = rel_hive_y_norm[0].item()
                print(f"[OBS DEBUG] Agent 0 -> Hive: AgentPos={a_pos.cpu().numpy()}, HivePos={h_pos.cpu().numpy()}, Vec={vec.cpu().numpy()}, UnitDir=({nx:.2f}, {ny:.2f})")
            # -----------------------------

            # Distance is always <= d_max_norm, so normalization stays in [0, 1]
            current_hive_dist_norm.scatter_(0, agent_indices, (hive_dist.flatten() / d_max_norm))
            # Health is already normalized from graph features
            own_hive_health_norm.scatter_(0, agent_indices, all_hive_health_norm_t[hive_indices])

        # --- NEW: ALIGNMENT SCORE (Dot Product of Velocity and Hive Vector) ---
        # Normalize velocity vector for pure direction comparison if moving
        agent_speed_batch = torch.linalg.norm(agent_vel_batch_t, dim=1).clamp(min=1e-6)
        agent_dir_x = agent_vel_batch_t[:, 0] / agent_speed_batch
        agent_dir_y = agent_vel_batch_t[:, 1] / agent_speed_batch
        
        # Alignment = dir_x * hive_x + dir_y * hive_y
        # Inputs are already normalized relative to map, but for dot product we want direction vectors.
        # rel_hive_x_norm is (dx/D_MAX). We want normalized direction.
        hive_dist_safe = current_hive_dist_norm.clamp(min=1e-6)
        hive_dir_x = rel_hive_x_norm / hive_dist_safe
        hive_dir_y = rel_hive_y_norm / hive_dist_safe
        
        alignment = (agent_dir_x * hive_dir_x + agent_dir_y * hive_dir_y)
        # Mask out agents that aren't moving significantly (avoid noise)
        alignment = torch.where(agent_speed_batch > 0.05, alignment, torch.zeros_like(alignment))

        res_mask = (self.current_step_all_types_t == self.node_type_def_const['resource']) & \
                   (self.current_step_all_feat_t[:, self.node_feature_map_const['is_delivered']] < 0.5)

        rel_res_x_norm = torch.zeros_like(self_obs_pos_x_norm)
        rel_res_y_norm = torch.zeros_like(self_obs_pos_x_norm)
        # Initialize Distance Channel (Default 1.0 = Far/Unseen)
        res_dist_norm_ch = torch.ones_like(self_obs_pos_x_norm)

        if res_mask.any():
            all_active_res_pos = self.current_step_all_pos_t[res_mask]
            if all_active_res_pos.numel() > 0:
                # Calculate distances more efficiently
                agent_pos_expanded = agent_pos_batch_t.unsqueeze(1)  # [N_agents, 1, 2]
                res_pos_expanded = all_active_res_pos.unsqueeze(0)   # [1, N_resources, 2]
                dists_to_res = torch.norm(agent_pos_expanded - res_pos_expanded, dim=2)  # [N_agents, N_resources]
                min_dists, min_indices = torch.min(dists_to_res, dim=1)
                closest_res_pos_batch = all_active_res_pos[min_indices]
                rel_res_vec = closest_res_pos_batch - agent_pos_batch_t
                within_obs_mask = min_dists < agent_obs_radii_batch_t
                rel_res_vec_filtered = torch.where(within_obs_mask.unsqueeze(1), rel_res_vec, torch.zeros_like(rel_res_vec))
                
                # --- FIX: USE UNIT VECTORS FOR RESOURCE NAVIGATION ---
                # Calculate raw distance for normalization
                # We use min_dists which is the raw distance to closest resource
                
                # 1. Update Direction Channels (Unit Vectors)
                res_dist_safe = torch.norm(rel_res_vec_filtered, dim=1, keepdim=True).clamp(min=1e-6)
                res_dir = rel_res_vec_filtered / res_dist_safe
                # Zero out unseen
                res_dir = torch.where(within_obs_mask.unsqueeze(1), res_dir, torch.zeros_like(res_dir))
                
                rel_res_x_norm = res_dir[:, 0]
                rel_res_y_norm = res_dir[:, 1]
                
                # 2. Update Distance Channel (Normalized)
                # If visible: dist/D_MAX. If not: 1.0.
                dist_norm = min_dists / D_MAX
                res_dist_norm_ch = torch.where(within_obs_mask, dist_norm, torch.ones_like(dist_norm))

        team_avg_energy_norm = torch.zeros_like(self_obs_pos_x_norm)
        unique_teams_in_batch = torch.unique(agent_team_ids_batch_t)
        for team_id_calc in unique_teams_in_batch:
            agents_in_team_mask_global = (self.current_step_all_types_t == self.node_type_def_const['agent']) & \
                                         (self.current_step_all_teams_t == team_id_calc.item())
            if agents_in_team_mask_global.any():
                energies_this_team = self.current_step_all_feat_t[agents_in_team_mask_global, self.node_feature_map_const['energy_norm']]
                avg_energy = energies_this_team.mean()
                batch_agents_of_this_team_mask = (agent_team_ids_batch_t == team_id_calc)
                team_avg_energy_norm[batch_agents_of_this_team_mask] = avg_energy

        boundary_feat = torch.stack([self_obs_pos_x_norm, 1.0 - self_obs_pos_x_norm, self_obs_pos_y_norm, 1.0 - self_obs_pos_y_norm], dim=1).clamp(0.0, 1.0)

        # --- Grip Strength Information (Single Feature) ---
        alive_agents_data = [self.agents[idx] for idx in alive_agent_env_indices]
        if alive_agents_data:
            grip_strength_norm_tensor = self.physics_manager._get_grip_strength_info_batched(alive_agents_data, self.device)
            grip_strength_norm = grip_strength_norm_tensor.squeeze(-1)
        else:
            grip_strength_norm = torch.zeros(num_alive_agents, device=self.device, dtype=torch.float32)

        # --- Agent ID Feature ---
        agent_ids_norm = torch.tensor(agent_ids_alive, dtype=torch.float32, device=self.device) / self.num_agents

        # --- V4: GRAPPLE AWARENESS FEATURES ---
        # Extract features for all ALIVE agents in this batch
        is_grappling_list = []
        is_grappled_list = []
        applied_torque_list = []
        grapple_tension_list = []
        
        MAX_TORQUE_NORM = 2.0 # Estimate max useful torque for normalization
        MAX_TENSION_NORM = 2000.0 # Estimate based on max_force settings

        for agent in alive_agents_data:
            # 1. Is Grappling
            is_grappling_list.append(1.0 if agent.get('is_grappling') else 0.0)
            
            # 2. Is Grappled
            is_grappled_list.append(1.0 if agent.get('is_grappled') else 0.0)
            
            # 3. Applied Torque (Normalized)
            torque = agent.get('applied_torque', 0.0)
            applied_torque_list.append(np.clip(torque / MAX_TORQUE_NORM, -1.0, 1.0))
            
            # 4. Grapple Tension (Approximated by last set force)
            tension = agent.get('grapple_last_set_force', 0.0) if agent.get('is_grappling') else 0.0
            grapple_tension_list.append(np.clip(tension / MAX_TENSION_NORM, 0.0, 1.0))

        is_grappling_t = torch.tensor(is_grappling_list, device=self.device, dtype=torch.float32).unsqueeze(1)
        is_grappled_t = torch.tensor(is_grappled_list, device=self.device, dtype=torch.float32).unsqueeze(1)
        applied_torque_t = torch.tensor(applied_torque_list, device=self.device, dtype=torch.float32).unsqueeze(1)
        grapple_tension_t = torch.tensor(grapple_tension_list, device=self.device, dtype=torch.float32).unsqueeze(1)

        self_obs_components = [
            self_obs_pos_x_norm.unsqueeze(1), self_obs_pos_y_norm.unsqueeze(1),
            self_obs_vel_x_norm.unsqueeze(1), self_obs_vel_y_norm.unsqueeze(1),
            rel_res_x_norm.unsqueeze(1), rel_res_y_norm.unsqueeze(1),
            rel_hive_x_norm.unsqueeze(1), rel_hive_y_norm.unsqueeze(1),
            self_obs_carrying_flag.unsqueeze(1),
            current_hive_dist_norm.unsqueeze(1),
            res_dist_norm_ch.unsqueeze(1), # Index 10
            self_obs_norm_effective_speed.unsqueeze(1),
            self_obs_agent_radius_norm.unsqueeze(1),
            self_obs_obs_radius_norm.unsqueeze(1),
            self_obs_norm_effective_strength.unsqueeze(1),
            own_hive_health_norm.unsqueeze(1),
            self_obs_energy_norm.unsqueeze(1),
            self_obs_health_norm.unsqueeze(1),
            team_avg_energy_norm.unsqueeze(1),
            boundary_feat,
            grip_strength_norm.unsqueeze(1),
            agent_ids_norm.unsqueeze(1),
            agent_team_ids_batch_t.clone().detach().to(device=self.device, dtype=torch.float32).unsqueeze(1), # team_id_val
            alignment.unsqueeze(1), # alignment_score
            # --- V4 Added Features ---
            is_grappling_t,
            is_grappled_t,
            applied_torque_t,
            grapple_tension_t
        ]
        self_obs_tensor_batch_unpadded = torch.cat(self_obs_components, dim=1)

        current_dim = self_obs_tensor_batch_unpadded.shape[1]
        if current_dim != self.env_self_obs_dim:
            if current_dim < self.env_self_obs_dim:
                padding = torch.zeros((num_alive_agents, self.env_self_obs_dim - current_dim), device=self.device, dtype=torch.float32)
                self_obs_tensor_batch = torch.cat([self_obs_tensor_batch_unpadded, padding], dim=1)
            else:
                self_obs_tensor_batch = self_obs_tensor_batch_unpadded[:, :self.env_self_obs_dim]
        else:
            self_obs_tensor_batch = self_obs_tensor_batch_unpadded

        obs_profiler.record("self_observation", time.time() - start_time)

        # --- 4. OPTIMIZED VISIBILITY & LOS (Using PyBullet Spatial Queries) ---
        start_time = time.time()
        final_visibility_mask = torch.zeros((num_alive_agents, self.current_step_all_pos_t.shape[0]), dtype=torch.bool, device=self.device)

        if self.current_step_all_pos_t.numel() > 0:
            # OPTIMIZATION: Use PyBullet spatial queries instead of expensive torch.cdist
            # This replaces O(N×M) distance calculations with much faster spatial queries
            entity_indices_per_agent = get_entities_in_radius_batched_pybullet(
                observer_positions=agent_pos_batch_t,
                observer_radii=agent_obs_radii_batch_t,
                physics_client_id=self.physics_client_id,
                all_entity_pos=self.current_step_all_pos_t,
                all_entity_radii=self.current_step_all_radii_t,
                all_entity_ids=torch.arange(self.current_step_all_pos_t.shape[0], device=self.device)
            )

            # --- DEBUG START: Print for Agent 0 ---
            if self.debug_mode and self.env.step_counter < 5 and 0 in alive_agent_env_indices:
                agent_0_batch_idx = alive_agent_env_indices.index(0)
                num_in_radius = len(entity_indices_per_agent[agent_0_batch_idx])
                print(f"  1a. Radius Check: Found {num_in_radius} entities within radius.")
            # --- DEBUG END ---

            # Build the visibility mask from the spatial query results
            for agent_idx, entity_indices in enumerate(entity_indices_per_agent):
                if entity_indices.numel() > 0:
                    final_visibility_mask[agent_idx, entity_indices] = True

            # Apply LOS checking only to the entities found by spatial queries
            obs_indices_b_rad, neigh_indices_all_rad = torch.where(final_visibility_mask)

            if obs_indices_b_rad.numel() > 0:
                obs_orig_node_idx_los = agent_node_indices_global_batch_t[obs_indices_b_rad]
                # We only need to check LOS for entities that are not the agent itself
                needs_los_mask = (obs_orig_node_idx_los != neigh_indices_all_rad)

                if needs_los_mask.any():
                    # Select only the pairs that need an LOS check
                    valid_los_observers = agent_pos_batch_t[obs_indices_b_rad[needs_los_mask]]
                    valid_los_targets = self.current_step_all_pos_t[neigh_indices_all_rad[needs_los_mask]]
                    target_body_ids = self.current_step_all_body_ids_t[neigh_indices_all_rad[needs_los_mask]]

                    # Choose LOS method based on configuration
                    if self.use_pybullet_raycasting:
                        # Use collision filtering to ignore agents/hives for LOS ---
                        los_collision_mask = COLLISION_GROUP_OBSTACLE | COLLISION_GROUP_RESOURCE
                        # Use PyBullet raycasting for LOS checking
                        los_results_subset = check_los_batched_pybullet(
                            valid_los_observers,
                            valid_los_targets,
                            target_body_ids,
                            self.physics_client_id,
                            collision_filter_mask=los_collision_mask
                        )
                    elif self.current_step_gpu_occlusion_field is not None:
                        # Use GPU sampling for LOS checking (fallback)
                        los_results_subset = check_los_batched_gpu_sampling(
                            valid_los_observers, valid_los_targets,
                            self.current_step_gpu_occlusion_field,
                            float(self.width), float(self.height),
                            self.occlusion_field_res_env, self.num_los_samples_env, self.los_occlusion_thresh_env
                        )
                    else:
                        # No occlusion field available, assume all LOS are clear
                        los_results_subset = torch.ones(valid_los_observers.shape[0], dtype=torch.bool, device=self.device)

                    # Create a full tensor of LOS results for all pairs within radius
                    los_clear_rad_pairs = torch.ones_like(obs_indices_b_rad, dtype=torch.bool, device=self.device)
                    # Update the results only for the pairs we checked
                    los_clear_rad_pairs[needs_los_mask] = los_results_subset

                    # Update the final visibility mask: an entity is visible if it's in radius AND has LOS
                    final_visibility_mask[obs_indices_b_rad, neigh_indices_all_rad] = los_clear_rad_pairs

        obs_profiler.record("visibility_los", time.time() - start_time)

        # --- DEBUG START: Print for Agent 0 ---
        if self.debug_mode and self.env.step_counter < 5 and 0 in alive_agent_env_indices:
            agent_0_batch_idx = alive_agent_env_indices.index(0)
            num_visible = final_visibility_mask[agent_0_batch_idx].sum().item()
            print(f"  1b. Visibility/LOS Check: Agent 0 sees {num_visible} total entities after LOS.")
        # --- DEBUG END ---

        # --- 5. BATCHED `raw_map` GENERATION ---
        start_time = time.time()
        agent_indices_flat_map, entity_indices_flat_map = torch.where(final_visibility_mask)
        if agent_indices_flat_map.numel() > 0:
            batched_raw_maps = self.generate_maps_wrapper(
                agent_indices_flat=agent_indices_flat_map,
                visible_entity_pos=self.current_step_all_pos_t[entity_indices_flat_map],
                visible_entity_feat=self.current_step_all_feat_t[entity_indices_flat_map],
                visible_entity_types=self.current_step_all_types_t[entity_indices_flat_map],
                visible_entity_teams=self.current_step_all_teams_t[entity_indices_flat_map],
                visible_entity_coop=self.current_step_all_coop_t[entity_indices_flat_map],
                visible_entity_radii=self.current_step_all_radii_t[entity_indices_flat_map], # ADDED
                observer_pos_batch=agent_pos_batch_t, observer_radii_batch=agent_obs_radii_batch_t,
                observer_teams_batch=agent_team_ids_batch_t, observer_feat_batch=self_obs_tensor_batch, # ADDED: Pass observer features
                batch_size=num_alive_agents,
                grid_size=self.raw_map_grid_size,
                world_to_map_scale=(2 * agent_obs_radii_batch_t) / self.raw_map_grid_size # Per-agent scale
            )
        else:
            batched_raw_maps = torch.zeros((num_alive_agents, self.raw_ch_count, self.raw_map_grid_size, self.raw_map_grid_size), device=self.device)
        
        obs_profiler.record("map_generation", time.time() - start_time)

        # --- 6. OPTIMIZED BATCHED GRAPH GENERATION (PyBullet Spatial Queries Only) ---
        start_time = time.time()
        is_ego_feature_idx = self.node_feature_map_const.get('is_ego', -1)
        if is_ego_feature_idx == -1: raise ValueError("'is_ego' key not found in self.node_feature_map_const.")

        # Ensure every agent gets at least an ego-node graph ---
        # Create a base list of graphs, each containing only the agent's self-representation (ego node).
        list_of_graph_data_objects = []
        for i in range(num_alive_agents):
            ego_features = agent_feat_batch_t[i:i+1].clone()
            ego_features[0, is_ego_feature_idx] = 1.0 # Mark this as the ego node

            ego_graph = Data(
                x=ego_features,
                pos=agent_pos_batch_t[i:i+1],
                radii=self.current_step_all_radii_t[agent_node_indices_global_batch_t[i:i+1]],
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                is_ego=torch.tensor([True], dtype=torch.bool, device=self.device),
                env_id=ego_features[:, self.node_feature_map_const['agent_id']]
            )
            list_of_graph_data_objects.append(ego_graph)

        # OPTIMIZATION: Only perform expensive graph edge generation if graphs are actually enabled
        if self.generate_memory_graph:
            agent_indices_flat_graph, entity_indices_flat_graph = torch.where(final_visibility_mask)

            if agent_indices_flat_graph.numel() > 0:
                device = agent_indices_flat_graph.device

                # 1. Prepare all visible node data in large tensors (same as before)
                visible_node_features = self.current_step_all_feat_t[entity_indices_flat_graph].clone()
                visible_node_pos = self.current_step_all_pos_t[entity_indices_flat_graph]
                visible_node_radii = self.current_step_all_radii_t[entity_indices_flat_graph]

                agent_global_indices_expanded = agent_node_indices_global_batch_t[agent_indices_flat_graph]
                is_ego_mask_flat = (agent_global_indices_expanded == entity_indices_flat_graph)
                visible_node_features[is_ego_mask_flat, is_ego_feature_idx] = 1.0

                # 2. Perform radius graph generation.
                # We first create a "superset" of edges with a large radius to connect all
                # nodes within each agent's observation batch.
                superset_pos = visible_node_pos
                superset_batch = agent_indices_flat_graph
                # MODIFICATION: Replaced knn_graph with radius_graph for more intuitive spatial connections.
                # This connects all nodes within a scaled observation radius.
                final_edge_index = pyg_nn.radius_graph(
                    superset_pos,
                    # The radius for connection is the agent's obs_radius scaled by a factor.
                    # We need to determine the radius for each node based on its batch index.
                    # A single radius for the whole batch is simpler if radii are similar. Let's use the max.
                    r=agent_obs_radii_batch_t.max() * self.graph_connection_radius_factor,
                    batch=superset_batch,
                    max_num_neighbors=32 # Cap neighbors to prevent quadratic complexity in dense scenarios.
                )

                # 4. Manually and efficiently split the batched tensors into a list of Data objects.
                # This avoids the incompatible .to_data_list() call while keeping the performance gain.
                unique_agent_indices_with_nodes = torch.unique(agent_indices_flat_graph)

                for agent_batch_idx in unique_agent_indices_with_nodes:
                    # Create a mask to select nodes belonging to the current agent
                    node_mask = (agent_indices_flat_graph == agent_batch_idx)

                    # Ensure ego node is not duplicated ---
                    # Exclude the ego-node from the visible entities if it's already there
                    agent_global_node_idx = agent_node_indices_global_batch_t[agent_batch_idx]
                    visible_entity_indices_for_agent = entity_indices_flat_graph[node_mask]

                    # Filter out the agent's own global index from its visible entities
                    non_ego_entity_indices = visible_entity_indices_for_agent[visible_entity_indices_for_agent != agent_global_node_idx]

                    # If there are no other entities, we've already created the ego-graph, so we can skip
                    if non_ego_entity_indices.numel() == 0:
                        continue

                    # Re-create the node_mask for slicing based on non-ego entities
                    # This requires finding the original flat indices for the non-ego entities
                    original_indices_mask = (agent_indices_flat_graph == agent_batch_idx)
                    original_indices = torch.where(original_indices_mask)[0]

                    final_node_mask = torch.zeros_like(agent_indices_flat_graph, dtype=torch.bool)
                    for idx in original_indices:
                        if entity_indices_flat_graph[idx] != agent_global_node_idx:
                            final_node_mask[idx] = True

                    # These are the global indices (0 to N-1) of nodes for this agent
                    global_node_indices_for_agent = torch.where(final_node_mask)[0]

                    # Slice the feature and position tensors
                    agent_x = visible_node_features[final_node_mask]
                    agent_pos = visible_node_pos[final_node_mask]
                    agent_radii = visible_node_radii[final_node_mask]
                    agent_is_ego = is_ego_mask_flat[final_node_mask]

                    # --- OPTIMIZED: Avoid per-agent lookup tables ---
                    # Find edges where the source node belongs to this agent
                    row, col = final_edge_index
                    edge_mask = (agent_indices_flat_graph[row] == agent_batch_idx)
                    agent_edges_global = final_edge_index[:, edge_mask]

                    # Use searchsorted instead of full lookup table
                    # global_node_indices_for_agent is already sorted by construction
                    # Compute ranks within local_idx using searchsorted
                    pos = torch.searchsorted(global_node_indices_for_agent, agent_edges_global.flatten())
                    # Check if all edges are valid (within bounds)
                    # This must be a two-step process to avoid out-of-bounds access on CUDA
                    clamped_pos = torch.clamp(pos, max=global_node_indices_for_agent.shape[0] - 1)

                    valid_mask = (pos < global_node_indices_for_agent.shape[0]) & \
                                 (global_node_indices_for_agent[clamped_pos] == agent_edges_global.flatten())

                    if valid_mask.all():
                        agent_edge_index_local = pos.reshape(2, -1)
                    else:
                        # Filter out invalid edges by reshaping the mask to match the edge pairs
                        valid_edge_mask = valid_mask.reshape(2, -1).all(dim=0)
                        valid_edges = agent_edges_global[:, valid_edge_mask]

                        # Re-calculate pos for the valid edges only
                        pos_valid = torch.searchsorted(global_node_indices_for_agent, valid_edges.flatten())
                        agent_edge_index_local = pos_valid.reshape(2, -1)

                    # Combine ego graph with neighbor graph ---
                    ego_graph = list_of_graph_data_objects[agent_batch_idx]

                    # Combine features, pos, radii
                    combined_x = torch.cat([ego_graph.x, agent_x], dim=0)

                    # --- NEW EGO DEBUG ---
                    if self.debug_mode and agent_batch_idx == 0 and self.env.step_counter < 5:
                        num_ego = torch.sum(combined_x[:, is_ego_feature_idx]).item()
                        print(f"[TRACEPOINT 1 @ Step {self.env.step_counter}, ObsManager] Live graph created. Ego nodes in 'x': {num_ego}.")
                    # --- END NEW EGO DEBUG ---

                    combined_pos = torch.cat([ego_graph.pos, agent_pos], dim=0)
                    combined_radii = torch.cat([ego_graph.radii, agent_radii], dim=0)

                    # Adjust edge indices: they are local to the 'agent_x' tensor, so we need to offset them
                    # by the number of nodes already in the ego_graph (which is 1).
                    offset_edge_index = agent_edge_index_local + ego_graph.num_nodes

                    # Re-create the ego mask for the combined graph
                    combined_is_ego = torch.zeros(combined_x.shape[0], dtype=torch.bool, device=device)
                    combined_is_ego[0] = True # The first node is always the ego node

                    # Create the final Data object and place it in the correct list position
                    graph_for_agent = Data(
                        x=combined_x,
                        pos=combined_pos,
                        radii=combined_radii,
                        is_ego=combined_is_ego,
                        edge_index=offset_edge_index,
                        env_id=combined_x[:, self.node_feature_map_const['agent_id']]
                    )
                    # --- NEW EGO DEBUG ---
                    if self.debug_mode and agent_batch_idx == 0 and self.env.step_counter < 5:
                        num_ego = torch.sum(graph_for_agent.is_ego).item()
                        print(f"[DEBUG @ Step {self.env.step_counter}, ObsManager] Live graph for Agent 0 created with {num_ego} ego node(s).")
                    # --- END NEW EGO DEBUG ---
                    list_of_graph_data_objects[agent_batch_idx] = graph_for_agent

        obs_profiler.record("graph_generation", time.time() - start_time)

        # --- 7. Final Assembly ---
        start_time = time.time()
        for i_out, original_env_idx_agent in enumerate(alive_agent_env_indices):
            obs_output_all_agents[original_env_idx_agent] = {
                "self": self_obs_tensor_batch[i_out],
                "map": batched_raw_maps[i_out],
                "graph": list_of_graph_data_objects[i_out]
            }

        obs_profiler.record("final_assembly", time.time() - start_time)
        obs_profiler.print_summary()

        return obs_output_all_agents



    def _get_default_observation(self) -> Dict[str, Union[torch.Tensor, Data]]:
        """
        Returns a default observation structure for dead/invalid agents,
        matching the new observation space with 'self', 'map', and 'graph', and ensuring
        the empty graph contains all required attributes.
        """
        self_feat = torch.zeros(self.env_self_obs_dim, device=self.device, dtype=torch.float32)
        map_feat = torch.zeros((self.raw_ch_count, self.raw_map_grid_size, self.raw_map_grid_size),
                               device=self.device, dtype=torch.float32)

        # The policy receives the unified graph, which will have MEM_NODE_FEATURE_DIM + 1 features.
        # We create a default empty graph that matches this final structure.
        unified_graph_feat_dim = MEM_NODE_FEATURE_DIM + 1
        graph_feat = Data(x=torch.empty((0, unified_graph_feat_dim), device=self.device, dtype=torch.float32),
                          edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                          pos=torch.empty((0, 2), device=self.device, dtype=torch.float32),
                          radii=torch.empty((0,), device=self.device, dtype=torch.float32),
                          is_ego=torch.empty((0,), dtype=torch.bool, device=self.device))

        # The final observation for a dead agent doesn't need a separate memory map.
        memory_map_feat = torch.zeros((OCC_CH_COUNT, self.memory_map_grid_size, self.memory_map_grid_size),
                                      device=self.device, dtype=torch.float32)

        return {
            "self": self_feat,
            "map": map_feat,
            "graph": graph_feat,
            "memory_map": memory_map_feat,
        }


    def _generate_final_observations(self, agent_idx_to_observe: Optional[int] = None) -> List[Dict[str, Union[torch.Tensor, Data]]]:
        """
        (Batched) Generates the complete, policy-ready observation for all agents.
        Conditionally uses either the per-agent memory loop or the fully batched memory manager.
        """
        obs_profiler.start_step()

        # 1. Get the live, egocentric observations for all agents
        start_time = time.time()

        # If visualizing a single agent, generate obs for that agent only
        if agent_idx_to_observe is not None:
            # This is a simplified path for visualization/debugging.
            # It's less efficient for training but much faster for a single agent.
            final_obs_list = [self._get_default_observation() for _ in range(self.num_agents)]
            agent_data = self.agents[agent_idx_to_observe]

            if agent_data and agent_data.get('alive'):
                live_obs = self._generate_all_observations_batched_internal(agent_idx_to_observe=agent_idx_to_observe)

                if live_obs:
                    agent_pos = torch.tensor(agent_data['pos'], device=self.device, dtype=torch.float32)
                    agent_radius = agent_data['obs_radius']

                    self.actor_map_states[agent_idx_to_observe].update(
                        live_obs[agent_idx_to_observe]['map'], agent_pos, agent_radius, self.env.step_counter
                    )
                    memory_map, coverage_map = self.actor_map_states[agent_idx_to_observe].get_global_context_map(
                        agent_pos, self.memory_map_grid_size, self.env.step_counter, self.recency_normalization_period, self.env_metadata
                    )

                    # For visualization, also include the unified memory graph for this one agent.
                    if self.generate_memory_graph and self.batched_graph_memory:
                        live_batch = Batch.from_data_list([live_obs[agent_idx_to_observe]['graph']])
                        self.batched_graph_memory.update_batch(live_batch, self.env.step_counter)
                        # Build unified graph for this agent
                        uni_batch = self.batched_graph_memory.get_graph_batch(
                            fovea_agent_positions=agent_pos.unsqueeze(0),
                            fovea_agent_radii=torch.tensor([agent_radius], device=self.device, dtype=torch.float32),
                            live_fovea_graph_list=[live_obs[agent_idx_to_observe]['graph']],
                            current_step=self.env.step_counter,
                            min_cluster_size=self.min_cluster_size,
                            max_cluster_size=self.max_cluster_size,
                            graph_connection_radius_factor=self.graph_connection_radius_factor,
                            cluster_aggressiveness=self.cluster_aggressiveness,
                            cluster_exclusion_radius_factor=self.cluster_exclusion_radius_factor,
                            detailed_clustering_radius_factor=self.detailed_clustering_radius_factor,
                            clustering_frequency=1,  # Set to 1 for smooth GIF generation
                            mem_skeleton_connection_factor=self.mem_skeleton_connection_factor,
                            max_steps=self.max_steps,
                            debug_mode=self.debug
                        )
                        unified_graph = uni_batch.to_data_list()[0] if uni_batch is not None and uni_batch.num_graphs > 0 else live_obs[agent_idx_to_observe]['graph']
                    else:
                        unified_graph = live_obs[agent_idx_to_observe]['graph']
                    final_obs_list[agent_idx_to_observe] = {
                        "self": live_obs[agent_idx_to_observe]['self'],
                        "map": live_obs[agent_idx_to_observe]['map'],
                        "graph": unified_graph,
                        "memory_map": memory_map,
                        "coverage_map": coverage_map # ADDED
                    }
            return final_obs_list

        live_obs_batched = self._generate_all_observations_batched_internal(agent_idx_to_observe=agent_idx_to_observe)
        obs_profiler.record("live_obs_generation", time.time() - start_time)

        start_time = time.time()
        final_obs_list = [self._get_default_observation() for _ in range(self.num_agents)]
        alive_indices = list(live_obs_batched.keys())
        obs_profiler.record("final_obs_init", time.time() - start_time)

        if not alive_indices:
            return final_obs_list

        memory_ops_start = time.time()

        # --- Common setup for both memory systems ---
        agent_positions = []
        agent_radii = []
        live_graphs = []

        for agent_idx in alive_indices:
            agent_data = self.agents[agent_idx]
            agent_positions.append(torch.tensor(agent_data['pos'], device=self.device, dtype=torch.float32))
            agent_radii.append(agent_data['obs_radius'])
            live_graphs.append(live_obs_batched[agent_idx]['graph'])

        # --- UPDATE Actor Map States (BATCHED - V9 OPTIMIZATION) ---
        # Collect all data for batch update
        map_states_to_update = [self.actor_map_states[agent_idx] for agent_idx in alive_indices]
        raw_maps_to_update = [live_obs_batched[agent_idx]['map'] for agent_idx in alive_indices]
        
        # Use batched update to eliminate Python for-loop overhead
        ActorMapState.update_batch(
            map_states_to_update,
            raw_maps_to_update,
            agent_positions,
            agent_radii,
            self.env.step_counter
        )

        # --- Conditional Memory Update and Graph Generation ---
        # --- BATCHED MEMORY PATH ---
        unified_graphs_list = []
        if self.generate_memory_graph and self.batched_graph_memory:
            # 1. Create a single batch of all new observations
            live_obs_batch_for_mem = Batch.from_data_list(live_graphs)

            if self.debug_mode and 0 in alive_indices and self.env.step_counter < 5:
                agent_0_alive_idx = alive_indices.index(0)
                agent_0_mask = (live_obs_batch_for_mem.batch == agent_0_alive_idx)
                agent_0_ego_sum = torch.sum(live_obs_batch_for_mem.x[agent_0_mask, self.node_feature_map_const['is_ego']]).item()
                print(f"[TRACEPOINT 1.5 @ Step {self.env.step_counter}, Pre-MemUpdate] Batched obs for Agent 0 has {agent_0_ego_sum} ego nodes.")
            # --- END NEW EGO DEBUG ---

            # 2. Update memory in one call
            self.batched_graph_memory.update_batch(live_obs_batch_for_mem, self.env.step_counter)

            # 3. Get all memory graphs in one call
            all_agent_pos_t = torch.stack(agent_positions)
            all_agent_radii_t = torch.tensor(agent_radii, device=self.device, dtype=torch.float32)

            # --- DEBUG START: Print inputs to get_graph_batch for Agent 0 ---
            if self.debug_mode and self.env.step_counter < 5 and 0 in alive_indices:
                agent_0_alive_idx = alive_indices.index(0)
                print(f"\n[DEBUG @ Step {self.env.step_counter}, ObsManager -> get_graph_batch]")
                print(f"  - Calling for {len(alive_indices)} alive agents.")
                print(f"  - Agent 0 Position: {_tensor_to_str(all_agent_pos_t[agent_0_alive_idx])}")
                print(f"  - Agent 0 Radius: {all_agent_radii_t[agent_0_alive_idx]:.2f}")
                agent_0_live_graph = live_graphs[agent_0_alive_idx]
                print(f"  - Agent 0 Live Graph: {agent_0_live_graph.num_nodes} nodes, {agent_0_live_graph.num_edges} edges.")
            # --- DEBUG END ---

            unified_graph_batch = self.batched_graph_memory.get_graph_batch(
                fovea_agent_positions=all_agent_pos_t,
                fovea_agent_radii=all_agent_radii_t,
                live_fovea_graph_list=live_graphs,
                current_step=self.env.step_counter,
                min_cluster_size=self.min_cluster_size,
                max_cluster_size=self.max_cluster_size,
                graph_connection_radius_factor=self.graph_connection_radius_factor,
                cluster_aggressiveness=self.cluster_aggressiveness,
                cluster_exclusion_radius_factor=self.cluster_exclusion_radius_factor,
                detailed_clustering_radius_factor=self.detailed_clustering_radius_factor,
                mem_skeleton_connection_factor=self.mem_skeleton_connection_factor,
                clustering_frequency=1,  # Set to 1 for smooth GIF generation
                max_steps=self.max_steps,
                debug_mode=self.debug
            )

            # 4. Split the batch back into a list of individual graphs
            unified_graphs_list = unified_graph_batch.to_data_list() if unified_graph_batch is not None and unified_graph_batch.num_graphs > 0 else []
        else:
            unified_graphs_list = [self._get_default_observation()['graph'] for _ in alive_indices]

        # 5. Assemble final observations
        for i, agent_idx in enumerate(alive_indices):
            if i < len(unified_graphs_list):
                live_obs_agent = live_obs_batched[agent_idx]

                # This is the correct place to normalize the 'last_observed_step' feature
                # for the policy, ensuring it's consistent with the memory map.
                # This block is skipped during visualization runs (agent_idx_to_observe is not None).
                unified_graph = unified_graphs_list[i]
                if agent_idx_to_observe is None and unified_graph.num_nodes > 0:
                    last_seen_abs = unified_graph.x[:, MEM_NODE_FEATURE_DIM] # Corrected index to MEM_NODE_FEATURE_DIM
                    steps_ago = (self.env.step_counter - last_seen_abs).float()
                    
                    # Normalize by the fixed relevance horizon, consistent with memory map.
                    normalized_steps_ago = torch.clamp(steps_ago / self.recency_normalization_period, 0.0, 1.0)
                    
                    # Overwrite the absolute step with the normalized age.
                    unified_graph.x[:, MEM_NODE_FEATURE_DIM] = normalized_steps_ago # Corrected index to MEM_NODE_FEATURE_DIM

                if self.generate_memory_map:
                    memory_map, coverage_map = self.actor_map_states[agent_idx].get_global_context_map(
                        agent_positions[i], self.memory_map_grid_size, self.env.step_counter, self.recency_normalization_period, self.env_metadata
                    )
                else:
                    memory_map = self._get_default_observation()['memory_map']
                    coverage_map = None

                final_obs_list[agent_idx] = {
                    "self": live_obs_agent['self'],
                    "map": live_obs_agent['map'],
                    "graph": unified_graph,
                    "memory_map": memory_map,
                    "coverage_map": coverage_map # ADDED
                }

        if self.debug_mode and self.env.step_counter < 5 and 0 in alive_indices:
            agent_0_final_graph = final_obs_list[0]["graph"]
            print(f"  4. Final Output Graph (Agent 0): {agent_0_final_graph.num_nodes} nodes, {agent_0_final_graph.num_edges} edges.")
        # --- DEBUG END ---

        obs_profiler.record("memory_ops_total", time.time() - memory_ops_start)

        self._cached_final_observations = final_obs_list
        obs_profiler.print_summary()
        return final_obs_list


    def __getattr__(self, name):
        """
        Dynamically delegate attribute access to the env object.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'ObservationManager' object and its 'env' have no attribute '{name}'")


class RenderManager(RenderGenerator):
    """Manages all rendering-related functionality for the environment using delegation."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """
        Dynamically delegate attribute access to the env object.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'RenderManager' object and its 'env' have no attribute '{name}'")

