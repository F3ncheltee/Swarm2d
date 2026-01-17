import numpy as np
import math
import torch
import torch.nn as nn
from typing import List, Dict, Union, Tuple
from collections import defaultdict
from torch_geometric.utils import scatter as pyg_scatter
import pybullet as p
import random

from constants import (
    AGENT_RADIUS, AGENT_BASE_STRENGTH, BEE_SPEED, AGENT_MAX_ENERGY, AGENT_MAX_HEALTH,
    OBS_RADIUS, SENSING_RANGE_FRACTION, HIVE_RADIUS_ASSUMED, OBSTACLE_RADIUS_MIN,
    OBSTACLE_RADIUS_MAX, HIVE_MAX_HEALTH, HIVE_MAX_FOOD, RESOURCE_MAX_SIZE,
    RESOURCE_MIN_SIZE, NODE_FEATURE_MAP, COMBAT_RADIUS, HIVE_ATTACK_RADIUS,
    AGENT_SLOWED_DURATION, REWARD_CONFIG, TEAMMATE_DEATH_VICINITY_RADIUS_FACTOR,
    AGENT_BASE_DAMAGE, AGENT_STRENGTH_DAMAGE_MOD, AGENT_ENERGY_DAMAGE_MOD,
    AGENT_SIZE_DAMAGE_MOD, MIN_AGENT_VELOCITY_FOR_DIRECTION,
    ATTACKER_STATIONARY_DAMAGE_MULTIPLIER, ATTACKER_FACING_THRESHOLD,
    ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER, FLANK_REAR_DEFENDER_FACING_THRESHOLD,
    FLANK_REAR_ATTACK_BONUS_MULTIPLIER, DEFENDER_STRENGTH_MITIGATION_MOD,
    DEFENDER_MASS_MITIGATION_MOD, MIN_DAMAGE_AFTER_MITIGATION,
    AGENT_DAMAGE_STOCHASTICITY, RESPAWN_COOLDOWN, ENERGY_BASE_COST,
    ENERGY_GRAPPLE_COST_MULTIPLIER, ENERGY_MOVEMENT_COST,
    AGENT_DEATH_ENERGY_THRESHOLD, AGENT_DEATH_DROP_RESOURCE_SIZE,
    AGENT_SLOWED_FACTOR, HIVE_DAMAGE_FACTOR, HIVE_DAMAGE_POINTS_PER_BLEED_CHUNK,
    HIVE_BLEED_RESOURCE_SIZE, HIVE_LOST_TIME_THRESHOLD,
    HIVE_CORE_FOOD_TO_SIZE_RATIO, HIVE_CORE_MIN_SIZE, HIVE_CORE_MAX_SIZE
)
from env.helper import normalize_vector, distance
from env.occlusion import generate_gpu_occlusion_field, _generate_global_occlusion_map_cpu


class DataManager(nn.Module):
    def _prepare_global_entity_tensors_for_step(self) -> None:
        """
        DATA MARSHALLING WRAPPER.
        Gathers data from Python objects (lists of dicts) into simple, pure
        tensors and passes them to a JIT-compiled helper function for fast processing.
        This function remains in Python, while the heavy computation is offloaded.
        """
        # --- 1. Marshall data from Python objects to Tensors ---
        device = self.device
        
        # Agent Properties
        agent_props_list = []
        agent_ids_for_map = []
        agent_id_to_prev_pickup_map = {
            agent_data.get('id', -1): action_data.get('pickup', 0)
            for agent_data, action_data in zip(self.agents, self.actions_prev_step)
            if agent_data and action_data
        }
        for agent_data in self.agents:
            if agent_data and agent_data.get('alive'):
                agent_id = agent_data.get('id', -1)
                agent_ids_for_map.append(agent_id)
                props = [
                    *agent_data.get('pos', np.zeros(2)),                           # 0, 1
                    *agent_data.get('vel', np.zeros(2)),                           # 2, 3
                    agent_data.get('team', -1),                                    # 4
                    agent_data.get('agent_radius', AGENT_RADIUS),                  # 5
                    agent_data.get('energy', 0.0), agent_data.get('max_energy', 1.0), # 6, 7
                    agent_data.get('health', 0.0), agent_data.get('max_health', 1.0), # 8, 9
                    agent_data.get('strength', 1.0),                               # 10
                    agent_data.get('speed', BEE_SPEED),                            # 11
                    agent_data.get('obs_radius', self.obs_radius),                 # 12
                    float(agent_data.get('has_resource', False)),                  # 13
                    float(agent_data.get('slowed_timer', 0) > 0),                  # 14
                    float(agent_id),                             # 15 (this is the agent-specific ID)
                    float(agent_id_to_prev_pickup_map.get(agent_id, 0)), # 16
                    float(agent_id),                              # 17 This is the generic entity_unique_id for agents
                    1.0 if agent_data.get('is_grappling', False) else 0.0, # 18
                    1.0 if agent_data.get('is_grappled', False) else 0.0,   # 19
                    agent_data.get('grapple_momentum_bonus', 0.0),      # 20
                    agent_data.get('applied_torque', 0.0),               # 21
                    float(agent_data.get('body_id', -1))                # 22
                ]
                agent_props_list.append(props)
        
        # Resource Properties
        res_props_list = []
        for res_data in self.resources:
            if res_data and not res_data.get('delivered'):
                props = [
                    *res_data.get('pos', np.zeros(2)),            # 0, 1
                    res_data.get('radius_pb', AGENT_RADIUS),      # 2
                    res_data.get('size', 1.0),                    # 3
                    float(res_data.get('cooperative', False)),    # 4
                    float(res_data.get('delivered', False)),      # 5
                    float(res_data.get('id', -1)),                 # 6 resource unique ID
                    float(res_data.get('body_id', -1))              # 7
                ]
                res_props_list.append(props)
                
        # Hive Properties
        hive_props_list = []
        for team_h_id, hive_data in self.hives.items():
            if hive_data:
                props = [
                    *hive_data.get('pos', np.zeros(2)),           # 0, 1
                    hive_data.get('radius', HIVE_RADIUS_ASSUMED), # 2
                    hive_data.get('health', 0.0),                 # 3
                    hive_data.get('owner', team_h_id),            # 4
                    hive_data.get('food_store', 0.0),             # 5
                    float(hive_data.get('state', 'active') == 'destroyed'), # 6
                    float(hive_data.get('id', -1)),                # 7 hive unique ID
                    float(hive_data.get('body_id', -1))             # 8
                ]
                hive_props_list.append(props)

        # Obstacle Properties
        obs_props_list = []
        for obs_data in self.obstacles:
            if obs_data:
                props = [
                    *obs_data.get('pos', np.zeros(2)),            # 0, 1
                    obs_data.get('radius', OBSTACLE_RADIUS_MIN),  # 2
                    float(obs_data.get('id', -1)),                 # 3 obstacle unique ID
                    float(obs_data.get('body_id', -1))              # 4
                ]
                obs_props_list.append(props)

        # Convert lists to tensors, handling empty cases (Update tensor shapes)
        agent_props_t = torch.tensor(agent_props_list, dtype=torch.float32, device=device) if agent_props_list else torch.empty((0, 23), dtype=torch.float32, device=device)
        res_props_t = torch.tensor(res_props_list, dtype=torch.float32, device=device) if res_props_list else torch.empty((0, 8), dtype=torch.float32, device=device)
        hive_props_t = torch.tensor(hive_props_list, dtype=torch.float32, device=device) if hive_props_list else torch.empty((0, 9), dtype=torch.float32, device=device)
        obs_props_t = torch.tensor(obs_props_list, dtype=torch.float32, device=device) if obs_props_list else torch.empty((0, 5), dtype=torch.float32, device=device)
                
        # --- 2. Call the JIT-compiled function ---
        (self.env.current_step_all_pos_t, self.env.current_step_all_feat_t,
        self.env.current_step_all_types_t, self.env.current_step_all_teams_t,
        self.env.current_step_all_radii_t, self.env.current_step_all_coop_t,
        self.env.current_step_all_body_ids_t,
        num_agents) = self._jit_prepare_tensors(
            agent_props_t, res_props_t, hive_props_t, obs_props_t,
            # Pass normalization constants and metadata
            width=float(self.width), height=float(self.height),
            max_speed=self.metadata.get('max_agent_speed_observed', self.bee_speed_config * 1.2),
            max_strength=self.metadata.get('max_agent_strength_observed', AGENT_BASE_STRENGTH * 1.2),
            max_hive_health=self.metadata.get('hive_max_health', HIVE_MAX_HEALTH),
            hive_max_food=self.metadata.get('hive_max_food', HIVE_MAX_FOOD),
            max_resource_size=self.metadata.get('resource_max_size', RESOURCE_MAX_SIZE),
            max_obs_radius=self.metadata.get('obs_radius', OBS_RADIUS) * 1.2,
            max_size_norm_divisor=self.metadata.get('max_size_norm_divisor'),
            # Pass constants
            node_feature_dim=self.node_feature_dim,
            agent_type=float(self.node_type_def_const['agent']),
            resource_type=float(self.node_type_def_const['resource']),
            hive_type=float(self.node_type_def_const['hive']),
            obstacle_type=float(self.node_type_def_const['obstacle']),
            node_feat_map=self.node_feature_map_const
        )
        
        # --- 3. Post-processing (Python-dependent parts) ---
        self.env.current_step_agent_id_to_node_idx_map.clear()
        for i, agent_id_val in enumerate(agent_ids_for_map):
            self.env.current_step_agent_id_to_node_idx_map[agent_id_val] = i

        # --- 4. Generate Occlusion Field (already GPU-based) ---
        self.env.current_step_gpu_occlusion_field = None
        self.env.current_step_cpu_occlusion_grid = None
        if self.env.current_step_all_pos_t.numel() > 0:
            if self.use_gpu_occlusion_in_env:
                self.env.current_step_gpu_occlusion_field = generate_gpu_occlusion_field(
                    self.env.current_step_all_pos_t, self.env.current_step_all_radii_t, self.env.current_step_all_types_t,
                    float(self.width), float(self.height), self.occlusion_field_res_env, self.device
                )
            else: # Fallback to CPU grid
                self.env.current_step_cpu_occlusion_grid = _generate_global_occlusion_map_cpu(
                    self.env.current_step_all_pos_t.cpu().numpy(),
                    self.env.current_step_all_types_t.cpu().numpy(),
                    self.env.current_step_all_radii_t.cpu().numpy(),
                    float(self.width), float(self.height), self.los_grid_cell_size_env
                )

    @staticmethod
    @torch.jit.script
    def _jit_prepare_tensors(
        agent_props_t: torch.Tensor, res_props_t: torch.Tensor,
        hive_props_t: torch.Tensor, obs_props_t: torch.Tensor,
        # Normalization constants
        width: float, height: float, max_speed: float, max_strength: float,
        max_hive_health: float, hive_max_food: float, max_resource_size: float,
        max_obs_radius: float, max_size_norm_divisor: float,
        # Other constants
        node_feature_dim: int, agent_type: float, resource_type: float,
        hive_type: float, obstacle_type: float,
        node_feat_map: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        device = agent_props_t.device
        
        max_val = 0
        for val in node_feat_map.values():
            if val > max_val:
                max_val = val
        actual_node_feature_dim = max_val + 1
        
        num_agents = agent_props_t.shape[0]
        num_resources = res_props_t.shape[0]
        num_hives = hive_props_t.shape[0]
        num_obstacles = obs_props_t.shape[0]

        all_feat_t = torch.zeros((num_agents + num_resources + num_hives + num_obstacles, actual_node_feature_dim), dtype=torch.float32, device=device)
        
        # --- Positions, Radii, Types, Teams (Same as before) ---
        agent_pos = agent_props_t[:, 0:2]
        res_pos = res_props_t[:, 0:2]
        hive_pos = hive_props_t[:, 0:2]
        obs_pos = obs_props_t[:, 0:2]
        all_pos_t = torch.cat([agent_pos, res_pos, hive_pos, obs_pos], dim=0)

        agent_radii = agent_props_t[:, 5]
        res_radii = res_props_t[:, 2]
        hive_radii = hive_props_t[:, 2]
        obs_radii = obs_props_t[:, 2]
        all_radii_t = torch.cat([agent_radii, res_radii, hive_radii, obs_radii], dim=0)

        agent_types = torch.full((num_agents,), agent_type, dtype=torch.long, device=device)
        res_types = torch.full((num_resources,), resource_type, dtype=torch.long, device=device)
        hive_types = torch.full((num_hives,), hive_type, dtype=torch.long, device=device)
        obs_types = torch.full((num_obstacles,), obstacle_type, dtype=torch.long, device=device)
        all_types_t = torch.cat([agent_types, res_types, hive_types, obs_types], dim=0)
        all_feat_t[:, node_feat_map['node_type_encoded']] = all_types_t.float()

        agent_teams = agent_props_t[:, 4].long()
        res_teams = torch.full((num_resources,), -1, dtype=torch.long, device=device)
        hive_teams = hive_props_t[:, 4].long()
        obs_teams = torch.full((num_obstacles,), -1, dtype=torch.long, device=device)
        all_teams_t = torch.cat([agent_teams, res_teams, hive_teams, obs_teams], dim=0)
        all_feat_t[:, node_feat_map['team_id']] = all_teams_t.float()
        
        # --- Unique Entity IDs ---
        agent_unique_ids = agent_props_t[:, 17]
        # --- GRAPPLE FEATURES ---
        agent_is_grappling = agent_props_t[:, 18]
        agent_is_grappled = agent_props_t[:, 19]
        agent_momentum_bonus = agent_props_t[:, 20]
        agent_applied_torque = agent_props_t[:, 21]
        
        res_unique_ids = res_props_t[:, 6] if res_props_t.numel() > 0 else torch.empty(0, device=device)
        hive_unique_ids = hive_props_t[:, 7] if hive_props_t.numel() > 0 else torch.empty(0, device=device)
        obs_unique_ids = obs_props_t[:, 3] if obs_props_t.numel() > 0 else torch.empty(0, device=device)
        all_unique_ids_t = torch.cat([agent_unique_ids, res_unique_ids, hive_unique_ids, obs_unique_ids], dim=0)
        all_feat_t[:, node_feat_map['agent_id']] = all_unique_ids_t
        
        # --- Body IDs ---
        agent_body_ids = agent_props_t[:, 22]
        res_body_ids = res_props_t[:, 7] if res_props_t.numel() > 0 else torch.empty(0, device=device)
        hive_body_ids = hive_props_t[:, 8] if hive_props_t.numel() > 0 else torch.empty(0, device=device)
        obs_body_ids = obs_props_t[:, 4] if obs_props_t.numel() > 0 else torch.empty(0, device=device)
        all_body_ids_t = torch.cat([agent_body_ids, res_body_ids, hive_body_ids, obs_body_ids], dim=0)
        
        # --- Cooperative Flags (Same as before) ---
        res_coop = res_props_t[:, 4] != 0 if res_props_t.numel() > 0 else torch.empty(0, dtype=torch.bool, device=device)
        all_coop_t = torch.cat([
            torch.zeros(num_agents, dtype=torch.bool, device=device),
            res_coop,
            torch.zeros(num_hives + num_obstacles, dtype=torch.bool, device=device)
        ], dim=0)
        
        # --- Feature Population (Same logic, just indices might change slightly) ---
        # Positions normalized by world dimensions - clamp as safety measure since physics might allow out-of-bounds
        all_feat_t[:, node_feat_map['pos_x_norm']] = torch.clamp(all_pos_t[:, 0] / width, 0.0, 1.0)
        all_feat_t[:, node_feat_map['pos_y_norm']] = torch.clamp(all_pos_t[:, 1] / height, 0.0, 1.0)
        # Size normalization uses max_size_norm_divisor which accounts for all entity types
        all_feat_t[:, node_feat_map['size_norm']] = all_radii_t / max_size_norm_divisor
        
        if num_agents > 0:
            start, end = 0, num_agents
            # Velocity can be negative, normalize by max_speed and clamp to [-1, 1] for directional info
            all_feat_t[start:end, node_feat_map['vel_x_norm']] = torch.clamp(agent_props_t[:, 2] / max_speed, -1.0, 1.0)
            all_feat_t[start:end, node_feat_map['vel_y_norm']] = torch.clamp(agent_props_t[:, 3] / max_speed, -1.0, 1.0)
            # Energy and health are ratios, should be in [0, 1] by definition
            all_feat_t[start:end, node_feat_map['energy_norm']] = agent_props_t[:, 6] / agent_props_t[:, 7].clamp(min=1e-6)
            all_feat_t[start:end, node_feat_map['health_norm']] = agent_props_t[:, 8] / agent_props_t[:, 9].clamp(min=1e-6)
            # Strength, speed, obs_radius normalized by theoretical maximums from metadata (accounting for randomization)
            all_feat_t[start:end, node_feat_map['strength_norm']] = agent_props_t[:, 10] / max_strength
            all_feat_t[start:end, node_feat_map['base_speed_norm']] = agent_props_t[:, 11] / max_speed
            all_feat_t[start:end, node_feat_map['obs_radius_norm']] = agent_props_t[:, 12] / max_obs_radius
            all_feat_t[start:end, node_feat_map['is_carrying']] = agent_props_t[:, 13]
            all_feat_t[start:end, node_feat_map['is_slowed']] = agent_props_t[:, 14]
            all_feat_t[start:end, node_feat_map['pickup_action']] = agent_props_t[:, 16]
            # value_or_size_norm is product of normalized values, so should stay in [0, 1]
            all_feat_t[start:end, node_feat_map['value_or_size_norm']] = all_feat_t[start:end, node_feat_map['health_norm']] * all_feat_t[start:end, node_feat_map['strength_norm']]
            
            # --- GRAPPLE FEATURES ---
            all_feat_t[start:end, node_feat_map['is_grappling']] = agent_is_grappling
            all_feat_t[start:end, node_feat_map['is_grappled']] = agent_is_grappled
        
        if num_resources > 0:
            start, end = num_agents, num_agents + num_resources
            all_feat_t[start:end, node_feat_map['is_cooperative']] = res_props_t[:, 4]
            all_feat_t[start:end, node_feat_map['is_delivered']] = res_props_t[:, 5]
            # Resource size normalized by max_resource_size, should be in [0, 1]
            all_feat_t[start:end, node_feat_map['value_or_size_norm']] = res_props_t[:, 3] / max_resource_size
        
        if num_hives > 0:
            start, end = num_agents + num_resources, num_agents + num_resources + num_hives
            # Hive health and food normalized by their maximums, should be in [0, 1]
            all_feat_t[start:end, node_feat_map['health_norm']] = hive_props_t[:, 3] / max_hive_health
            all_feat_t[start:end, node_feat_map['hive_food_norm']] = hive_props_t[:, 5] / hive_max_food
            all_feat_t[start:end, node_feat_map['is_destroyed']] = hive_props_t[:, 6]
            all_feat_t[start:end, node_feat_map['value_or_size_norm']] = all_feat_t[start:end, node_feat_map['health_norm']]

        return all_pos_t, all_feat_t, all_types_t, all_teams_t, all_radii_t, all_coop_t, all_body_ids_t, num_agents

    def _vectorized_proximity_search(self) -> Dict:
        """
        Builds neighbor lists and mappings needed by interactions and combat.
        Uses pybullet.getOverlappingObjects to replace expensive Python loops,
        providing a significant performance increase.
        """
        # 1. Prepare Mappings for Efficient Lookup
        body_id_to_agent = {a['body_id']: a for a in self.agents if a and a.get('alive') and 'body_id' in a}
        body_id_to_resource = {r['body_id']: r for r in self.resources if r and not r.get('delivered') and 'body_id' in r}
        body_id_to_hive = {h['body_id']: h for h in self.hives.values() if h and 'body_id' in h}
        all_body_ids_map = {**body_id_to_agent, **body_id_to_resource, **body_id_to_hive}
        
        # This mapping is crucial for linking physics objects back to their GNN representation
        gnn_idx_to_obj_map = {}
        gnn_idx_to_res_obj_map = {}
        gnn_idx_to_hive_obj_map = {}
        if self.current_step_all_types_t is not None:
            unique_ids = self.current_step_all_feat_t[:, NODE_FEATURE_MAP['agent_id']].long().cpu().numpy()
            for gnn_idx, unique_id in enumerate(unique_ids):
                # Find the corresponding object via its unique ID
                # This is more robust than assuming order.
                obj = next((item for item in self.agents + self.resources + list(self.hives.values()) if item and item.get('id') == unique_id), None)
                if obj:
                    gnn_idx_to_obj_map[gnn_idx] = obj
                    if obj in self.resources: gnn_idx_to_res_obj_map[gnn_idx] = obj
                    if obj in self.hives.values(): gnn_idx_to_hive_obj_map[gnn_idx] = obj

        # Initialize result containers
        agent_resource_map = defaultdict(list)
        agent_enemy_map = defaultdict(list)
        agent_agent_combat_pairs_set = set()
        agent_hive_interaction_pairs_set = set()

        # 2. Iterate Through Agents and Query PyBullet Broadphase
        for agent_idx, agent in enumerate(self.agents):
            if not agent or not agent.get('alive') or 'body_id' not in agent:
                continue

            agent_pos = agent['pos']
            obs_radius = agent.get('obs_radius', self.obs_radius)
            
            # Define the Axis-Aligned Bounding Box (AABB) for the query
            aabb_min = [agent_pos[0] - obs_radius, agent_pos[1] - obs_radius, -1]
            aabb_max = [agent_pos[0] + obs_radius, agent_pos[1] + obs_radius, 1]
            
            # Get all unique body IDs that overlap with the agent's observation area
            try:
                overlapping_body_ids = {item[0] for item in p.getOverlappingObjects(aabb_min, aabb_max, physicsClientId=self.physicsClient)}
            except p.error:
                continue # Skip if PyBullet error occurs

            # 3. Process Overlapping Objects (Narrow Phase)
            for body_id in overlapping_body_ids:
                if body_id == agent['body_id']: continue

                other_obj = all_body_ids_map.get(body_id)
                if not other_obj or 'pos' not in other_obj: continue
                
                dist_sq = np.sum((agent_pos - other_obj['pos'])**2)

                # Agent-Resource Interactions
                if 'size' in other_obj and dist_sq < (obs_radius + other_obj.get('radius_pb', AGENT_RADIUS))**2:
                    if not other_obj.get('delivered'):
                        agent_resource_map[agent_idx].append(other_obj)

                # Agent-Agent Interactions
                elif 'team' in other_obj and other_obj['team'] != agent['team']:
                    if dist_sq < (obs_radius + other_obj.get('agent_radius', AGENT_RADIUS))**2:
                        agent_enemy_map[agent_idx].append(other_obj)
                    if dist_sq < (COMBAT_RADIUS**2):
                        pair = tuple(sorted((agent_idx, other_obj['id'])))
                        agent_agent_combat_pairs_set.add(pair)
                
                # Agent-Hive Interactions
                elif 'owner' in other_obj and dist_sq < (HIVE_ATTACK_RADIUS**2):
                    # Find the GNN index for this hive object
                    hive_gnn_idx = next((idx for idx, obj in gnn_idx_to_hive_obj_map.items() if obj is other_obj), None)
                    if hive_gnn_idx is not None:
                        agent_hive_interaction_pairs_set.add((agent_idx, hive_gnn_idx))

        # 4. Finalize and Format Output Tensors
        agent_agent_pairs = torch.tensor(list(agent_agent_combat_pairs_set), dtype=torch.long, device=self.device).t() if agent_agent_combat_pairs_set else torch.empty((2, 0), dtype=torch.long, device=self.device)
        agent_hive_pairs = torch.tensor(list(agent_hive_interaction_pairs_set), dtype=torch.long, device=self.device).t() if agent_hive_interaction_pairs_set else torch.empty((2, 0), dtype=torch.long, device=self.device)

        return {
            'agent_resource_map': dict(agent_resource_map),
            'agent_enemy_map': dict(agent_enemy_map),
            'agent_agent_combat_pairs': agent_agent_pairs,
            'agent_hive_interaction_pairs': agent_hive_pairs,
            'gnn_idx_to_res_obj_map': gnn_idx_to_res_obj_map,
            'gnn_idx_to_hive_obj_map': gnn_idx_to_hive_obj_map
        }

class CombatManager (nn.Module):
    def _get_effective_combat_strength(self, agent: Dict) -> float:
        """
        (Grapple Penalty) Calculates effective combat strength.
        Significantly penalizes agents that are currently being grappled.
        """
        if not agent or not agent.get('alive', False):
            return 0.0

        # An agent being held can barely fight back.
        grappled_penalty = 0.1 if agent.get('is_grappled', False) else 1.0

        # 1. Base Strength
        base_strength = agent.get('strength', 1.0)

        # 2. Energy Factor
        max_e = agent.get('max_energy', 1.0)
        energy_ratio = agent.get('energy', 0.0) / max_e if max_e > 0 else 0.0
        energy_factor = max(0.1, energy_ratio**0.5) # Strength fades with low energy

        # 3. Carrying Penalty
        carrying_penalty = 0.75 if agent.get('has_resource', False) else 1.0

        # 4. Health Factor
        max_h = agent.get('max_health', 1.0)
        health_ratio = agent.get('health', 0.0) / max_h if max_h > 0 else 0.0
        health_factor = 0.25 + 0.75 * (health_ratio ** 0.75) # Operates at 25% effectiveness at 0 health
        
        effective_strength = base_strength * energy_factor * carrying_penalty * health_factor * grappled_penalty
        
        return max(0.0, effective_strength)

    
    def _apply_damage_and_process_deaths(self, damage_to_apply: Dict[int, float], rewards: List[Dict], potential_killers_map: Dict[int, set], infos: Dict) -> List[int]:
        """Helper to apply accumulated damage, process deaths, and distribute rewards."""
        agents_that_died_this_step = []
        for agent_idx_victim, total_damage in damage_to_apply.items():
            if total_damage <= 0:
                continue

            agent_victim = self.agents[agent_idx_victim]
            if not agent_victim or not agent_victim.get('alive'):
                continue
            
            previous_health = agent_victim['health']
            
            # Apply damage (energy shield logic)
            energy_damage_absorbed = min(agent_victim['energy'], total_damage)
            agent_victim['energy'] -= energy_damage_absorbed
            health_damage = total_damage - energy_damage_absorbed
            if health_damage > 0:
                agent_victim['health'] = max(0.0, agent_victim['health'] - health_damage)
            
            agent_victim['slowed_timer'] = AGENT_SLOWED_DURATION

            # Check for death
            if agent_victim['health'] <= 0 and previous_health > 0:
                agent_victim["alive"] = False
                agents_that_died_this_step.append(agent_idx_victim)
                self.agents_killed_count += 1
                infos['deaths_by_team'][agent_victim['team']] += 1
                
                # Drop resource on death
                agent_death_pos = agent_victim.get('pos')
                if agent_death_pos is not None:
                    self.spawn_manager.resource_spawn._spawn_resource_at_location(target_pos=agent_death_pos, size=REWARD_CONFIG['r_death']['default_value'], cooperative=False)

                # Use the new RewardManager to get scaled rewards
                lose_reward = self.env.reward_manager.get_reward(agent_victim['team'], 'r_combat_lose', base_value=REWARD_CONFIG['r_combat_lose']['default_value'])
                rewards[agent_idx_victim]["r_combat_lose"] += lose_reward
                
                death_penalty = self.env.reward_manager.get_reward(agent_victim['team'], 'r_death', base_value=REWARD_CONFIG['r_death']['default_value'])
                rewards[agent_idx_victim]["r_death"] += death_penalty
                
                killers_indices = potential_killers_map.get(agent_idx_victim, set())
                valid_killers_count = sum(1 for ki in killers_indices if self.agents[ki].get('alive'))
                
                # Vicinity Bonus Radius (e.g., 20.0 units or similar to combat radius)
                # We can reuse TEAMMATE_DEATH_VICINITY_RADIUS_FACTOR or define a new constant
                kill_assist_radius_sq = (self.obs_radius * 0.5)**2 

                if valid_killers_count > 0:
                    win_reward_base = REWARD_CONFIG['r_combat_win']['default_value']
                    
                    # 1. Primary Killers Reward (Direct Damage Dealers)
                    # Reduced slightly to fund the vicinity bonus pool? Or just add on top.
                    # Let's keep the split logic but maybe ensure minimums.
                    reward_per_killer_base = win_reward_base / valid_killers_count
                    for killer_idx in killers_indices:
                        if self.agents[killer_idx].get('alive'):
                            killer_agent = self.agents[killer_idx]
                            infos['kills_by_team'][killer_agent['team']] += 1
                            final_reward = self.env.reward_manager.get_reward(killer_agent['team'], 'r_combat_win', base_value=reward_per_killer_base)
                            rewards[killer_idx]["r_combat_win"] += final_reward
                            
                            # 2. Vicinity/Witness Bonus (New)
                            # Reward ALL teammates near the kill (witnesses/supporters)
                            # This encourages swarming even if they didn't land a hit.
                            if agent_death_pos is not None:
                                for teammate_idx, teammate_obj in enumerate(self.agents):
                                    # Don't reward the killer twice (optional, but cleaner)
                                    if teammate_idx == killer_idx or not teammate_obj.get('alive'):
                                        continue
                                    
                                    if teammate_obj['team'] == killer_agent['team']:
                                        dist_sq = np.sum((agent_death_pos - teammate_obj['pos'])**2)
                                        if dist_sq < kill_assist_radius_sq:
                                            # Assist reward is a fraction of the kill reward (e.g., 20%)
                                            assist_base = win_reward_base * 0.2
                                            final_assist = self.env.reward_manager.get_reward(teammate_obj['team'], 'r_combat_win', base_value=assist_base)
                                            # We add it to 'r_combat_win' bucket or a new one? Reusing helps analysis.
                                            rewards[teammate_idx]["r_combat_win"] += final_assist

                
                # Teammate vicinity penalty
                victim_team = agent_victim['team']
                if agent_death_pos is not None:
                    # <<< Renamed loop variable for clarity from 'agent' to 'teammate_obj' >>>
                    for teammate_idx, teammate_obj in enumerate(self.agents):
                        if teammate_obj and teammate_obj.get('alive') and teammate_obj['team'] == victim_team:
                            # Now calculate the radius_sq INSIDE the loop, using the correct teammate_obj
                            vicinity_radius_sq = (teammate_obj.get('obs_radius', self.obs_radius) * TEAMMATE_DEATH_VICINITY_RADIUS_FACTOR)**2
                            dist_sq = np.sum((agent_death_pos - teammate_obj['pos'])**2)
                            if dist_sq < vicinity_radius_sq:
                                vicinity_penalty = self.env.reward_manager.get_reward(victim_team, 'r_teammate_lost_nearby', base_value=REWARD_CONFIG['r_teammate_lost_nearby']['default_value'])
                                rewards[teammate_idx]["r_teammate_lost_nearby"] += vicinity_penalty

        return agents_that_died_this_step
    
    @staticmethod
    @torch.jit.script
    def _calculate_combat_damage_batched(
        att_props: torch.Tensor, 
        def_props: torch.Tensor,
        AGENT_RADIUS: float,
        AGENT_BASE_DAMAGE: float,
        AGENT_STRENGTH_DAMAGE_MOD: float,
        AGENT_ENERGY_DAMAGE_MOD: float,
        AGENT_SIZE_DAMAGE_MOD: float,
        MIN_AGENT_VELOCITY_FOR_DIRECTION: float,
        ATTACKER_STATIONARY_DAMAGE_MULTIPLIER: float,
        ATTACKER_FACING_THRESHOLD: float,
        ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER: float,
        FLANK_REAR_DEFENDER_FACING_THRESHOLD: float,
        FLANK_REAR_ATTACK_BONUS_MULTIPLIER: float,
        DEFENDER_STRENGTH_MITIGATION_MOD: float,
        DEFENDER_MASS_MITIGATION_MOD: float,
        MIN_DAMAGE_AFTER_MITIGATION: float,
        AGENT_DAMAGE_STOCHASTICITY: float
    ) -> torch.Tensor:
        """
        (V2-Batched & JIT-Compatible) JIT-compiled function to calculate combat damage for all pairs in parallel.
        Accepts two tensors, one for attackers and one for defenders, with identical feature layouts.
        Feature Layout: [pos_x, pos_y, vel_x, vel_y, strength, energy, max_energy,
                         health, max_health, radius, mass, team_id]
        """
        # Unpack properties for clarity
        att_pos, def_pos = att_props[:, 0:2], def_props[:, 0:2]
        att_vel, def_vel = att_props[:, 2:4], def_props[:, 2:4]
        att_str, def_str = att_props[:, 4], def_props[:, 4]
        att_nrg, def_nrg = att_props[:, 5], def_props[:, 5]
        att_max_nrg, def_max_nrg = att_props[:, 6], def_props[:, 6]
        att_hea, def_hea = att_props[:, 7], def_props[:, 7]
        att_max_hea, def_max_hea = att_props[:, 8], def_props[:, 8]
        att_rad, def_rad = att_props[:, 9], def_props[:, 9]
        att_mass, def_mass = att_props[:, 10], def_props[:, 10]
        
        # --- Effective Combat Strength (Vectorized) ---
        att_energy_ratio = (att_nrg / att_max_nrg.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        att_health_ratio = (att_hea / att_max_hea.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        att_eff_strength = att_str * (att_energy_ratio.pow(0.5).clamp(min=0.1)) * (0.25 + 0.75 * att_health_ratio.pow(0.75))
        def_energy_ratio = (def_nrg / def_max_nrg.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        def_health_ratio = (def_hea / def_max_hea.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        def_eff_strength = def_str * (def_energy_ratio.pow(0.5).clamp(min=0.1)) * (0.25 + 0.75 * def_health_ratio.pow(0.75))

        # 1. Base Damage Potential
        energy_factor_att = (att_nrg / att_max_nrg.clamp(min=1e-6)).clamp(min=0.05)
        size_factor_att = att_rad / AGENT_RADIUS
        base_damage = AGENT_BASE_DAMAGE * (1.0 + AGENT_STRENGTH_DAMAGE_MOD * att_eff_strength + \
                                           AGENT_ENERGY_DAMAGE_MOD * energy_factor_att + AGENT_SIZE_DAMAGE_MOD * size_factor_att)

        # 2. Attacker Facing / Engagement Angle
        vec_att_to_def = def_pos - att_pos
        dist_att_def = torch.linalg.norm(vec_att_to_def, dim=1).clamp(min=1e-6)
        vec_att_to_def_norm = vec_att_to_def / dist_att_def.unsqueeze(1)
        
        att_speed = torch.linalg.norm(att_vel, dim=1)
        is_moving_mask = att_speed > MIN_AGENT_VELOCITY_FOR_DIRECTION
        
        engagement_mult = torch.full_like(att_speed, ATTACKER_STATIONARY_DAMAGE_MULTIPLIER)
        if is_moving_mask.any():
            att_vel_norm = att_vel[is_moving_mask] / att_speed[is_moving_mask].unsqueeze(1)
            dot_prod = torch.sum(att_vel_norm * vec_att_to_def_norm[is_moving_mask], dim=1)
            not_facing_mult = torch.where(dot_prod < ATTACKER_FACING_THRESHOLD, ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER, 1.0)
            engagement_mult[is_moving_mask] = not_facing_mult
        modified_damage = base_damage * engagement_mult

        # 3. Flank/Rear Attack Bonus
        def_speed = torch.linalg.norm(def_vel, dim=1)
        is_def_moving_mask = def_speed > MIN_AGENT_VELOCITY_FOR_DIRECTION
        flank_mult = torch.ones_like(def_speed)
        if is_def_moving_mask.any():
            def_vel_norm = def_vel[is_def_moving_mask] / def_speed[is_def_moving_mask].unsqueeze(1)
            dot_prod_flank = torch.sum(def_vel_norm * -vec_att_to_def_norm[is_def_moving_mask], dim=1)
            flank_mult[is_def_moving_mask] = torch.where(dot_prod_flank < FLANK_REAR_DEFENDER_FACING_THRESHOLD, FLANK_REAR_ATTACK_BONUS_MULTIPLIER, 1.0)
        modified_damage *= flank_mult

        # 4. Defender Mitigations
        strength_mit_val = DEFENDER_STRENGTH_MITIGATION_MOD * def_eff_strength
        mass_mit_val = DEFENDER_MASS_MITIGATION_MOD * (def_mass - 1.0).clamp(min=0.0)
        total_mit_factor = ((1.0 - strength_mit_val) * (1.0 - mass_mit_val)).clamp(min=MIN_DAMAGE_AFTER_MITIGATION)
        final_damage_pre_stoch = modified_damage * total_mit_factor
        
        # 5. Stochasticity
        stoch_mod = 1.0 + torch.rand_like(final_damage_pre_stoch) * (2 * AGENT_DAMAGE_STOCHASTICITY) - AGENT_DAMAGE_STOCHASTICITY
        final_damage = (final_damage_pre_stoch * stoch_mod).clamp(min=0.0)
        
        return final_damage

    def process_combat(self, rewards, agent_agent_pairs: torch.Tensor, infos: Dict):
        """
        (JIT Compatible) Resolves all combat for a step.
        Data preparation is vectorized, and constants are passed to the JIT-compiled damage calculator.
        """
        potential_killers_map = defaultdict(set)
        total_damage_per_defender = torch.zeros(len(self.agents), dtype=torch.float32, device=self.device)

        # --- Standard Combat Section ---
        if agent_agent_pairs.numel() > 0:
            att_indices_env_all = agent_agent_pairs[0]
            def_indices_env_all = agent_agent_pairs[1]

            # 1. Filter out pairs involving dead agents
            alive_mask_att = torch.tensor([self.agents[i]['alive'] for i in att_indices_env_all], dtype=torch.bool, device=self.device)
            alive_mask_def = torch.tensor([self.agents[i]['alive'] for i in def_indices_env_all], dtype=torch.bool, device=self.device)
            valid_combat_mask = alive_mask_att & alive_mask_def

            if valid_combat_mask.any():
                att_indices_env = att_indices_env_all[valid_combat_mask]
                def_indices_env = def_indices_env_all[valid_combat_mask]

                # 2. Vectorized Data Marshalling
                unique_combatant_indices = torch.unique(torch.cat([att_indices_env, def_indices_env]))
                
                combatant_props_list = [
                    [
                        *self.agents[i]['pos'], *self.agents[i]['vel'], self.agents[i]['strength'],
                        self.agents[i]['energy'], self.agents[i]['max_energy'], self.agents[i]['health'],
                        self.agents[i]['max_health'], self.agents[i]['agent_radius'], self.agents[i]['mass'],
                        self.agents[i]['team']
                    ] for i in unique_combatant_indices.tolist()
                ]
                all_combatant_props_t = torch.tensor(combatant_props_list, dtype=torch.float32, device=self.device)

                map_env_idx_to_combat_idx = {env_idx.item(): i for i, env_idx in enumerate(unique_combatant_indices)}
                gather_indices_att = torch.tensor([map_env_idx_to_combat_idx[i.item()] for i in att_indices_env], device=self.device, dtype=torch.long)
                gather_indices_def = torch.tensor([map_env_idx_to_combat_idx[i.item()] for i in def_indices_env], device=self.device, dtype=torch.long)
                
                attacker_props_t = torch.gather(all_combatant_props_t, 0, gather_indices_att.unsqueeze(1).expand(-1, all_combatant_props_t.shape[1]))
                defender_props_t = torch.gather(all_combatant_props_t, 0, gather_indices_def.unsqueeze(1).expand(-1, all_combatant_props_t.shape[1]))

                # 3. Calculate all damages in one batched call
                damages_per_pair_t = self._calculate_combat_damage_batched(
                    attacker_props_t, defender_props_t,
                    # --- Pass all required global constants ---
                    float(AGENT_RADIUS), float(AGENT_BASE_DAMAGE), float(AGENT_STRENGTH_DAMAGE_MOD),
                    float(AGENT_ENERGY_DAMAGE_MOD), float(AGENT_SIZE_DAMAGE_MOD), float(MIN_AGENT_VELOCITY_FOR_DIRECTION),
                    float(ATTACKER_STATIONARY_DAMAGE_MULTIPLIER), float(ATTACKER_FACING_THRESHOLD),
                    float(ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER), float(FLANK_REAR_DEFENDER_FACING_THRESHOLD),
                    float(FLANK_REAR_ATTACK_BONUS_MULTIPLIER), float(DEFENDER_STRENGTH_MITIGATION_MOD),
                    float(DEFENDER_MASS_MITIGATION_MOD), float(MIN_DAMAGE_AFTER_MITIGATION),
                    float(AGENT_DAMAGE_STOCHASTICITY)
                )

                # 4. Aggregate damage per defender
                total_damage_per_defender += pyg_scatter(
                    src=damages_per_pair_t, index=def_indices_env,
                    dim=0, dim_size=len(self.agents), reduce='add'
                )
                
                # Aggregate damage dealt by team
                for i in range(att_indices_env.shape[0]):
                    attacker_team = self.agents[att_indices_env[i].item()]['team']
                    infos['damage_by_team'][attacker_team] += damages_per_pair_t[i].item()

                # 5. Distribute continuous combat rewards
                # The base value for continuous reward is the damage dealt.
                continuous_reward_base = damages_per_pair_t
                continuous_reward_per_attacker = pyg_scatter(
                    src=continuous_reward_base, index=att_indices_env,
                    dim=0, dim_size=len(self.agents), reduce='add'
                )

                for i in range(len(self.agents)):
                    if continuous_reward_per_attacker[i] > 0:
                        agent_obj = self.agents[i]
                        # The magnitude is now controlled by the multiplier in the config.
                        final_reward = self.env.reward_manager.get_reward(
                            agent_obj['team'], 
                            'r_combat_continuous', 
                            base_value=continuous_reward_per_attacker[i].item()
                        )
                        rewards[i]["r_combat_continuous"] += final_reward
                
                # Populate potential killers from standard combat
                for i in range(att_indices_env.shape[0]):
                    if damages_per_pair_t[i] > 0:
                        potential_killers_map[def_indices_env[i].item()].add(att_indices_env[i].item())

        # --- Grapple Crush Damage ---
        for agent_idx, agent in enumerate(self.agents):
            if agent and agent.get('is_grappling'):
                target_idx = agent.get('grappled_agent_id')
                if target_idx is not None and self.agents[target_idx].get('alive'):
                    crush_damage = self.grapple_crush_damage_rate * (agent.get('strength', 1.0))
                    target_agent = self.agents[target_idx]

                    # --- Positional Critical Hit Logic ---
                    crit_chance = self.grapple_crit_chance
                    vec_grappler_to_target = normalize_vector(target_agent['pos'] - agent['pos'])
                    target_speed = np.linalg.norm(target_agent.get('vel', np.zeros(2)))
                    if target_speed > MIN_AGENT_VELOCITY_FOR_DIRECTION:
                        normalized_target_vel = normalize_vector(target_agent['vel'])
                        if np.dot(normalized_target_vel, vec_grappler_to_target) > 0.7:
                            crit_chance *= self.grapple_rear_crit_bonus_multiplier

                    if crit_chance > 0 and random.random() < crit_chance:
                        crush_damage *= self.grapple_crit_multiplier
                        agent['last_event'] = 'CRITICAL_HIT'
                    
                    agent['damage_dealt_this_step'] = crush_damage
                    total_damage_per_defender[target_idx] += crush_damage
                    potential_killers_map[target_idx].add(agent_idx)
                    infos['damage_by_team'][agent['team']] += crush_damage
        
        # --- Grapple Struggle Damage (from target to grappler) ---
        grappler_map = {a.get('grappled_agent_id'): a for a in self.agents if a and a.get('is_grappling')}
        for agent_idx, agent in enumerate(self.agents):
            if agent and agent.get('is_grappled'):
                grappler_obj = grappler_map.get(agent_idx)
                if grappler_obj and grappler_obj.get('alive'):
                    # --- Refined Struggle Damage with Counter-Play ---
                    # The target must be actively trying to move to deal damage.
                    target_vel_norm = np.linalg.norm(agent.get('vel', np.zeros(2)))
                    
                    if target_vel_norm > MIN_AGENT_VELOCITY_FOR_DIRECTION:
                        target_move_dir = normalize_vector(agent['vel'])
                        
                        # The grappler can mitigate damage by moving *with* the struggling target.
                        grappler_vel_norm = np.linalg.norm(grappler_obj.get('vel', np.zeros(2)))
                        grappler_move_dir = normalize_vector(grappler_obj['vel']) if grappler_vel_norm > MIN_AGENT_VELOCITY_FOR_DIRECTION else np.zeros(2)

                        # Alignment: 1.0 if moving together, -1.0 if moving opposite, 0.0 if perpendicular.
                        alignment = np.dot(target_move_dir, grappler_move_dir)

                        # Damage is highest when movements are opposed (alignment is -1).
                        # Mitigation factor maps alignment from [-1, 1] to a damage multiplier of [1.0, 0.25]
                        mitigation_factor = 1.0 - (0.75 * ((alignment + 1.0) / 2.0))
                        
                        # Base damage now gives equal weight to strength and mass.
                        base_struggle_damage = self.grapple_struggle_damage_rate * \
                            (agent.get('strength', 1.0) * 0.5 + agent.get('mass', 1.0) * 0.5)
                        
                        final_struggle_damage = base_struggle_damage * mitigation_factor
                        
                        if final_struggle_damage > 0:
                            total_damage_per_defender[grappler_obj['id']] += final_struggle_damage
                            potential_killers_map[grappler_obj['id']].add(agent_idx)
                            agent['damage_dealt_this_step'] = final_struggle_damage
                            infos['damage_by_team'][agent['team']] += final_struggle_damage

        # 6. Apply damage and process deaths
        damage_to_apply_dict = {i: total_damage_per_defender[i].item() for i in torch.where(total_damage_per_defender > 0)[0].tolist()}
        
        agents_that_died = self._apply_damage_and_process_deaths(damage_to_apply_dict, rewards, potential_killers_map, infos)
        for dead_agent_idx in agents_that_died:
            if 0 <= dead_agent_idx < len(self.agents):
                self.agents[dead_agent_idx]["cooldown"] = RESPAWN_COOLDOWN



class StatusManager (nn.Module):
    def _get_effective_push_strength(self, agent):
        """Calculates effective strength considering energy."""
        if not agent['alive']: return 0.0
        energy_factor = max(0.05, agent['energy'] / agent['max_energy'])
        # Can add other factors like health or status effects here if needed
        return agent['strength'] * energy_factor

    def _get_agent_state(self, agent):
        """Safely gets agent position and velocity from PyBullet."""
        if not agent or not agent.get('alive'):
            # Return cached or zeros if dead/invalid agent dict
            return agent.get('pos', np.zeros(2)) if isinstance(agent, dict) else np.zeros(2), \
                   agent.get('vel', np.zeros(2)) if isinstance(agent, dict) else np.zeros(2)
        try:
            # Ensure body_id exists and is valid
            body_id = agent.get('body_id')
            if body_id is None or body_id < 0:
                 # print(f"Warning: Invalid body_id {body_id} for agent {agent.get('id')}. Returning cached state.")
                 return agent.get('pos', np.zeros(2)), agent.get('vel', np.zeros(2))

            pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.physicsClient)
            vel, _ = p.getBaseVelocity(body_id, physicsClientId=self.physicsClient)
            # Update cache DIRECTLY here - IMPORTANT for KD-tree accuracy if built before physics
            agent["pos"] = np.array(pos[:2])
            agent["vel"] = np.array(vel[:2])
            return agent["pos"], agent["vel"]
        except p.error as e:
            # print(f"Warning: PyBullet error getting state for agent {agent.get('id', 'N/A')}: {e}. Returning cached/zero state.")
            # Return last known position/velocity from cache, or zeros if cache is also invalid
            return agent.get('pos', np.zeros(2)), agent.get('vel', np.zeros(2))

    def _get_grip_strength_components(self, agent: Dict) -> Dict[str, float]:
        """
        Returns a detailed breakdown of all components contributing to grip strength.
        This is the definitive calculation, including fatigue and torque penalties.
        """
        components = {
            'base_grip': 0.0, 'energy_factor': 0.0, 'strength_factor': 0.0,
            'health_factor': 0.0, 'fatigue_factor': 1.0, 'torque_penalty': 1.0,
            'final_grip': 0.0
        }
        if not agent or not agent.get('alive'):
            return components

        # 1. Base Grip
        base_grip = self.pb_agent_constraint_max_force * self.current_agent_interaction_force_scale
        components['base_grip'] = base_grip

        # 2. Factors based on agent's state
        energy_ratio = np.clip(agent.get('energy', 0.0) / agent.get('max_energy', 1.0), 0.0, 1.0)
        components['energy_factor'] = energy_ratio ** 0.75

        components['strength_factor'] = agent.get('strength', 1.0)

        health_ratio = np.clip(agent.get('health', 0.0) / agent.get('max_health', 1.0), 0.0, 1.0)
        components['health_factor'] = 0.1 + 0.9 * health_ratio # Grip drops off more sharply with health

        # 3. Penalties from the struggle itself
        if agent.get('is_grappling'):
            # Fatigue Penalty
            fatigue = agent.get('grapple_fatigue', 0.0)
            components['fatigue_factor'] = max(0.1, 1.0 - (self.grapple_fatigue_rate * fatigue))

            # Torque Penalty
            target_agent = self.agents[agent.get('grappled_agent_id')] if agent.get('grappled_agent_id') is not None else None
            if target_agent:
                grappler_torque = agent.get('applied_torque', 0.0)
                target_torque = target_agent.get('applied_torque', 0.0)
                if np.sign(grappler_torque) != np.sign(target_torque) and abs(target_torque) > 0.1:
                    ratio = np.clip(abs(target_torque) / (abs(grappler_torque) + 1e-6), 0.0, 1.0)
                    components['torque_penalty'] = 1.0 - (self.grapple_torque_escape_strength * ratio)

        # Calculate final grip
        components['final_grip'] = (
            components['base_grip'] * components['energy_factor'] *
            components['strength_factor'] * components['health_factor'] *
            components['fatigue_factor'] * components['torque_penalty']
        )
        return components

    def _get_effective_grip_strength(self, agent: Dict) -> float:
        """
        Enhanced grip strength calculation with momentum bonuses and synergy effects.
        This value is used for both resource carrying and agent-agent grappling.
        """
        if not agent or not agent.get('alive'):
            return 0.0
        
        # For grappling, use the new detailed component calculation
        if agent.get('is_grappling'):
            components = self._get_grip_strength_components(agent)
            return max(0.0, components['final_grip'])

        # --- Original logic for resource carrying (simpler) ---
        base_grip = self.pb_resource_constraint_max_force * self.current_resource_interaction_force_scale
        res_obj = agent.get('resource_obj')
        if res_obj and res_obj.get("cooperative"):
            base_grip = self.pb_coop_resource_constraint_max_force * self.current_resource_interaction_force_scale

        energy_ratio = np.clip(agent.get('energy', 0.0) / agent.get('max_energy', 1.0), 0.0, 1.0)
        energy_factor = energy_ratio ** 0.75

        strength_factor = agent.get('strength', AGENT_BASE_STRENGTH) / AGENT_BASE_STRENGTH

        health_ratio = np.clip(agent.get('health', 0.0) / agent.get('max_health', 1.0), 0.0, 1.0)
        health_factor = 0.5 + 0.5 * health_ratio

        calculated_force = base_grip * energy_factor * strength_factor * health_factor
        return max(0.0, calculated_force)

    def _get_grip_strength_info_batched(self, agents_batch: List[Dict], device: torch.device) -> torch.Tensor:
        """
        Calculates grip strength for a batch of agents in a vectorized manner.
        """
        num_agents = len(agents_batch)
        if num_agents == 0:
            return torch.empty((0, 1), device=device)

        # 1. Extract data into NumPy arrays first for efficiency, as agent data is heterogeneous.
        # We filter for only alive agents to avoid errors with missing data.
        alive_agents = [a for a in agents_batch if a and a.get('alive')]
        num_alive = len(alive_agents)
        if num_alive == 0:
            return torch.zeros((num_agents, 1), device=device)

        energies = np.array([a.get('energy', 0.0) for a in alive_agents], dtype=np.float32)
        max_energies = np.array([a.get('max_energy', 1.0) for a in alive_agents], dtype=np.float32)
        strengths = np.array([a.get('strength', AGENT_BASE_STRENGTH) for a in alive_agents], dtype=np.float32)
        vels = np.array([a.get('vel', np.zeros(2)) for a in alive_agents], dtype=np.float32)
        max_speeds = np.array([a.get('speed', self.bee_speed_config) for a in alive_agents], dtype=np.float32)
        healths = np.array([a.get('health', 0.0) for a in alive_agents], dtype=np.float32)
        max_healths = np.array([a.get('max_health', 1.0) for a in alive_agents], dtype=np.float32)

        # The synergy_bonus part still requires a loop due to Python object references.
        synergy_bonuses = np.zeros(num_alive, dtype=np.float32)
        for i, agent in enumerate(alive_agents):
            if agent.get('has_resource', False):
                res_obj = agent.get('resource_obj')
                if res_obj and res_obj.get("cooperative"):
                    num_carriers = len(res_obj.get("carriers", []))
                    if num_carriers > 1:
                        synergy_bonuses[i] = min(0.5, (num_carriers - 1) * 0.15)
        
        # 2. Convert to Torch tensors for GPU-accelerated computation.
        energies_t = torch.from_numpy(energies).to(device)
        max_energies_t = torch.from_numpy(max_energies).to(device)
        strengths_t = torch.from_numpy(strengths).to(device)
        vels_t = torch.from_numpy(vels).to(device)
        max_speeds_t = torch.from_numpy(max_speeds).to(device)
        healths_t = torch.from_numpy(healths).to(device)
        max_healths_t = torch.from_numpy(max_healths).to(device)
        synergy_bonuses_t = torch.from_numpy(synergy_bonuses).to(device)

        # 3. Perform all numerical calculations in a single vectorized operation.
        # Using .div() with in-place replacement for max_energies_t to avoid division by zero.
        energy_ratios = torch.clamp(energies_t.div(max_energies_t.clamp(min=1e-6)), 0.0, 1.0)
        energy_factors = energy_ratios.pow(0.75)

        strength_factors = strengths_t / AGENT_BASE_STRENGTH

        current_speeds = torch.linalg.norm(vels_t, dim=1)
        momentum_bonuses = (current_speeds / max_speeds_t.clamp(min=1e-6)) * 0.2

        health_ratios = torch.clamp(healths_t.div(max_healths_t.clamp(min=1e-6)), 0.0, 1.0)
        health_factors = 0.5 + 0.5 * health_ratios

        max_possible_grip = self.pb_resource_constraint_max_force * self.current_resource_interaction_force_scale
        
        current_grips = max_possible_grip * energy_factors * strength_factors * (1.0 + momentum_bonuses) * (1.0 + synergy_bonuses_t) * health_factors
        
        if max_possible_grip > 0:
            grip_strength_norms = torch.clamp(current_grips / max_possible_grip, max=1.0)
        else:
            grip_strength_norms = torch.zeros_like(current_grips)

        # Create a result tensor for all agents (including dead ones) and fill it.
        # This ensures the output shape matches the input batch size.
        final_grip_strengths = torch.zeros(num_agents, device=device)
        alive_indices = [i for i, a in enumerate(agents_batch) if a and a.get('alive')]
        if alive_indices:
            final_grip_strengths[torch.tensor(alive_indices, device=device)] = grip_strength_norms

        return final_grip_strengths.unsqueeze(1)

    def set_movement_force_scale(self, new_scale):
        """ Sets the movement force scale multiplier for agent movement. """
        self.current_movement_force_scale = float(new_scale)
    
    def set_resource_interaction_force_scale(self, new_scale):
        """ Sets the interaction force scale for resource carrying. """
        self.current_resource_interaction_force_scale = float(new_scale)

    def set_agent_interaction_force_scale(self, new_scale):
        """ Sets the interaction force scale for agent grappling. """
        self.current_agent_interaction_force_scale = float(new_scale)

    def _vectorized_state_decay_and_respawn(self, rewards: List[Dict[str, float]]):
        """ Handles all passive state changes like energy decay, cooldowns, and respawning."""
        if not self.agents:
            return

        # Single pass over all agents
        for agent_idx, agent in enumerate(self.agents):
            if not agent:
                continue

            # --- Handle dead agents ---
            if not agent['alive']:
                agent['cooldown'] -= 1
                if agent['cooldown'] <= 0:
                    self.spawn_manager.respawn_agent(agent)
                continue # Skip to next agent if dead

            # --- Handle alive agents ---
            # Apply energy decay
            grapple_cost = ENERGY_BASE_COST * ENERGY_GRAPPLE_COST_MULTIPLIER if agent.get('is_grappling', False) else 0.0
            energy_decay = ENERGY_MOVEMENT_COST * agent.get('last_desired_force', 0.0) + ENERGY_BASE_COST + grapple_cost
            agent['energy'] = max(0, agent['energy'] - energy_decay)
            
            # Apply health drain if out of energy
            if agent['energy'] <= AGENT_DEATH_ENERGY_THRESHOLD:
                agent['health'] -= 0.1
            
            # Tick down status effects
            if agent.get('slowed_timer', 0) > 0:
                agent['slowed_timer'] -= 1

            # Check for death this step
            if agent['health'] <= 0:
                if agent.get('pos') is not None:
                    self.spawn_manager.resource_spawn._spawn_resource_at_location(agent['pos'], AGENT_DEATH_DROP_RESOURCE_SIZE, False)
                self._cleanup_agent_attachments(agent)
                agent["alive"] = False
                agent["cooldown"] = RESPAWN_COOLDOWN

    def _get_effective_escape_strength(self, agent: Dict) -> float:
        """
        Calculates the effective strength of a grappled agent trying to escape.
        This combines physical stats with active struggle (movement and torque).
        """
        if not agent or not agent.get('alive') or not agent.get('is_grappled'):
            return 0.0

        # 1. Base physical potential (health, energy, strength)
        energy_ratio = np.clip(agent.get('energy', 0.0) / agent.get('max_energy', 1.0), 0.0, 1.0)
        energy_factor = energy_ratio ** 0.5 # Less penalty than grip, more about burst

        health_ratio = np.clip(agent.get('health', 0.0) / agent.get('max_health', 1.0), 0.0, 1.0)
        health_factor = 0.25 + 0.75 * health_ratio # Can still struggle when hurt

        # Scaled to be in a similar range to grip forces for comparison in logs
        base_potential = agent.get('strength', 1.0) * energy_factor * health_factor * 1000 

        # 2. Active struggle components
        # Raw pulling force (is heavily nerfed, but still contributes)
        pulling_force = agent.get('debug_applied_force', 0.0) 

        # Torque is a major component of breaking free
        # High scaler to show its importance in logs
        torque_component = abs(agent.get('applied_torque', 0.0)) * self.grapple_torque_escape_strength * 2000 

        return base_potential + pulling_force + torque_component


class InteractionManager (nn.Module):
    
    def _iterative_apply_interaction_logic(self, actions, rewards, agent_resource_map, agent_enemy_map, infos) -> List[Dict]:
        """(Iterative) Handles non-batchable interactions like constraint creation."""
        grapple_break_events = []
        for idx, agent in enumerate(self.agents):
            if not agent['alive']: continue
            act = actions[idx]
            was_grappled = agent.get('is_grappled', False)

            if act.get("pickup") == 1 and not agent.get("has_resource") and not agent.get("is_grappling") and not was_grappled:
                self._handle_pickup_or_grapple_action(idx, agent, rewards, agent_resource_map, agent_enemy_map, infos)
            elif act.get("pickup") == 2:
                if was_grappled: grapple_break_events.append({'escaper_id': agent['id']})
                self._cleanup_agent_attachments(agent)

        # Update dynamic constraints
        for agent in self.agents:
            if agent.get('is_grappling') and agent.get('grapple_constraint_id') is not None:
                target_id = agent.get('grappled_agent_id')
                if not (target_id is not None and 0 <= target_id < len(self.agents) and self.agents[target_id] and self.agents[target_id].get('alive')):
                    self._cleanup_agent_attachments(agent)
                    continue

                target_agent = self.agents[target_id]

                # --- Net Grip Force Calculation ---
                # 1. Grappler's potential grip
                eff_grip_grappler = self._get_effective_grip_strength(agent)
                bonus = agent.get('grapple_momentum_bonus', 0.0) * self.grapple_momentum_decay
                agent['grapple_momentum_bonus'] = bonus if bonus > 1.0 else 0.0
                total_grappler_force = eff_grip_grappler + agent['grapple_momentum_bonus']

                # 2. Target's active counter-grip (skill-based)
                target_action = actions[target_id]
                eff_grip_target_counter = 0.0
                if target_action.get("pickup") == 2: # "Break Free" action
                    eff_grip_target_counter = self._get_effective_grip_strength(target_agent) * self.grappled_agent_counter_grip_scale
                
                # 3. Net force is the difference
                net_grip_force = total_grappler_force - eff_grip_target_counter

                # 4. Check for grapple break
                if net_grip_force <= 1.0: # Break if grip is negligible
                    grapple_break_events.append({'escaper_id': target_id})
                    self._cleanup_agent_attachments(agent)
                    continue

                try:
                    cap = self.pb_agent_constraint_max_force * 1.5
                    new_force = min(net_grip_force, cap)
                    p.changeConstraint(agent['grapple_constraint_id'], maxForce=new_force)
                    agent['grapple_last_set_force'] = new_force # Update for logging
                except p.error:
                    self._cleanup_agent_attachments(agent)

        # Also refresh resource carrier constraint forces using up-to-date grip
        for agent in self.agents:
            if agent.get('has_resource') and agent.get('resource_obj'):
                res_obj_car = agent['resource_obj']
                cid = res_obj_car.get('carrier_constraints', {}).get(agent['id'])
                if cid is not None:
                    try:
                        cap = self.pb_resource_constraint_max_force * 1.5
                        if res_obj_car.get("cooperative"):
                            cap = self.pb_coop_resource_constraint_max_force * 1.5
                        eff_grip = self._get_effective_grip_strength(agent)
                        p.changeConstraint(cid, maxForce=min(eff_grip, cap))
                    except p.error:
                        self._cleanup_agent_attachments(agent)
        
        return grapple_break_events



class AttachmentManager (nn.Module):
    def _cleanup_resource_attachments(self, resource_obj: Dict):
        """ Removes ALL of a resource's constraints and marks it as delivered."""
        # Iterates through the dictionary of constraints and removes each one.
        if "carrier_constraints" in resource_obj:
            for cid in resource_obj["carrier_constraints"].values():
                try:
                    p.removeConstraint(cid, self.physicsClient)
                except p.error:
                    pass  # Ignore if already gone
        
        # Clear all carrier info and mark as delivered
        resource_obj.update({"delivered": True, "carriers": [], "carrier_constraints": {}})
        
        # Remove the resource body from the simulation
        if resource_obj.get("body_id") is not None:
            try:
                p.removeBody(resource_obj["body_id"], self.physicsClient)
            except p.error:
                pass
            resource_obj['body_id'] = None
        
        # Reset the attachment reward flag so respawned/recycled resources (if any) can be rewarding again
        # Note: If resources are fully respawned as new dicts, this isn't strictly necessary, but good safety.
        resource_obj["attachment_reward_given"] = False
    
    def _cleanup_agent_attachments(self, agent: Dict):
        """
        Unified cleanup for an agent releasing a resource OR a grapple.
        Handles individual constraints for each carrier.
        """
        # --- Release a carried resource ---
        if agent.get("has_resource") and (res_obj := agent.get("resource_obj")):
            # Remove this agent from the list of carriers on the resource object.
            if 'carriers' in res_obj:
                res_obj["carriers"] = [c for c in res_obj.get("carriers", []) if c.get('id') != agent['id']]

            # Remove this agent's specific constraint from the simulation
            if 'carrier_constraints' in res_obj and agent['id'] in res_obj['carrier_constraints']:
                constraint_id = res_obj['carrier_constraints'].pop(agent['id'])
                try:
                    p.removeConstraint(constraint_id, self.physicsClient)
                except p.error:
                    pass # Ignore error if constraint was already gone
            
            # Put the resource on a short cooldown to prevent instant re-attachment
            res_obj["pickup_cooldown"] = 5
            # If no carriers, revert dynamics to static values for free resources
            if res_obj.get('body_id') is not None and len(res_obj.get('carriers', [])) == 0:
                try:
                    p.changeDynamics(
                        res_obj['body_id'], -1,
                        linearDamping=self.pb_res_damping_static,
                        angularDamping=self.pb_res_damping_static,
                        lateralFriction=self.pb_res_friction_static,
                        physicsClientId=self.physicsClient
                    )
                except p.error:
                    pass
        
        # --- Release a grapple it initiated ---
        if agent.get("is_grappling") and agent.get("grapple_constraint_id") is not None:
            if (target_agent_id := agent.get('grappled_agent_id')) is not None and 0 <= target_agent_id < len(self.agents):
                if target_agent := self.agents[target_agent_id]:
                    target_agent['is_grappled'] = False
            
            try:
                p.removeConstraint(agent['grapple_constraint_id'], self.physicsClient)
            except p.error:
                pass

        # --- Reset the agent's own state variables ---
        agent["has_resource"] = False
        agent["resource_obj"] = None
        agent['is_grappling'] = False
        agent['grappled_agent_id'] = None
        agent['grapple_constraint_id'] = None
        agent['is_grappled'] = False
        agent['grapple_fatigue'] = 0.0 # Reset fatigue

    def _handle_pickup_or_grapple_action(self, idx, agent, rewards, agent_resource_map, agent_enemy_map, infos):
        """
        Helper to process a single agent's pickup/grapple attempt.
        This version is proximity-based: the agent will interact with the closest valid object,
        be it a resource or an enemy.
        """
        if self.env.debug_mode:
            print(f"[ENV DEBUG] Agent {agent['id']} sent pickup action. Checking nearby objects (proximity-based)...")

        agent_radius = agent.get('agent_radius', AGENT_RADIUS)
        
        # Find the closest valid resource
        closest_res = None
        min_res_dist_sq = float('inf')
        nearby_resources = agent_resource_map.get(idx, [])
        for res_obj in nearby_resources:
            # Check if non-cooperative resource is already occupied
            if not res_obj.get("cooperative", False) and len(res_obj.get("carriers", [])) > 0:
                if self.env.debug_mode: print(f"Agent {agent['id']} ignores Res {res_obj['id']} (Already carried & Non-Coop)")
                continue
            
            # Check if cooperative resource is overcrowded (prevent farming)
            if res_obj.get("cooperative", False):
                required = res_obj.get("required_agents", 1)
                # Allow max 1 extra helper to account for sync issues or weak agents, but generally cap it.
                # Actually, strictly capping at required_agents is the safest way to prevent "pile-ons" for rewards.
                if len(res_obj.get("carriers", [])) >= required:
                    if self.env.debug_mode: print(f"Agent {agent['id']} ignores Res {res_obj['id']} (Coop Full: {len(res_obj['carriers'])}/{required})")
                    continue

            # DEBUG: Print why resources are skipped if close enough
            pickup_radius = (agent_radius + res_obj.get('radius_pb', 0)) * 1.1
            dist_sq = np.sum((agent['pos'] - res_obj['pos'])**2)
            
            if dist_sq < (pickup_radius**2):
                if res_obj.get("pickup_cooldown", 0) > 0:
                     if self.env.debug_mode: print(f"Agent {agent['id']} ignores Res {res_obj['id']} due to cooldown {res_obj['pickup_cooldown']}")
                else:
                    if dist_sq < min_res_dist_sq:
                        min_res_dist_sq = dist_sq
                        closest_res = res_obj

        # Find the closest valid enemy
        closest_enemy = None
        min_enemy_dist_sq = float('inf')
        nearby_enemies = agent_enemy_map.get(idx, [])
        for enemy in nearby_enemies:
            if not enemy.get('is_grappled'): # Can't grapple someone already grappled
                grapple_radius = (agent_radius + enemy.get('agent_radius', AGENT_RADIUS)) * self.grapple_initiation_distance_multiplier
                dist_sq = np.sum((agent['pos'] - enemy['pos'])**2)
                if dist_sq < (grapple_radius**2) and dist_sq < min_enemy_dist_sq:
                    min_enemy_dist_sq = dist_sq
                    closest_enemy = enemy

        # Decide which object to interact with based on proximity
        if closest_res and (not closest_enemy or min_res_dist_sq <= min_enemy_dist_sq):
            # Moved reward calculation logic INTO _attach_resource_to_agent 
            # to ensure it happens exactly when the constraint is created.
            if self.env.debug_mode:
                print(f"[ENV DEBUG] Agent {agent['id']} chose to PICKUP closest resource {closest_res['id']} (dist^2: {min_res_dist_sq:.2f}).")
            self._attach_resource_to_agent(agent, closest_res, rewards)
        elif closest_enemy:
            if self.env.debug_mode:
                print(f"[ENV DEBUG] Agent {agent['id']} chose to GRAPPLE closest enemy {closest_enemy['id']} (dist^2: {min_enemy_dist_sq:.2f}).")
            self._attempt_grapple(agent, closest_enemy, infos) # Pass the specific enemy
        else:
            if self.env.debug_mode:
                print(f"[ENV DEBUG] Agent {agent['id']} found no valid targets in range for pickup or grapple.")

    def _attach_resource_to_agent(self, agent: Dict, resource_obj: Dict, rewards: List[Dict]):
        """Creates a PyBullet constraint between an agent and a resource."""
        resource_obj.setdefault("carriers", []).append({'id': agent['id'], 'team': agent['team']})
        agent["has_resource"], agent["resource_obj"] = True, resource_obj
        if self.env.debug_mode:
            print(f"[ENV DEBUG] Agent {agent['id']} state updated. has_resource={agent.get('has_resource')}") # DEBUG
        
        if len(resource_obj["carriers"]) == 1:
            self.env.resources_picked_count += 1
            resource_obj["target_hive"] = agent["team"]
            th_pos = self.hives.get(agent["team"], {}).get("pos")
            if th_pos is not None:
                resource_obj["prev_distance"] = distance(resource_obj['pos'], th_pos)

        # ONE-TIME REWARD EVENT (Logic: Only reward FIRST successful attachment)
        # Check if this resource has already given an attachment reward
        if not resource_obj.get("attachment_reward_given", False):
            attachment_reward = self.env.reward_manager.get_reward(agent['team'], 'r_attachment', base_value=REWARD_CONFIG['r_attachment']['default_value'])
            rewards[agent['id']]["r_attachment"] += attachment_reward
            resource_obj["attachment_reward_given"] = True # Mark as rewarded
            
            if self.env.debug_mode:
                 print(f"[REWARD DEBUG] Agent {agent['id']} (Team {agent['team']}) got r_attachment={attachment_reward:.2f} for Res {resource_obj['id']} (FIRST TIME)")
        else:
             if self.env.debug_mode:
                 print(f"[REWARD DEBUG] Agent {agent['id']} (Team {agent['team']}) SKIPPED r_attachment for Res {resource_obj['id']} (Already Given)")

        try:
            # --- Corrected Surface-to-Surface Constraint Creation ---
            # Get current world positions and orientations
            agent_pos_3d, agent_orn = p.getBasePositionAndOrientation(agent['body_id'], self.physicsClient)
            res_pos_3d, res_orn = p.getBasePositionAndOrientation(resource_obj['body_id'], self.physicsClient)

            # World-space vector from agent to resource
            vec_agent_to_res_w = np.array(res_pos_3d[:2]) - np.array(agent_pos_3d[:2])
            norm = np.linalg.norm(vec_agent_to_res_w)

            # If objects are on top of each other, use agent's forward direction
            if norm < 1e-6:
                rot_matrix = p.getMatrixFromQuaternion(agent_orn)
                forward_vec = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
                vec_agent_to_res_w = forward_vec[:2]
            else:
                vec_agent_to_res_w /= norm

            # Calculate pivot points in WORLD coordinates
            pivot_on_agent_w = np.array(agent_pos_3d[:2]) + vec_agent_to_res_w * (agent['agent_radius'] * 0.05)
            pivot_on_res_w = np.array(res_pos_3d[:2]) - vec_agent_to_res_w * (resource_obj['radius_pb'] * 0.05)

            # Transform world pivot points to LOCAL coordinates for the constraint
            inv_agent_pos, inv_agent_orn = p.invertTransform(agent_pos_3d, agent_orn)
            inv_res_pos, inv_res_orn = p.invertTransform(res_pos_3d, res_orn)

            child_pivot, _ = p.multiplyTransforms(inv_agent_pos, inv_agent_orn, [pivot_on_agent_w[0], pivot_on_agent_w[1], 0], [0,0,0,1])
            parent_pivot, _ = p.multiplyTransforms(inv_res_pos, inv_res_orn, [pivot_on_res_w[0], pivot_on_res_w[1], 0], [0,0,0,1])
            
            cid = p.createConstraint(
                parentBodyUniqueId=resource_obj['body_id'],
                parentLinkIndex=-1,
                childBodyUniqueId=agent['body_id'],
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=parent_pivot,
                childFramePosition=child_pivot,
                physicsClientId=self.physicsClient
            )
            cap = self.pb_resource_constraint_max_force * 1.5
            if resource_obj.get("cooperative"):
                cap = self.pb_coop_resource_constraint_max_force * 1.5
            eff = self._get_effective_grip_strength(agent)
            p.changeConstraint(cid, maxForce=min(eff, cap))
            resource_obj.setdefault('carrier_constraints', {})[agent['id']] = cid
            # Switch resource dynamics to dynamic values while carried to reduce lag
            try:
                p.changeDynamics(
                    resource_obj['body_id'], -1,
                    linearDamping=self.pb_res_damping_dynamic,
                    angularDamping=self.pb_res_damping_dynamic,
                    lateralFriction=self.pb_res_friction_dynamic,
                    physicsClientId=self.physicsClient
                )
            except p.error:
                pass
        except p.error as e:
            print(f"PyBullet Error creating resource constraint for agent {agent['id']}: {e}")
            resource_obj["carriers"] = [c for c in resource_obj.get("carriers", []) if c.get('id') != agent['id']]
            agent["has_resource"], agent["resource_obj"] = False, None
    
    def _attempt_grapple(self, agent: Dict, closest_enemy: Dict, infos: Dict):
        """Processes an agent's attempt to grapple a nearby enemy."""
        if not closest_enemy:
            if self.env.debug_mode:
                print(f"[ENV DEBUG] Grapple attempt by Agent {agent['id']} failed: No valid enemy provided.")
            return

        agent_radius = agent.get('agent_radius', AGENT_RADIUS)
        enemy_radius = closest_enemy.get('agent_radius', AGENT_RADIUS)
        
        # Final check of grapple distance
        grapple_radius = (agent_radius + enemy_radius) * self.grapple_initiation_distance_multiplier
        dist_sq = np.sum((agent['pos'] - closest_enemy['pos'])**2)

        if dist_sq < grapple_radius**2:
            if self.env.debug_mode:
                print(f"[ENV DEBUG] Grapple SUCCESS: Agent {agent['id']} grappling Enemy {closest_enemy['id']}.")
            
            infos['grapples_initiated_by_team'][agent['team']] += 1

            # --- Momentum Bonus ---
            # Bonus for grappling a fast-moving target.
            momentum_bonus = (np.linalg.norm(closest_enemy.get('vel',np.zeros(2))) * closest_enemy.get('mass',1.0)) * self.grapple_momentum_bonus_scale
            
            # --- Constraint Creation ---
            # It now calculates the pivot point between the agents for a stable grapple
            # This prevents weird physics artifacts from off-center constraints.
            pos_grappler = agent['pos']
            pos_enemy = closest_enemy['pos']
            pivot_world = (pos_grappler + pos_enemy) / 2.0
            pivot_grappler_local = [pivot_world[0] - pos_grappler[0], pivot_world[1] - pos_grappler[1], 0]
            pivot_enemy_local = [pivot_world[0] - pos_enemy[0], pivot_world[1] - pos_enemy[1], 0]

            cid = p.createConstraint(
                parentBodyUniqueId=agent['body_id'],
                parentLinkIndex=-1,
                childBodyUniqueId=closest_enemy['body_id'],
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1], # Specifies the axis, though less important for P2P
                parentFramePosition=pivot_grappler_local,
                childFramePosition=pivot_enemy_local,
                physicsClientId=self.physicsClient
            )

            final_grip = self._get_effective_grip_strength(agent)
            p.changeConstraint(cid, maxForce=final_grip, physicsClientId=self.physicsClient)
            closest_enemy['is_grappled'] = True
            agent.update({'is_grappling':True, 'grappled_agent_id':closest_enemy['id'], 'grapple_constraint_id':cid, 'grapple_last_set_force':final_grip, 'grapple_momentum_bonus':momentum_bonus, 'grapple_fatigue': 0.0})
        else:
             if self.env.debug_mode:
                print(f"[ENV DEBUG] Grapple attempt by Agent {agent['id']} failed: Enemy {closest_enemy['id']} moved out of range.")

    def _update_and_correct_physics_states(self):
        """Updates agent/resource dicts with post-simulation states and corrects Z-axis drift."""
        for agent in self.agents:
            if agent['alive']:
                try:
                    pos,ori=p.getBasePositionAndOrientation(agent['body_id'],self.physicsClient)
                    vel,ang=p.getBaseVelocity(agent['body_id'],self.physicsClient)
                    
                    # Correct Z-position to keep agents on the plane
                    p.resetBasePositionAndOrientation(agent['body_id'],[pos[0],pos[1],agent['agent_radius']], ori, self.physicsClient)
                    
                    # Zero out any Z-velocity and non-Z angular velocity to enforce 2D
                    p.resetBaseVelocity(agent['body_id'], linearVelocity=[vel[0], vel[1], 0], angularVelocity=[0, 0, ang[2]])

                    agent['pos'],agent['vel']=np.array(pos[:2]),np.array(vel[:2])

                except p.error: agent['pos'],agent['vel']=None,None
        for res in self.resources:
            if res and not res.get('delivered') and res.get('body_id') is not None:
                try:
                    pos,ori=p.getBasePositionAndOrientation(res['body_id'],self.physicsClient)
                    vel,ang=p.getBaseVelocity(res['body_id'], self.physicsClient)
                    p.resetBasePositionAndOrientation(res['body_id'],[pos[0],pos[1],res['radius_pb']],ori,self.physicsClient)
                    p.resetBaseVelocity(res['body_id'], linearVelocity=[vel[0], vel[1], 0], angularVelocity=[0, 0, ang[2]])
                    res['pos']=np.array(pos[:2])
                    res['vel']=np.array(vel[:2])
                except p.error: 
                    res['pos']=None
                    res['vel']=None

    def _process_grapple_outcomes(self, rewards: List[Dict], grapple_break_events: List[Dict], infos: Dict):
        """Processes grapple torque rewards and break events."""
        for agent in self.agents:
            if agent.get('is_grappling') and agent.get('grappled_agent_id') is not None and self.agents[agent['grappled_agent_id']].get('alive'):
                net_torque = abs(agent.get('applied_torque',0.0)) - abs(self.agents[agent['grappled_agent_id']].get('applied_torque',0.0))
                if net_torque > 0:
                    # The base value for torque win is the net torque itself.
                    final_reward = self.env.reward_manager.get_reward(agent['team'], 'r_torque_win', base_value=net_torque)
                    rewards[agent['id']]['r_torque_win'] += final_reward
            if not agent.get('is_grappling') and agent.get('grapple_constraint_id') is not None:
                target_id = agent.get('grappled_agent_id')
                if target_id is not None and not self.agents[target_id].get('is_grappled'):
                    grapple_break_events.append({'escaper_id':target_id})
                self._cleanup_agent_attachments(agent)

        for event in grapple_break_events:
            if event.get('escaper_id') is not None:
                escaper_agent = self.agents[event['escaper_id']]
                infos['grapples_broken_by_team'][escaper_agent['team']] += 1
                break_reward = self.env.reward_manager.get_reward(escaper_agent['team'], 'r_grapple_break', base_value=REWARD_CONFIG['r_grapple_break']['default_value'])
                rewards[event['escaper_id']]['r_grapple_break'] += break_reward


class MovementManager (nn.Module): 
    def _vectorized_apply_movement_forces(self, actions: List[Dict]):
        """(OPTIMIZED) Calculates and applies all movement forces to PyBullet in a single pass."""
        alive_indices = [i for i, a in enumerate(self.agents) if a and a.get('alive')]
        if not alive_indices:
            return

        resource_map = {r['id']: r for r in self.resources if r}

        for i in alive_indices:
            agent = self.agents[i]
            action = actions[i]
            
            # --- Calculate Effective Force Magnitude ---
            # Base Energy Ratio
            raw_energy_ratio = agent['energy'] / max(1e-6, agent['max_energy'])
            
            # [MODIFICATION] Desperation Move / Adrenaline:
            # If health is good (>20%), ensure we can always move at least 30% speed.
            # This prevents "healthy but tired" agents from being totally stuck.
            health_ratio = agent['health'] / max(1e-6, agent['max_health'])
            
            min_mobility = 0.05 # Absolute minimum (crawling)
            if health_ratio > 0.20:
                min_mobility = 0.30 # Desperation move
            
            energy_ratio = max(min_mobility, raw_energy_ratio)
            
            slowed_multiplier = AGENT_SLOWED_FACTOR if agent.get('slowed_timer', 0) > 0 else 1.0
            
            # --- Grapple Penalty ---
            # Reduce movement force significantly if grappling or grappled.
            # This makes the grapple constraint the dominant force.
            grapple_multiplier = 0.1 if agent.get('is_grappling') or agent.get('is_grappled') else 1.0

            # Calculate cooperative penalty if carrying a coop resource
            coop_penalty = 1.0
            if agent.get('has_resource', False):
                res_obj = agent.get('resource_obj')
                if res_obj and res_obj.get("cooperative"):
                    num_alive_carriers = sum(1 for c in res_obj.get("carriers", []) if self.agents[c['id']].get('alive'))
                    required_agents = res_obj.get("required_agents", 1)
                    if required_agents > 0:
                        coop_penalty = np.clip((num_alive_carriers / required_agents) ** 1.3, 0.1, 1.0)
            
            force_magnitude = self.current_movement_force_scale * agent.get('speed', self.bee_speed_config) * energy_ratio * slowed_multiplier * coop_penalty * grapple_multiplier
            agent["last_desired_force"] = force_magnitude

            # --- Calculate Force Vector ---
            mov_vec = action['movement']
            norm = np.linalg.norm(mov_vec)
            if norm > 1e-6:
                mov_vec = mov_vec / norm
            
            force_vector = mov_vec * force_magnitude
            
            # --- Torque Calculation ---
            # If grappling, calculate torque generated by movement force relative to the target.
            # Torque = (Pos_Self - Pos_Target) x Force_Vector
            agent['applied_torque'] = 0.0 # Reset
            if agent.get('is_grappling') and agent.get('grappled_agent_id') is not None:
                target_id = agent['grappled_agent_id']
                if 0 <= target_id < len(self.agents) and self.agents[target_id].get('alive'):
                    target_pos = self.agents[target_id]['pos']
                    self_pos = agent['pos']
                    
                    # Vector from Target (Pivot) to Self
                    rx = self_pos[0] - target_pos[0]
                    ry = self_pos[1] - target_pos[1]
                    
                    # 2D Cross Product (rx*Fy - ry*Fx)
                    raw_torque = rx * force_vector[1] - ry * force_vector[0]
                    
                    # Normalize/Scale for reward
                    # Raw torque can be ~2000 (Force 200 * Dist 10). 
                    # Scale by 0.001 to get range ~0.0 to 2.0
                    agent['applied_torque'] = raw_torque * 0.001

            # --- Apply Force in PyBullet ---
            try:
                p.applyExternalForce(
                    agent['body_id'], -1, 
                    [force_vector[0], force_vector[1], 0], 
                    [0, 0, 0], 
                    p.WORLD_FRAME, 
                    physicsClientId=self.physicsClient
                )
            except p.error:
                continue # Ignore if body no longer exists


class HiveCombatManager (nn.Module):
    

    def process_hive_engagements(self, rewards: List[Dict[str, float]], agent_hive_pairs: torch.Tensor, gnn_idx_to_hive_obj_map: Dict[int, Dict], infos: Dict):
        """
        Resolves hive engagements using a pre-computed list of
        proximate agent-hive pairs and a robust mapping from GNN node index to hive objects.
        """
        if agent_hive_pairs.numel() == 0:
            return

        # 1. Populate the interaction dictionary from the pre-computed pairs and robust map
        hive_interactions = {team_id: {'allied_defenders': [], 'enemy_attackers': [], 'original_team_members': []} for team_id in self.hives.keys()}
        
        agent_indices_env = agent_hive_pairs[0].tolist()
        hive_node_indices = agent_hive_pairs[1].tolist()

        for agent_env_idx, hive_node_idx in zip(agent_indices_env, hive_node_indices):
            agent_obj = self.agents[agent_env_idx]
            hive_obj = gnn_idx_to_hive_obj_map.get(hive_node_idx)

            if not agent_obj or not agent_obj.get('alive') or not hive_obj:
                continue

            original_hive_team_id = hive_obj["original_team_id"]
            current_owner_team_id = hive_obj.get("owner", original_hive_team_id)
            interaction_dict = hive_interactions.get(original_hive_team_id)
            if not interaction_dict: continue

            if agent_obj['team'] == original_hive_team_id:
                interaction_dict['original_team_members'].append(agent_env_idx)
            
            if agent_obj['team'] == current_owner_team_id:
                interaction_dict['allied_defenders'].append(agent_env_idx)
            else:
                interaction_dict['enemy_attackers'].append(agent_env_idx)

        # 2. Process engagements for each hive 
        for team_id_hive_original, interaction_data in hive_interactions.items():
            hive = self.hives.get(team_id_hive_original)
            if not hive or hive.get('pos') is None: continue
            
            hive_pos = hive["pos"]; current_owner_team_id = hive.get("owner", team_id_hive_original)
            allied_defenders = interaction_data['allied_defenders']; enemy_attackers = interaction_data['enemy_attackers']
            original_team_members = interaction_data['original_team_members']

            if hive.get("state") == "active":
                if enemy_attackers:
                    allied_strength = sum(self._get_effective_combat_strength(self.agents[i]) for i in allied_defenders)
                    enemy_strength = sum(self._get_effective_combat_strength(self.agents[i]) for i in enemy_attackers)

                    if enemy_strength > allied_strength:
                        total_damage = (enemy_strength - allied_strength) * HIVE_DAMAGE_FACTOR
                        dmg_food, dmg_health = 0, 0
                        rem_dmg = total_damage
                        if hive["food_store"] > 0 and rem_dmg > 0:
                            dmg_food = min(hive["food_store"], rem_dmg); hive["food_store"] -= dmg_food; rem_dmg -= dmg_food
                        if hive["food_store"] <= 1e-6 and rem_dmg > 0 and hive["health"] > 0:
                            dmg_health = min(hive["health"], rem_dmg); hive["health"] -= dmg_health
                        
                        hive["bleed_damage_accumulator_food"] = hive.get("bleed_damage_accumulator_food", 0.0) + dmg_food
                        while hive["bleed_damage_accumulator_food"] >= HIVE_DAMAGE_POINTS_PER_BLEED_CHUNK:
                            angle=np.random.uniform(0,2*math.pi); dist=HIVE_ATTACK_RADIUS+np.random.uniform(5,10)
                            pos=np.clip([hive_pos[0]+dist*math.cos(angle),hive_pos[1]+dist*math.sin(angle)],0,self.width)
                            self.spawn_manager.resource_spawn._spawn_resource_at_location(target_pos=pos, size=HIVE_BLEED_RESOURCE_SIZE, cooperative=False)
                            hive["bleed_damage_accumulator_food"] -= HIVE_DAMAGE_POINTS_PER_BLEED_CHUNK
                        
                        if dmg_health > 0 and enemy_strength > 0:
                            for attacker_idx in enemy_attackers:
                                share = self._get_effective_combat_strength(self.agents[attacker_idx]) / enemy_strength if enemy_strength > 0 else 0
                                # The base value is the damage dealt. Multiplier is in the config.
                                base_value = (dmg_food + dmg_health) * share
                                attacker_team = self.agents[attacker_idx]['team']
                                infos['hive_damage_by_team'][attacker_team] += base_value
                                final_reward = self.env.reward_manager.get_reward(self.agents[attacker_idx]['team'], 'r_hive_attack_continuous', base_value)
                                rewards[attacker_idx]["r_hive_attack_continuous"] += final_reward
                        
                        if hive["health"] <= 0:
                            core_size = np.clip(hive.get("max_food_capacity_at_destruction", 100.)*HIVE_CORE_FOOD_TO_SIZE_RATIO, HIVE_CORE_MIN_SIZE, HIVE_CORE_MAX_SIZE)
                            self.spawn_manager.resource_spawn._spawn_resource_at_location(hive_pos, core_size, True)
                            hive.update({"state": "destroyed", "respawn_timer": HIVE_LOST_TIME_THRESHOLD, "lost_counter": hive.get("lost_counter",0)+1, "bleed_damage_accumulator_food":0.0})
                            
                            # Team-wide penalty for the team that lost its hive
                            penalty_val = self.env.reward_manager.get_reward(current_owner_team_id, 'r_hive_destroyed_penalty', base_value=REWARD_CONFIG['r_hive_destroyed_penalty']['default_value'])
                            for agent in self.agents:
                                if agent and agent.get('alive') and agent['team'] == current_owner_team_id:
                                    rewards[agent['id']]["r_hive_destroyed_penalty"] += penalty_val
            
            elif hive.get("state") == "destroyed":
                hive["respawn_timer"] = max(0, hive["respawn_timer"]-1)
                if hive["respawn_timer"] <= 0:
                    original_strength = sum(self._get_effective_combat_strength(self.agents[i]) for i in original_team_members)
                    enemy_strengths = defaultdict(float)
                    # This check still needs to be broad as any enemy team can capture
                    for agent in self.agents:
                        if agent and agent.get('alive') and agent.get('pos') is not None and agent['team'] != team_id_hive_original:
                            if np.sum((agent['pos'] - hive_pos)**2) < HIVE_ATTACK_RADIUS**2:
                                enemy_strengths[agent['team']] += self._get_effective_combat_strength(agent)
                    
                    strongest_enemy_id, max_enemy_strength = max(enemy_strengths.items(), key=lambda i: i[1]) if enemy_strengths else (-1, 0.0)
                    max_health, max_food= self.metadata.get('hive_max_health', 100.), self.metadata.get('hive_max_food', 100.)
                    if original_strength > 0 and original_strength > max_enemy_strength:
                        # Original team must have a CLEAR advantage to rebuild. A tie is not enough.
                        hive.update({"state": "active", "health": max_health * 0.5, "food_store": max_food * 0.25, "owner": team_id_hive_original})
                        rebuild_reward_val = self.env.reward_manager.get_reward(team_id_hive_original, 'r_hive_rebuild', base_value=REWARD_CONFIG['r_hive_rebuild']['default_value'])
                        for agent_idx in original_team_members: 
                            rewards[agent_idx]["r_hive_rebuild"] += rebuild_reward_val
                    elif max_enemy_strength > 0 and max_enemy_strength > original_strength:
                        hive.update({"state":"active", "health":max_health, "food_store":max_food*0.5, "owner":strongest_enemy_id})
                        capture_reward_val = self.env.reward_manager.get_reward(strongest_enemy_id, 'r_hive_capture', base_value=REWARD_CONFIG['r_hive_capture']['default_value'])
                        for agent in self.agents:
                            if agent and agent.get('alive') and agent['team'] == strongest_enemy_id: 
                                rewards[agent['id']]["r_hive_capture"] += capture_reward_val
                    else:
                        hive["respawn_timer"] = HIVE_LOST_TIME_THRESHOLD // 2
                        





