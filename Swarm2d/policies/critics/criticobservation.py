
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
from typing import Optional, List, Dict, Tuple
import numpy as np

from env.observations import ActorMapState
from constants import (
    GLOBAL_CUE_DIM, NODE_TYPE, NODE_FEATURE_MAP, MEM_NODE_FEAT_IDX,
    OBS_RADIUS, CLUSTER_CELL_SIZE, RAW_CH_COUNT, CRITIC_GRID_SIZE,
    OCC_CH_COUNT, MEM_NODE_FEATURE_DIM, REWARD_COMPONENT_KEYS
)
from training.scatter_helpers import (
    aggregate_grid_data_scatter,
    calculate_density_scatter,
    calculate_vec_to_target_scatter
)

class CriticObservationGenerator(torch.jit.ScriptModule):
    def __init__(self, node_feature_map: Dict[str, int]):
        super().__init__()
        # Store the dictionary as a constant for TorchScript
        self.NODE_FEATURE_MAP = node_feature_map
    
    @torch.jit.script_method
    def _generate_global_occ_map_gpu_scatter(
        self,
        team_id: int,
        all_pos: torch.Tensor, all_types: torch.Tensor, all_teams: torch.Tensor, all_feat: torch.Tensor,
        current_step: int, grid_size: int, world_to_map_scale: float,
        width: float, height: float, max_steps: int, own_hive_pos: Optional[torch.Tensor],
        occ_ch_count: int,
        ch_obstacle: int, ch_ally_density: int, ch_enemy_density: int, ch_resource_density: int,
        ch_last_seen_ally: int, ch_last_seen_enemy: int, ch_last_seen_resource: int,
        ch_explored: int, ch_cert_obstacle: int, ch_cert_resource: int,
        ch_vec_hive_x: int, ch_vec_hive_y: int, ch_step_norm: int, ch_avg_team_energy: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        (V5 JIT-Optimized & GPU-Native) Generates a perfect, ground-truth map for a Global Critic using scatter ops.
        """
        occ_map = torch.zeros((occ_ch_count, grid_size, grid_size), device=device, dtype=torch.float32)
        
        if all_pos.numel() == 0:
            return occ_map.unsqueeze(0)

        map_coords_x = (all_pos[:, 0] / world_to_map_scale).long().clamp(0, grid_size - 1)
        map_coords_y = (all_pos[:, 1] / world_to_map_scale).long().clamp(0, grid_size - 1)
        flat_indices = map_coords_y * grid_size + map_coords_x

        scatter_dim_size = grid_size * grid_size

        obstacle_type, agent_type, resource_type = 3, 0, 1
        
        is_obstacle = (all_types == obstacle_type)
        is_ally = (all_types == agent_type) & (all_teams == team_id)
        is_enemy = (all_types == agent_type) & (all_teams != team_id) & (all_teams >= 0)
        is_resource = (all_types == resource_type)

        # Use scatter for densities to handle multiple entities in one cell
        if is_obstacle.any():
            obstacle_flat = torch.scatter(
                input=torch.zeros(scatter_dim_size, device=device, dtype=torch.float),
                dim=0,
                index=flat_indices[is_obstacle],
                src=torch.ones_like(flat_indices[is_obstacle], dtype=torch.float)
            )
            occ_map[ch_obstacle] = obstacle_flat.view(grid_size, grid_size)
        if is_ally.any():
            ally_density_flat = torch.scatter(
                input=torch.zeros(scatter_dim_size, device=device, dtype=torch.float),
                dim=0,
                index=flat_indices[is_ally],
                src=torch.ones_like(flat_indices[is_ally], dtype=torch.float)
            )
            occ_map[ch_ally_density] = torch.tanh(ally_density_flat.view(grid_size, grid_size))
        if is_enemy.any():
            enemy_density_flat = torch.scatter(
                input=torch.zeros(scatter_dim_size, device=device, dtype=torch.float),
                dim=0,
                index=flat_indices[is_enemy],
                src=torch.ones_like(flat_indices[is_enemy], dtype=torch.float)
            )
            occ_map[ch_enemy_density] = torch.tanh(enemy_density_flat.view(grid_size, grid_size))
        if is_resource.any():
            resource_density_flat = torch.scatter(
                input=torch.zeros(scatter_dim_size, device=device, dtype=torch.float),
                dim=0,
                index=flat_indices[is_resource],
                src=torch.ones_like(flat_indices[is_resource], dtype=torch.float)
            )
            occ_map[ch_resource_density] = torch.tanh(resource_density_flat.view(grid_size, grid_size))

        occ_map[ch_last_seen_ally] = (occ_map[ch_ally_density] > 0).float()
        occ_map[ch_last_seen_enemy] = (occ_map[ch_enemy_density] > 0).float()
        occ_map[ch_last_seen_resource] = (occ_map[ch_resource_density] > 0).float()
        occ_map[ch_explored] = torch.max(torch.stack([occ_map[ch_last_seen_ally], occ_map[ch_last_seen_enemy], occ_map[ch_last_seen_resource]]), dim=0)[0]
        
        occ_map[ch_cert_obstacle:ch_cert_resource+1] = 1.0

        d_max = (width**2 + height**2)**0.5
        if own_hive_pos is not None:
            map_scale = width / float(grid_size)
            gx_range = torch.arange(grid_size, device=device, dtype=torch.float32)
            gy_range = torch.arange(grid_size, device=device, dtype=torch.float32)
            cell_centers_y, cell_centers_x = torch.meshgrid((gy_range + 0.5) * map_scale, (gx_range + 0.5) * map_scale, indexing='ij')
            rel_vec_x = own_hive_pos[0] - cell_centers_x
            rel_vec_y = own_hive_pos[1] - cell_centers_y
            occ_map[ch_vec_hive_x] = rel_vec_x / (d_max / 2.0)
            occ_map[ch_vec_hive_y] = rel_vec_y / (d_max / 2.0)

        occ_map[ch_step_norm] = float(current_step) / float(max_steps)

        energy_norm_idx = self.NODE_FEATURE_MAP['energy_norm']
        ally_energies = all_feat[is_ally, energy_norm_idx]
        avg_energy = ally_energies.mean() if ally_energies.numel() > 0 else torch.tensor(0.0, device=device, dtype=torch.float32)
        occ_map[ch_avg_team_energy].fill_(avg_energy.item())
        
        return occ_map.unsqueeze(0)

# Create a global instance of the generator
_global_occ_map_generator = CriticObservationGenerator(NODE_FEATURE_MAP)

def generate_occ_map_for_critic(
    team_id: int,
    is_global_critic: bool,
    team_actor_states: List[ActorMapState],
    global_entity_data: Dict,
    current_step: int,
    critic_grid_size: int,
    critic_world_to_map_scale: float,
    env_metadata: Dict,
    device: torch.device
) -> torch.Tensor:
    """
    (V2 Wrapper) Generates the final occupancy map for a critic's observation.
    Now correctly passes constants to the JIT-compiled global map function.
    """
    if not is_global_critic:
        return aggregate_maps_for_critic(
            team_actor_states=team_actor_states,
            team_id=team_id,
            current_step=current_step,
            critic_grid_size=critic_grid_size,
            critic_world_to_map_scale=critic_world_to_map_scale,
            env_metadata=env_metadata,
            device=device
        )
    else:
        hives_info = env_metadata.get('hives_info_for_critic', {})
        own_hive_pos_list = hives_info.get(team_id)
        own_hive_tensor = torch.tensor(own_hive_pos_list, device=device, dtype=torch.float32) if own_hive_pos_list is not None else None

        # Use the global instance of the generator
        return _global_occ_map_generator._generate_global_occ_map_gpu_scatter(
            team_id=team_id,
            all_pos=global_entity_data['positions'],
            all_types=global_entity_data['types'],
            all_teams=global_entity_data['teams'],
            all_feat=global_entity_data['features'],
            current_step=current_step,
            grid_size=critic_grid_size,
            world_to_map_scale=critic_world_to_map_scale,
            width=env_metadata.get('width', 1000.0),
            height=env_metadata.get('height', 1000.0),
            max_steps=env_metadata.get('max_steps', 1000),
            own_hive_pos=own_hive_tensor,
            # --- Pass the required constants ---
            occ_ch_count=OCC_CH_COUNT,
            ch_obstacle=OCC_CH['obstacle'],
            ch_ally_density=OCC_CH['ally_density'],
            ch_enemy_density=OCC_CH['enemy_density'],
            ch_resource_density=OCC_CH['resource_density'],
            ch_last_seen_ally=OCC_CH['last_seen_ally_or_present'],
            ch_last_seen_enemy=OCC_CH['last_seen_enemy_or_present'],
            ch_last_seen_resource=OCC_CH['last_seen_resource_or_present'],
            ch_explored=OCC_CH['explored_time_or_presence'],
            ch_cert_obstacle=OCC_CH['certainty_obstacle'],
            ch_cert_resource=OCC_CH['certainty_resource'],
            ch_vec_hive_x=OCC_CH['vec_hive_x'],
            ch_vec_hive_y=OCC_CH['vec_hive_y'],
            ch_step_norm=OCC_CH['step_norm'],
            ch_avg_team_energy=OCC_CH['avg_team_energy'],
            # ---
            device=device
        )
@torch.no_grad()
def aggregate_maps_for_critic(
    team_actor_states: List[ActorMapState],
    team_id: int,
    current_step: int,
    critic_grid_size: int,
    critic_world_to_map_scale: float,
    env_metadata: Dict,
    device: torch.device
) -> torch.Tensor:
    """
    (V2 - Scatter) Aggregates individual `ActorMapState` maps into a single team-level
    occupancy map for the limited critic, using efficient scatter operations.
    """
    num_maps = len(team_actor_states)
    if num_maps == 0:
        return torch.zeros((1, OCC_CH_COUNT, critic_grid_size, critic_grid_size), device=device)

    # --- 1. Gather all map chunks and metadata from the team's actors ---
    all_chunks_list, all_coords_list, all_last_updated_list, all_certainty_list = [], [], [], []
    for state in team_actor_states:
        chunks, coords, last_updated, certainty = state.get_all_chunks_for_critic()
        all_chunks_list.append(chunks)
        all_coords_list.append(coords)
        all_last_updated_list.append(last_updated)
        all_certainty_list.append(certainty)

    # --- 2. Combine into large tensors ---
    all_chunks = torch.cat(all_chunks_list, dim=0).to(device)
    all_coords = torch.cat(all_coords_list, dim=0).to(device)
    all_last_updated = torch.cat(all_last_updated_list, dim=0).to(device)
    all_certainty = torch.cat(all_certainty_list, dim=0).to(device)

    if all_coords.numel() == 0:
        return torch.zeros((1, OCC_CH_COUNT, critic_grid_size, critic_grid_size), device=device)

    # --- 3. Use scatter operations to aggregate the data on the critic's grid ---
    # This function would contain the scatter_add_, scatter_mean_, etc. calls
    final_map = aggregate_grid_data_scatter(
        all_coords, all_chunks, all_last_updated, all_certainty,
        critic_grid_size, critic_world_to_map_scale, current_step,
        env_metadata, team_id
    )
    return final_map.unsqueeze(0)


def vectorize_reward(reward_dict: Dict) -> torch.Tensor:
    """ Converts a reward dictionary to a fixed-order tensor. """
    return torch.tensor([reward_dict.get(k, 0.0) for k in REWARD_COMPONENT_KEYS], dtype=torch.float32)


@torch.no_grad()
def merge_and_deduplicate_graphs(
    graph_list: List[Data],
    device: torch.device,
    connection_radius: float,
    max_neighbors: int = 32
) -> Batch:
    if not graph_list:
        return Batch.from_data_list([])
    
    combined_graph = Batch.from_data_list(graph_list).to(device)
    
    unique_ids, unique_indices = torch.unique(combined_graph.x[:, NODE_FEATURE_MAP['agent_id']], return_inverse=True)
    
    num_unique_nodes = unique_ids.size(0)
    
    # Use scatter to get the most recent version of each node
    # We can use the 'last_observed_step' if available, otherwise just the first occurrence
    if 'last_observed_step' in MEM_NODE_FEAT_IDX and combined_graph.x.shape[1] > MEM_NODE_FEAT_IDX['last_observed_step']:
        timestamps = combined_graph.x[:, MEM_NODE_FEAT_IDX['last_observed_step']]
        # To get the index of the max timestamp for each unique ID:
        _, max_time_indices = torch.scatter_max(timestamps, unique_indices, dim=0)
    else:
        # If no timestamp, just take the first index found for each unique ID
        max_time_indices = torch.arange(num_unique_nodes, device=device)
        # This part is a bit tricky, we need to find the first occurrence index for each unique ID
        # A simpler way without timestamps is just to use the indices from the unique call if we don't care about which version.
        # Let's find a better way to do this. We can create a sorted list and then pick.
        sorted_indices = torch.argsort(unique_indices)
        _, first_occurrence_indices = torch.unique(unique_indices[sorted_indices], return_inverse=False, sorted=True)
        max_time_indices = sorted_indices[first_occurrence_indices]


    unique_x = combined_graph.x[max_time_indices]
    unique_pos = combined_graph.pos[max_time_indices]
    unique_radii = combined_graph.radii[max_time_indices]

    # Re-compute edges on the unique set of nodes
    edge_index = pyg_nn.radius_graph(unique_pos, r=connection_radius, max_num_neighbors=max_neighbors)
    
    deduplicated_data = Data(
        x=unique_x,
        pos=unique_pos,
        radii=unique_radii,
        edge_index=edge_index
    )
    
    # We need to compute edge features if the model expects them
    if hasattr(combined_graph, 'edge_attr') and combined_graph.edge_attr is not None:
         # Simplified edge feature: relative position and distance
        row, col = edge_index
        rel_pos = unique_pos[row] - unique_pos[col]
        dist = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)
        # Add other features as needed, e.g., difference in features
        # This part needs to match what the GNN expects
        deduplicated_data.edge_attr = torch.cat([rel_pos, dist], dim=-1)

    return Batch.from_data_list([deduplicated_data])

@torch.no_grad()
def generate_foveated_graph(
    all_entity_positions: torch.Tensor,
    all_entity_features: torch.Tensor,
    all_entity_radii: torch.Tensor,
    fovea_agent_positions: torch.Tensor,
    fovea_agent_radii: torch.Tensor,
    live_fovea_graph_data: Optional[Data], # <<< ADDED
    mid_periphery_scale: float,
    mid_cluster_cell_size: float,
    far_cluster_cell_size: float,
    connection_radius: float
) -> Data:
    """
    (V4 - With Live Fovea) Generates a multi-resolution graph centered on a team's fovea.
    """
    if torch.randint(0, 100, (1,)).item() == 0: # Sporadic print
        print(f"\n--- DEBUG: generate_critic_observation ---")
        print(f"  - Is Global: {is_global_critic}")
        print(f"  - Team Memory Map is None: {team_memory_map is None}")
        print(f"  - Team Persistent Graph is None: {team_persistent_graph is None}")
        if team_persistent_graph is not None:
            print(f"    - Nodes: {team_persistent_graph.num_nodes}")
        print(f"  - Team Live Graph is None: {team_live_graph is None}")
        if team_live_graph is not None:
            print(f"    - Nodes: {team_live_graph.num_nodes}")
        print("-" * 20)

    device = all_entity_positions.device
    num_total_entities = all_entity_positions.shape[0]

    # 1. Handle the Live Fovea Data First
    if live_fovea_graph_data is not None and live_fovea_graph_data.num_nodes > 0:
        fovea_nodes_mask = torch.ones(live_fovea_graph_data.num_nodes, dtype=torch.bool, device=device)
        fovea_indices = torch.where(fovea_nodes_mask)[0]
    else:
        # If no live data, create an empty tensor for indices to avoid errors
        fovea_indices = torch.tensor([], dtype=torch.long, device=device)

    # 2. Define Periphery Regions
    if fovea_agent_positions.numel() > 0:
        team_centroid = fovea_agent_positions.mean(dim=0)
        max_fovea_radius = fovea_agent_radii.max() if fovea_agent_radii.numel() > 0 else 0
        
        mid_periphery_radius = max_fovea_radius * mid_periphery_scale
        
        dist_to_centroid = torch.norm(all_entity_positions - team_centroid, dim=1)
        
        mid_periphery_mask = (dist_to_centroid <= mid_periphery_radius)
        far_periphery_mask = (dist_to_centroid > mid_periphery_radius)
    else:
        mid_periphery_mask = torch.zeros(num_total_entities, dtype=torch.bool, device=device)
        far_periphery_mask = torch.ones(num_total_entities, dtype=torch.bool, device=device)

    # Exclude nodes already in the live fovea from periphery consideration
    # This assumes that live_fovea_graph_data contains global IDs or a way to map them
    if live_fovea_graph_data is not None and live_fovea_graph_data.num_nodes > 0:
        live_node_ids = live_fovea_graph_data.x[:, NODE_FEATURE_MAP['agent_id']].long()
        # Create a mask for all entities based on whether their ID is in the live set
        is_in_live_fovea = torch.isin(all_entity_features[:, NODE_FEATURE_MAP['agent_id']].long(), live_node_ids)
        mid_periphery_mask &= ~is_in_live_fovea
        far_periphery_mask &= ~is_in_live_fovea

    mid_indices = torch.where(mid_periphery_mask)[0]
    far_indices = torch.where(far_periphery_mask)[0]
    
    # 3. Process each region
    final_x, final_pos, final_radii = [], [], []

    # Live Fovea (High Detail)
    if fovea_indices.numel() > 0:
        final_x.append(live_fovea_graph_data.x)
        final_pos.append(live_fovea_graph_data.pos)
        final_radii.append(live_fovea_graph_data.radii)
    
    # Mid Periphery (Medium Detail - Individual nodes)
    if mid_indices.numel() > 0:
        final_x.append(all_entity_features[mid_indices])
        final_pos.append(all_entity_positions[mid_indices])
        final_radii.append(all_entity_radii[mid_indices])

    # Far Periphery (Low Detail - Clustering)
    if far_indices.numel() > 0:
        far_pos = all_entity_positions[far_indices]
        far_feat = all_entity_features[far_indices]
        
        cluster_ids = (far_pos // far_cluster_cell_size).long()
        unique_clusters, inverse_indices = torch.unique(cluster_ids, dim=0, return_inverse=True)

        num_clusters = unique_clusters.shape[0]
        
        # Use scatter_mean to aggregate features and positions for each cluster
        # Ensure inverse_indices is correctly shaped for scatter operations
        inverse_indices_expanded = inverse_indices.unsqueeze(1).expand_as(far_feat)
        
        # Initialize tensors for aggregated data
        clustered_feat = torch.zeros(num_clusters, far_feat.shape[1], device=device).scatter_add_(0, inverse_indices_expanded, far_feat)
        clustered_pos = torch.zeros(num_clusters, 2, device=device).scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 2), far_pos)

        # Count nodes per cluster to compute the mean
        cluster_counts = torch.zeros(num_clusters, device=device).scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
        
        # Avoid division by zero
        cluster_counts = cluster_counts.clamp(min=1).unsqueeze(1)
        
        clustered_pos /= cluster_counts
        clustered_feat /= cluster_counts
        
        # For radii, we could take the max or mean. Let's take the mean.
        clustered_radii = torch.zeros(num_clusters, device=device).scatter_add_(0, inverse_indices, all_entity_radii[far_indices])
        clustered_radii /= cluster_counts.squeeze(1)

        # Mark these as clustered nodes
        clustered_feat[:, MEM_NODE_FEAT_IDX['node_status']] = 1

        final_x.append(clustered_feat)
        final_pos.append(clustered_pos)
        final_radii.append(clustered_radii)

    # 4. Combine and build final graph
    if not final_x:
        return Data(x=torch.empty(0, all_entity_features.shape[1], device=device),
                    pos=torch.empty(0, 2, device=device),
                    radii=torch.empty(0, device=device),
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=device))

    final_x_tensor = torch.cat(final_x, dim=0)
    final_pos_tensor = torch.cat(final_pos, dim=0)
    final_radii_tensor = torch.cat(final_radii, dim=0)
    
    # Build edges on the final, combined set of nodes
    edge_index = pyg_nn.radius_graph(final_pos_tensor, r=connection_radius, max_num_neighbors=32)

    return Data(x=final_x_tensor, pos=final_pos_tensor, radii=final_radii_tensor, edge_index=edge_index)


@torch.no_grad()
def generate_cues_from_graph(
    graph_data: Data,
    team_id: int,
    env_metadata: dict,
    current_step: int
) -> torch.Tensor:
    """
    Generates a sophisticated global cue vector (20 cues) using ONLY the provided graph data.
    This function can be used for both Global and Limited critics by feeding them
    a graph built from the true state or the persistent memory, respectively.

    Args:
        graph_data (Data): The PyG Data object (from true state or memory).
                           Must contain x, pos, types, teams.
        team_id (int): The ID of the team for which cues are being generated.
        env_metadata (Dict): Environment metadata for normalization constants.
        current_step (int): The current simulation step.

    Returns:
        torch.Tensor: A (1, 20) tensor of calculated cues.
    """
    device = graph_data.x.device
    cues = torch.zeros(1, GLOBAL_CUE_DIM, device=device)

    num_nodes = graph_data.num_nodes
    if num_nodes == 0:
        return cues # Return zero cues if the graph is empty

    # --- Extract data from graph tensors ---
    features = graph_data.x
    pos = graph_data.pos
    types = features[:, NODE_FEATURE_MAP['node_type_encoded']].long()
    teams = features[:, NODE_FEATURE_MAP['team_id']].long()
    health_norm = features[:, NODE_FEATURE_MAP['health_norm']]
    energy_norm = features[:, NODE_FEATURE_MAP['energy_norm']]
    is_carrying = features[:, NODE_FEATURE_MAP['is_carrying']]
    is_delivered = features[:, NODE_FEATURE_MAP['is_delivered']]
    hive_food_norm = features[:, NODE_FEATURE_MAP['hive_food_norm']]
    
    # For limited critics, we might want to weight stats by certainty
    # For global critics, certainty will be 1.0
    certainty = features[:, MEM_NODE_FEAT_IDX['certainty']] if 'certainty' in MEM_NODE_FEAT_IDX and features.shape[1] > MEM_NODE_FEAT_IDX['certainty'] else torch.ones_like(health_norm)

    # --- Metadata & Normalization ---
    width = env_metadata.get('width', 1000.0)
    height = env_metadata.get('height', 1000.0)
    map_diag = np.sqrt(width**2 + height**2)
    max_steps_env = max(1, env_metadata.get('max_steps', 1000))
    initial_agents_team = max(1, env_metadata.get(f'initial_agents_team_{team_id}', 1))
    initial_total_enemies = sum(env_metadata.get(f'initial_agents_team_{i}', 1) for i in range(env_metadata.get('num_teams', 6)) if i != team_id)

    # --- Create masks for entity types ---
    ally_mask = (types == NODE_TYPE['agent']) & (teams == team_id)
    enemy_mask = (types == NODE_TYPE['agent']) & (teams != team_id) & (teams >= 0)
    resource_mask = (types == NODE_TYPE['resource'])
    ally_hive_mask = (types == NODE_TYPE['hive']) & (teams == team_id)
    enemy_hive_mask = (types == NODE_TYPE['hive']) & (teams != team_id) & (teams >= 0)

    # Weighted sum for stats (certainty * value)
    num_allies_sensed = certainty[ally_mask].sum()
    num_enemies_sensed = certainty[enemy_mask].sum()

    # --- Populate Cues (20 Dims) ---
    # 0, 1: Time
    cues[0, 0] = float(current_step) / max_steps_env
    cues[0, 1] = 1.0 - cues[0, 0]

    # 2, 3, 4, 5: Hive Health/Food
    cues[0, 2] = (health_norm[ally_hive_mask] * certainty[ally_hive_mask]).sum() # Only one own hive, so sum is fine
    cues[0, 3] = (health_norm[enemy_hive_mask] * certainty[enemy_hive_mask]).sum() / (certainty[enemy_hive_mask].sum() + 1e-6)
    cues[0, 4] = (hive_food_norm[ally_hive_mask] * certainty[ally_hive_mask]).sum()
    cues[0, 5] = (hive_food_norm[enemy_hive_mask] * certainty[enemy_hive_mask]).sum() / (certainty[enemy_hive_mask].sum() + 1e-6)

    # 6, 7: Agent Counts (normalized by initial counts)
    cues[0, 6] = num_allies_sensed / initial_agents_team
    cues[0, 7] = num_enemies_sensed / (initial_total_enemies + 1e-6)

    # 8, 9, 10, 11: Own Team Stats (Energy/Health Mean & Var)
    ally_energies = energy_norm[ally_mask]
    ally_healths = health_norm[ally_mask]
    ally_certainties = certainty[ally_mask]
    
    cues[0, 8] = (ally_energies * ally_certainties).sum() / (num_allies_sensed + 1e-6)
    cues[0, 10] = (ally_healths * ally_certainties).sum() / (num_allies_sensed + 1e-6)
    if num_allies_sensed > 1:
        cues[0, 9] = torch.var(ally_energies) # Variance of what's seen, not weighted
        cues[0, 11] = torch.var(ally_healths)

    # 12, 13: Resource Stats
    # Note: For limited critic, this only reflects *known* resources
    num_total_resources = (types == NODE_TYPE['resource']).sum()
    num_delivered_resources = (is_delivered[resource_mask] > 0.5).sum()
    cues[0, 12] = (num_total_resources - num_delivered_resources) / (env_metadata.get('initial_resources', 1) + 1e-6)
    cues[0, 13] = num_delivered_resources / (env_metadata.get('initial_resources', 1) + 1e-6)

    # 14, 15: Carrying Ratios
    cues[0, 14] = (is_carrying[ally_mask] * ally_certainties).sum() / (num_allies_sensed + 1e-6)
    cues[0, 15] = (is_carrying[enemy_mask] * certainty[enemy_mask]).sum() / (num_enemies_sensed + 1e-6)

    # 16: Relative Strength (Own vs Enemies)
    own_strength_metric = num_allies_sensed * cues[0, 8] * cues[0, 10]
    enemy_avg_energy = (energy_norm[enemy_mask] * certainty[enemy_mask]).sum() / (num_enemies_sensed + 1e-6)
    enemy_avg_health = (health_norm[enemy_mask] * certainty[enemy_mask]).sum() / (num_enemies_sensed + 1e-6)
    enemy_strength_metric = num_enemies_sensed * enemy_avg_energy * enemy_avg_health
    cues[0, 16] = torch.tanh((own_strength_metric - enemy_strength_metric) / 10.0) # Normalize difference

    # 17: Team Cohesion
    ally_pos = pos[ally_mask]
    if ally_pos.shape[0] > 1:
        ally_centroid = ally_pos.mean(dim=0)
        cohesion = torch.cdist(ally_pos, ally_centroid.unsqueeze(0)).mean()
        cues[0, 17] = cohesion / (map_diag / 2.0 + 1e-6) # Normalize by half-diagonal
    
    # 18, 19: Proximity to objectives
    res_pos = pos[resource_mask]
    enemy_pos = pos[enemy_mask]
    
    if ally_pos.shape[0] > 0 and res_pos.shape[0] > 0:
        dist_to_res = torch.cdist(ally_pos, res_pos).min(dim=1)[0].mean()
        cues[0, 18] = torch.exp(-dist_to_res / (map_diag * 0.1)) # Exponential decay normalized
    
    if ally_pos.shape[0] > 0 and enemy_pos.shape[0] > 0:
        dist_to_enemy = torch.cdist(ally_pos, enemy_pos).min(dim=1)[0].mean()
        cues[0, 19] = torch.exp(-dist_to_enemy / (map_diag * 0.1))

    return torch.nan_to_num(cues, nan=0.0, posinf=0.0, neginf=0.0)

def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    return pyg_nn.radius(x, y, r, batch_x, batch_y, max_num_neighbors)

def generate_critic_observation_from_tensors(
    team_id: int,
    is_global_critic: bool,
    # Directly receive pre-computed tensors and aggregated views
    all_pos: torch.Tensor,
    all_feat: torch.Tensor,
    all_types: torch.Tensor,
    all_teams: torch.Tensor,
    all_radii: torch.Tensor,
    current_step: int,
    team_memory_map: Optional[torch.Tensor],
    team_persistent_graph: Optional[Data],
    team_raw_map: Optional[torch.Tensor],
    team_live_graph: Optional[Data],
    env_metadata: Dict,
    device: torch.device
) -> Dict:
    """
    (V6 - Optimized) Generates critic observations from pre-computed tensors.
    Standalone version of the method from CriticObservationManager.
    """
    if torch.randint(0, 100, (1,)).item() == 0: # Sporadic print
        print(f"\n--- DEBUG: generate_critic_observation ---")
        print(f"  - Is Global: {is_global_critic}")
        print(f"  - Team Memory Map is None: {team_memory_map is None}")
        print(f"  - Team Persistent Graph is None: {team_persistent_graph is None}")
        if team_persistent_graph is not None:
            print(f"    - Nodes: {team_persistent_graph.num_nodes}")
        print(f"  - Team Live Graph is None: {team_live_graph is None}")
        if team_live_graph is not None:
            print(f"    - Nodes: {team_live_graph.num_nodes}")
        print("-" * 20)

    # --- 1. Define the fovea region based on the team's agents ---
    agent_mask_all = (all_types == NODE_TYPE['agent'])
    team_agent_mask = agent_mask_all & (all_teams == team_id)
    
    fovea_agent_positions = all_pos[team_agent_mask]
    max_obs_radius = env_metadata.get('max_obs_radius_possible', OBS_RADIUS * 1.2)
    fovea_agent_radii = all_feat[team_agent_mask, NODE_FEATURE_MAP['obs_radius_norm']] * max_obs_radius

    # --- 2. Generate the specific inputs for each critic type ---
    if is_global_critic:
        # GLOBAL CRITIC uses ground-truth for everything.
        env_graph = generate_foveated_graph(
            all_entity_positions=all_pos, all_entity_features=all_feat, all_entity_radii=all_radii,
            fovea_agent_positions=fovea_agent_positions, fovea_agent_radii=fovea_agent_radii,
            live_fovea_graph_data=team_live_graph, # Use the ground-truth live fovea
            mid_periphery_scale=2.5, mid_cluster_cell_size=CLUSTER_CELL_SIZE / 3.0,
            far_cluster_cell_size=CLUSTER_CELL_SIZE, connection_radius=env_metadata['obs_radius'] * 3.0
        )
        
        all_data_for_occ = {'positions': all_pos, 'features': all_feat, 'types': all_types, 'teams': all_teams}
        occ_map = generate_occ_map_for_critic(team_id, True, [], all_data_for_occ, current_step, CRITIC_GRID_SIZE, 5.0, env_metadata, device)
        raw_map = team_raw_map.unsqueeze(0) if team_raw_map is not None else torch.zeros(1, RAW_CH_COUNT, CRITIC_GRID_SIZE, CRITIC_GRID_SIZE, device=device)
        
        full_graph_for_cues = Data(x=all_feat, pos=all_pos, types=all_types, teams=all_teams, radii=all_radii)
        cues = generate_cues_from_graph(full_graph_for_cues, team_id, env_metadata, current_step)

    else:
        # LIMITED CRITIC uses its aggregated memory.
        if team_persistent_graph is None or team_memory_map is None or team_raw_map is None:
            team_persistent_graph = Data(x=torch.empty(0, MEM_NODE_FEATURE_DIM + 1, device=device), pos=torch.empty(0, 2, device=device), radii=torch.empty(0, device=device))
            team_memory_map = torch.zeros(OCC_CH_COUNT, CRITIC_GRID_SIZE, CRITIC_GRID_SIZE, device=device)
            team_raw_map = torch.zeros(RAW_CH_COUNT, CRITIC_GRID_SIZE, CRITIC_GRID_SIZE, device=device)
        
        env_graph = generate_foveated_graph(
            all_entity_positions=team_persistent_graph.pos, all_entity_features=team_persistent_graph.x, all_entity_radii=team_persistent_graph.radii,
            fovea_agent_positions=fovea_agent_positions, fovea_agent_radii=fovea_agent_radii,
            live_fovea_graph_data=team_live_graph,
            mid_periphery_scale=2.5, mid_cluster_cell_size=CLUSTER_CELL_SIZE / 3.0,
            far_cluster_cell_size=CLUSTER_CELL_SIZE, connection_radius=env_metadata['obs_radius'] * 3.0
        )

        occ_map = team_memory_map.unsqueeze(0)
        raw_map = team_raw_map.unsqueeze(0)
        cues = generate_cues_from_graph(team_persistent_graph, team_id, env_metadata, current_step)

    return {
        'occ_map': occ_map,
        'raw_map': raw_map,
        'cues': cues,
        'env_graph': Batch.from_data_list([env_graph]),
        'obs_radius': env_metadata['obs_radius']
    }


class CriticObservationManager(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def generate_occ_map_for_critic(
        team_id: int,
        is_global_critic: bool,
        team_actor_states: List[ActorMapState],
        global_entity_data: Dict,
        current_step: int,
        critic_grid_size: int,
        critic_world_to_map_scale: float,
        env_metadata: Dict,
        device: torch.device
    ) -> torch.Tensor:
        """
        (V2 Wrapper) Generates the final occupancy map for a critic's observation.
        Now correctly passes constants to the JIT-compiled global map function.
        """
        if not is_global_critic:
            return aggregate_maps_for_critic(
                team_actor_states=team_actor_states,
                team_id=team_id,
                current_step=current_step,
                critic_grid_size=critic_grid_size,
                critic_world_to_map_scale=critic_world_to_map_scale,
                env_metadata=env_metadata,
                device=device
            )
        else:
            hives_info = env_metadata.get('hives_info_for_critic', {})
            own_hive_pos_list = hives_info.get(team_id)
            own_hive_tensor = torch.tensor(own_hive_pos_list, device=device, dtype=torch.float32) if own_hive_pos_list is not None else None

            # Use the global instance of the generator
            return _global_occ_map_generator._generate_global_occ_map_gpu_scatter(
                team_id=team_id,
                all_pos=global_entity_data['positions'],
                all_types=global_entity_data['types'],
                all_teams=global_entity_data['teams'],
                all_feat=global_entity_data['features'],
                current_step=current_step,
                grid_size=critic_grid_size,
                world_to_map_scale=critic_world_to_map_scale,
                width=env_metadata.get('width', 1000.0),
                height=env_metadata.get('height', 1000.0),
                max_steps=env_metadata.get('max_steps', 1000),
                own_hive_pos=own_hive_tensor,
                # --- Pass the required constants ---
                occ_ch_count=OCC_CH_COUNT,
                ch_obstacle=OCC_CH['obstacle'],
                ch_ally_density=OCC_CH['ally_density'],
                ch_enemy_density=OCC_CH['enemy_density'],
                ch_resource_density=OCC_CH['resource_density'],
                ch_last_seen_ally=OCC_CH['last_seen_ally_or_present'],
                ch_last_seen_enemy=OCC_CH['last_seen_enemy_or_present'],
                ch_last_seen_resource=OCC_CH['last_seen_resource_or_present'],
                ch_explored=OCC_CH['explored_time_or_presence'],
                ch_cert_obstacle=OCC_CH['certainty_obstacle'],
                ch_cert_resource=OCC_CH['certainty_resource'],
                ch_vec_hive_x=OCC_CH['vec_hive_x'],
                ch_vec_hive_y=OCC_CH['vec_hive_y'],
                ch_step_norm=OCC_CH['step_norm'],
                ch_avg_team_energy=OCC_CH['avg_team_energy'],
                # ---
                device=device
            )
    