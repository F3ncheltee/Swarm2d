import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional, List, Union
import time
import torch.nn as nn
from collections import defaultdict
import torch_geometric.data
from torch_geometric.utils import scatter
import torch_geometric.nn as pyg_nn
import torch
import numpy as np
import cProfile
import pstats
from contextlib import contextmanager
from constants import (
    NODE_FEATURE_DIM, NODE_TYPE, NUM_NODE_TYPES, RAW_CH_IDX_TO_NAME, NODE_FEATURE_MAP,
    GLOBAL_CUE_DIM, RAW_CH_COUNT, OCC_CH_COUNT, OCC_CH, CLUSTER_CELL_SIZE,
    AGENT_BASE_STRENGTH, AGENT_RADIUS, MEM_NODE_FEATURE_DIM, MEM_NODE_FEAT_IDX, RAW_CH, AGENT_SLOWED_FACTOR,
    ADAPTIVE_CLUSTERING_MAX_NEIGHBORS, ADAPTIVE_CLUSTERING_CELL_SIZE_FACTOR,
    COLLISION_GROUP_OBSTACLE, COLLISION_GROUP_RESOURCE
)
from env.occlusion import check_los_batched_gpu_sampling, check_los_batched_pybullet, get_entities_in_radius_batched_pybullet


@contextmanager
def profile_section(name: str):
    """Context manager for profiling specific sections of code."""
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"  {name}: {end_time - start_time:.4f}s")


class ObservationProfiler:
    """A simple profiler to measure performance of different observation generation stages."""
    def __init__(self):
        self.timings = {}
        self.step_count = 0
        self.last_printed_step = -1
    
    def start_step(self):
        self.step_count += 1
        self.timings = {}
        self.start_time = time.time()
    
    def record(self, name: str, duration: float):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def print_summary(self):
        """Print detailed profiling information for observation generation."""
        if len(self.timings) == 0:
            return
        
        # Only print every 500 steps to avoid spam
        if self.step_count - self.last_printed_step < 500:
            return
        
        self.last_printed_step = self.step_count
        
        print(f"\n{'='*80}")
        print(f"Observation Generation Profile - Step {self.step_count}")
        print(f"{'='*80}")
        
        # Calculate totals
        total_time = sum(sum(durations) for durations in self.timings.values())
        
        # Sort by total time (descending)
        sorted_timings = sorted(self.timings.items(), 
                               key=lambda x: sum(x[1]), 
                               reverse=True)
        
        print(f"{'Operation':<35} {'Total (s)':>12} {'Avg (s)':>12} {'Calls':>8} {'% Time':>8}")
        print(f"{'-'*80}")
        
        for name, durations in sorted_timings:
            total_dur = sum(durations)
            avg_dur = total_dur / len(durations) if durations else 0
            pct = (total_dur / total_time * 100) if total_time > 0 else 0
            print(f"{name:<35} {total_dur:>12.4f} {avg_dur:>12.4f} {len(durations):>8} {pct:>7.1f}%")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<35} {total_time:>12.4f}")
        print(f"{'='*80}\n")


# Global profiler instance
obs_profiler = ObservationProfiler()


def _tensor_to_str(tensor: torch.Tensor, top_k=3):
    """Helper to create a concise string representation of a tensor for debugging."""
    if not isinstance(tensor, torch.Tensor):
        return str(tensor)
    if tensor.numel() == 0:
        return "[]"
    if tensor.numel() > 2 * top_k:
        flat_tensor = tensor.flatten()
        start = ', '.join(f'{x:.2f}' for x in flat_tensor[:top_k])
        end = ', '.join(f'{x:.2f}' for x in flat_tensor[-top_k:])
        return f"[{start}, ..., {end}] (Shape: {list(tensor.shape)})"
    return f"[{', '.join(f'{x:.2f}' for x in tensor.flatten())}] (Shape: {list(tensor.shape)})"


@torch.jit.script
def cluster_nodes_by_voxel(
    positions: torch.Tensor,
    features: torch.Tensor,
    radii: torch.Tensor,
    cluster_cell_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Coarsens a set of graph nodes by grouping them into voxels (grid cells) using a JIT-compiled function.

    This function takes clouds of nodes (e.g., from an agent's memory) and groups
    nodes that are close to each other into single "super nodes". This is a form
    of graph pooling or downsampling, essential for managing the complexity of
    long-term memory graphs. The properties of the super nodes (position, features)
    are aggregated from the original nodes they contain.

    Args:
        positions (torch.Tensor): A (N, 2) tensor of original node positions.
        features (torch.Tensor): A (N, F) tensor of original node features.
        radii (torch.Tensor): A (N,) tensor of original node physical radii.
        cluster_cell_size (float): The side length of the square voxels used for clustering.

    Returns:
        A tuple containing:
        - super_node_pos (torch.Tensor): (M, 2) positions of the M new super nodes.
        - super_node_features (torch.Tensor): (M, F+1) features of the super nodes. A "count" feature is added.
        - super_node_radii (torch.Tensor): (M,) radii of the super nodes.
        - map_original_to_super (torch.Tensor): (N,) mapping from original node index to its new super node index.
    """
    device = positions.device
    num_original_nodes = positions.shape[0]

    if num_original_nodes == 0:
        empty_pos = torch.empty((0, 2), device=device, dtype=positions.dtype)
        empty_feat = torch.empty((0, features.shape[1] + 1), device=device, dtype=features.dtype)
        empty_radii = torch.empty((0,), device=device, dtype=radii.dtype)
        empty_map = torch.empty((0,), device=device, dtype=torch.long)
        return empty_pos, empty_feat, empty_radii, empty_map

    # 1. Assign each node to a voxel
    voxel_indices = (positions / cluster_cell_size).long()
    
    # 2. Create a unique key for each voxel using Morton code (64-bit linearization)
    # This avoids hash collisions and is more efficient
    vx = voxel_indices[:, 0].to(torch.int64)
    vy = voxel_indices[:, 1].to(torch.int64)
    voxel_keys = (vx << 32) ^ (vy & 0xffffffff)
    unique_keys, map_original_to_super, counts = torch.unique(voxel_keys, return_inverse=True, return_counts=True)
    num_super_nodes = unique_keys.shape[0]

    # 3. Aggregate properties for each super node using scatter
    # scatter_mean is numerically equivalent to scatter_sum / counts
    super_node_pos = scatter(positions, map_original_to_super, dim=0, dim_size=num_super_nodes, reduce='mean')
    super_node_features_agg = scatter(features, map_original_to_super, dim=0, dim_size=num_super_nodes, reduce='mean')
    # For radii, 'max' often makes more sense to represent the bounding area of the cluster
    super_node_radii = scatter(radii, map_original_to_super, dim=0, dim_size=num_super_nodes, reduce='max')

    # 4. Add the node count as a new feature to the super nodes
    # This gives the network crucial information about the density of the cluster
    count_feature = counts.float().unsqueeze(1)
    super_node_features = torch.cat([super_node_features_agg, count_feature], dim=1)

    return super_node_pos, super_node_features, super_node_radii, map_original_to_super


class ActorMapState:
    """
    Manages the persistent, allocentric (world-fixed) memory map for a single agent.

    This class maintains a full-resolution map of the world for one agent. At each step,
    it "paints" the agent's current egocentric observation onto this persistent map.
    The map has channels representing the certainty of different entities (obstacles,
    resources, etc.) and decays over time, simulating forgetting. This provides the
    agent with a persistent spatial memory of the environment.

    Attributes:
        persistent_map (torch.Tensor): The main high-resolution map storing entity certainties.
        _last_seen_step_map (torch.Tensor): A map tracking the simulation step when each cell was last observed.
    """
    def __init__(self, agent_id: int, team_id: int, world_width: int, world_height: int, grid_size: int, world_to_map_scale: float, chunk_size: int, device: torch.device):
        self.agent_id = agent_id
        self.team_id = team_id
        self.device = device
        self.world_to_map_scale = float(world_to_map_scale)
        
        self.world_width = world_width
        self.world_height = world_height
        self.map_width_cells = int(world_width / self.world_to_map_scale)
        self.map_height_cells = int(world_height / self.world_to_map_scale)

        # --- OPTIMIZATION: Create a persistent, full-resolution map ---
        # The persistent map now stores TIMESTAMPS (float) of the last observation for each channel.
        # A value of 0.0 means "never seen". Obstacles are an exception and store 1.0 for "present".
        self.persistent_map = torch.zeros(
            (OCC_CH_COUNT, self.map_height_cells, self.map_width_cells),
            device=self.device, dtype=torch.float32
        )
        
        # Cache frequently used tensor indices to avoid recreation
        self.dynamic_channels_indices = torch.tensor([
            OCC_CH['last_seen_resource'], OCC_CH['last_seen_coop_resource'],
            OCC_CH['last_seen_hive_ally'], OCC_CH['last_seen_hive_enemy'],
            OCC_CH['last_seen_ally'], OCC_CH['last_seen_enemy'],
            OCC_CH['last_seen_self']
        ], device=self.device, dtype=torch.long)
        
        # Cache channel mappings for update
        channel_map = [
            ('resource_presence', 'last_seen_resource'),
            ('coop_resource_presence', 'last_seen_coop_resource'),
            ('hive_ally_presence', 'last_seen_hive_ally'),
            ('hive_enemy_presence', 'last_seen_hive_enemy'),
            ('ally_presence', 'last_seen_ally'),
            ('enemy_presence', 'last_seen_enemy'),
            ('self_presence', 'last_seen_self'),
        ]
        self.raw_channels = torch.tensor([RAW_CH[raw_key] for raw_key, _ in channel_map], device=self.device, dtype=torch.long)
        self.occ_channels = torch.tensor([OCC_CH[occ_key] for _, occ_key in channel_map], device=self.device, dtype=torch.long)
        
        # V8 OPTIMIZATION: Cache for downsampled map to avoid expensive pooling every step
        self._downsampled_cache = None
        self._cache_step = -999
        self._cache_update_frequency = 1  # Set to 1 for smooth GIF generation (was 5)

    def _world_to_map_coords(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Converts world coordinates to allocentric map cell indices."""
        return (world_pos / self.world_to_map_scale).long()
    
    @staticmethod
    def update_batch(actor_map_states: List['ActorMapState'], 
                     raw_map_snapshots: List[torch.Tensor],
                     agent_world_positions: List[torch.Tensor], 
                     obs_radii: List[float],
                     current_step: int):
        """
        Batched update for multiple agents' persistent maps.
        
        CRITICAL: Must update EVERY step to prevent data loss. The cache in 
        get_global_context_map is for downsampling optimization only, not for 
        skipping data collection. If we skip painting, observations from intermediate 
        steps will be lost from the persistent memory map.
        
        Args:
            actor_map_states: List of ActorMapState instances
            raw_map_snapshots: List of [C, H, W] raw observation tensors
            agent_world_positions: List of [2] position tensors
            obs_radii: List of observation radii (floats)
            current_step: Current simulation step
        """
        if not actor_map_states:
            return
            
        # Update all agents every step - the cache in get_global_context_map
        # only optimizes the expensive downsampling operation, not data collection
        for i in range(len(actor_map_states)):
            actor_map_states[i]._update_internal(
                raw_map_snapshots[i],
                agent_world_positions[i],
                obs_radii[i],
                current_step
            )
    
    def _update_internal(self, raw_map_snapshot: torch.Tensor, agent_world_pos: torch.Tensor, 
                        obs_radius: float, current_step: int):
        """
        Internal optimized update - extracted from update() for batching.
        """
        C, H, W = raw_map_snapshot.shape
        if H != W: 
            raise ValueError("Egocentric raw map must be square.")
        
        # Fast path: skip if no observations
        presence_mask = torch.any(raw_map_snapshot > 0, dim=0)
        if not presence_mask.any():
            return
        
        # World to map coordinate conversion
        agent_map_pos = (agent_world_pos / self.world_to_map_scale).long()
        agent_map_x, agent_map_y = agent_map_pos[0].item(), agent_map_pos[1].item()
        
        # Extract observed cell coordinates (ego-centric)
        observed_ego_coords_y, observed_ego_coords_x = torch.where(presence_mask)
        
        # Convert to world coordinates then to map indices
        local_world_offsets_x = (observed_ego_coords_x.float() - W / 2.0) * (2 * obs_radius / W)
        local_world_offsets_y = (observed_ego_coords_y.float() - H / 2.0) * (2 * obs_radius / H)
        
        observed_world_pos = torch.stack([
            agent_world_pos[0] + local_world_offsets_x,
            agent_world_pos[1] + local_world_offsets_y
        ], dim=1)
        
        map_indices = (observed_world_pos / self.world_to_map_scale).long()
        map_xs, map_ys = map_indices[:, 0], map_indices[:, 1]
        
        # Filter valid coordinates
        valid_mask = (map_xs >= 0) & (map_xs < self.map_width_cells) & \
                     (map_ys >= 0) & (map_ys < self.map_height_cells)
        
        if not valid_mask.any():
            return
        
        map_xs = map_xs[valid_mask]
        map_ys = map_ys[valid_mask]
        observed_ego_coords_y = observed_ego_coords_y[valid_mask]
        observed_ego_coords_x = observed_ego_coords_x[valid_mask]
        
        # Paint timestamps onto persistent map
        timestamp_value = float(current_step + 1)
        
        # Update obstacle channel
        obstacle_mask = raw_map_snapshot[RAW_CH['obstacle_presence'], observed_ego_coords_y, observed_ego_coords_x] > 0
        if obstacle_mask.any():
            self.persistent_map[OCC_CH['obstacle_presence'], map_ys[obstacle_mask], map_xs[obstacle_mask]] = timestamp_value
        
        # Update dynamic entity channels (vectorized)
        presence_data = raw_map_snapshot[self.raw_channels][:, observed_ego_coords_y, observed_ego_coords_x]
        presence_masks = presence_data > 0
        
        for c_idx, occ_channel in enumerate(self.occ_channels):
            mask = presence_masks[c_idx]
            if mask.any():
                self.persistent_map[occ_channel, map_ys[mask], map_xs[mask]] = timestamp_value
        
        # Mark explored cells
        self.persistent_map[OCC_CH['explored'], map_ys, map_xs] = timestamp_value
        
        # Periodic maintenance: explored channel and stale memory clearing
        # Set to run every step for smooth GIF generation (was % 10)
        if True:  # current_step % 10 == 0:
            obs_radius_cells = obs_radius / self.world_to_map_scale
            y_start = max(0, int(agent_map_y - obs_radius_cells))
            y_end = min(self.map_height_cells, int(agent_map_y + obs_radius_cells) + 1)
            x_start = max(0, int(agent_map_x - obs_radius_cells))
            x_end = min(self.map_width_cells, int(agent_map_x + obs_radius_cells) + 1)
            
            if y_end > y_start and x_end > x_start:
                # Circular mask for FOV
                y_indices = torch.arange(y_start, y_end, device=self.device, dtype=torch.float32)
                x_indices = torch.arange(x_start, x_end, device=self.device, dtype=torch.float32)
                dy_sq = (y_indices.unsqueeze(1) - agent_map_y) ** 2
                dx_sq = (x_indices.unsqueeze(0) - agent_map_x) ** 2
                dist_sq = (dy_sq + dx_sq) * (self.world_to_map_scale ** 2)
                in_radius_mask = dist_sq <= (obs_radius ** 2)
                
                # Mark explored in circular FOV
                self.persistent_map[OCC_CH['explored'], y_start:y_end, x_start:x_end] = torch.where(
                    in_radius_mask,
                    torch.full_like(self.persistent_map[OCC_CH['explored'], y_start:y_end, x_start:x_end], timestamp_value),
                    self.persistent_map[OCC_CH['explored'], y_start:y_end, x_start:x_end]
                )
                
                # Clear stale memories
                y_grid, x_grid = torch.where(in_radius_mask)
                if y_grid.numel() > 0:
                    actual_y = y_grid + y_start
                    actual_x = x_grid + x_start
                    age_threshold = 30.0
                    
                    timestamps_in_area = self.persistent_map[self.dynamic_channels_indices[:, None], actual_y[None, :], actual_x[None, :]]
                    age_of_memories = float(current_step) - timestamps_in_area
                    stale_mask = (age_of_memories > age_threshold) & (timestamps_in_area > 0)
                    
                    for c_idx, channel_idx in enumerate(self.dynamic_channels_indices):
                        if stale_mask[c_idx].any():
                            self.persistent_map[channel_idx, actual_y[stale_mask[c_idx]], actual_x[stale_mask[c_idx]]] = 0.0

    def update(self, raw_map_snapshot: torch.Tensor, agent_world_pos: torch.Tensor, obs_radius: float, current_step: int):
        """
        Updates the persistent map by "painting" a new egocentric perception onto it.
        
        Legacy method maintained for compatibility. Delegates to _update_internal().

        Args:
            raw_map_snapshot (torch.Tensor): The agent's current egocentric 2D observation.
            agent_world_pos (torch.Tensor): The agent's current [x, y] position in the world.
            obs_radius (float): The agent's current observation radius.
            current_step (int): The current simulation step, used for tracking when cells were last seen.
        """
        start_time = time.time()
        self._update_internal(raw_map_snapshot, agent_world_pos, obs_radius, current_step)
        obs_profiler.record("actor_map_update_internal", time.time() - start_time)

    @staticmethod
    def get_global_context_map_batched(
        actor_map_states: List['ActorMapState'],
        agent_world_positions: List[torch.Tensor],
        output_size: int,
        current_step: int,
        recency_normalization_period: float,
        env_metadata: Dict,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched version of get_global_context_map.
        Generates global maps for multiple agents in parallel.
        """
        start_time = time.time()
        num_agents = len(actor_map_states)
        if num_agents == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        # 1. Stack persistent maps: (N, C_occ, H_map, W_map)
        persistent_maps = torch.stack([s.persistent_map for s in actor_map_states])
        
        # 2. Age-Based Logic (Vectorized over batch)
        # Create age map initialized to 1.1 (never seen)
        age_maps = torch.full_like(persistent_maps, 1.1)
        
        timestamp_channels = [
            OCC_CH['obstacle_presence'], OCC_CH['last_seen_resource'], OCC_CH['last_seen_coop_resource'],
            OCC_CH['last_seen_hive_ally'], OCC_CH['last_seen_hive_enemy'], 
            OCC_CH['last_seen_ally'], OCC_CH['last_seen_enemy'], OCC_CH['last_seen_self']
        ]
        
        timestamp_data = persistent_maps[:, timestamp_channels]
        time_delta = torch.clamp(float(current_step) - timestamp_data, min=0)
        normalized_age = torch.clamp(time_delta / max(1.0, recency_normalization_period), 0.0, 1.0)
        
        valid_timestamps_mask = timestamp_data > 0
        
        # Scatter ages back
        # We can use direct indexing since we extracted the slice
        age_maps[:, timestamp_channels] = torch.where(valid_timestamps_mask, normalized_age, 1.1)
        
        # Explored channel
        explored_timestamps = persistent_maps[:, OCC_CH['explored']]
        explored_delta = torch.clamp(float(current_step) - explored_timestamps, min=0)
        explored_age = torch.clamp(explored_delta / max(1.0, recency_normalization_period), 0.0, 1.0)
        age_maps[:, OCC_CH['explored']] = torch.where(explored_timestamps > 0, explored_age, 1.1)
        
        # 3. Coverage Map (Vectorized)
        presence_channels_keys = [
            'obstacle_presence', 'last_seen_resource', 'last_seen_coop_resource', 'last_seen_hive_ally',
            'last_seen_hive_enemy', 'last_seen_ally', 'last_seen_enemy', 'last_seen_self'
        ]
        presence_indices = [OCC_CH[key] for key in presence_channels_keys]
        
        # (N, C_pres, H, W) -> (N, H, W) -> (N, 1, H, W)
        presence_map = (age_maps[:, presence_indices] < 1.1).any(dim=1).float().unsqueeze(1)
        
        # Pool: (N, 1, H, W) -> (N, 1, Out, Out) -> (N, Out, Out)
        coverage_maps = F.adaptive_avg_pool2d(
            presence_map, output_size=(output_size, output_size)
        ).squeeze(1)
        
        # 4. Intelligent Downsampling (Vectorized)
        downsampled_maps = torch.zeros((num_agents, OCC_CH_COUNT, output_size, output_size), 
                                     device=device, dtype=torch.float32)
                                     
        min_pool_keys = ['last_seen_resource', 'last_seen_coop_resource', 'last_seen_hive_ally', 
                        'last_seen_hive_enemy', 'last_seen_ally', 'last_seen_enemy', 'last_seen_self', 'explored',
                        'obstacle_presence'] # Moved obstacle_presence here to preserve 'seen' status (low age)
        min_indices = [OCC_CH[key] for key in min_pool_keys]
        
        if min_indices:
            # MIN pooling: min(x) = -max(-x)
            negated_age = -age_maps[:, min_indices]
            negated_max = F.adaptive_max_pool2d(negated_age, output_size=(output_size, output_size))
            downsampled_maps[:, min_indices] = -negated_max
            
        max_pool_keys = [] # Empty now, as obstacles moved to min pooling
        max_indices = [OCC_CH[key] for key in max_pool_keys]
        
        if max_indices:
            max_pooled = F.adaptive_max_pool2d(age_maps[:, max_indices], output_size=(output_size, output_size))
            downsampled_maps[:, max_indices] = max_pooled
            
        # 5. Final Map Assembly
        final_maps = torch.zeros_like(downsampled_maps)
        final_maps[:, :OCC_CH_COUNT] = downsampled_maps
        final_maps[:, OCC_CH['coverage']] = coverage_maps
        
        # Global channels
        max_steps = env_metadata.get('max_steps', 1000)
        final_maps[:, OCC_CH['step_norm']] = float(current_step) / max(1.0, float(max_steps))
        
        # Vectors to hive (This part depends on team_id, can be batched by team)
        # Group agents by team to batch calculate vectors
        agent_teams = [s.team_id for s in actor_map_states]
        unique_teams = set(agent_teams)
        
        for tid in unique_teams:
            # Indices of agents belonging to this team
            indices = [i for i, t in enumerate(agent_teams) if t == tid]
            if not indices: continue
            
            vec_x, vec_y, _ = _calculate_common_channels(output_size, tid, current_step, env_metadata, device)
            
            # Broadcast to all agents of this team
            indices_t = torch.tensor(indices, device=device)
            final_maps[indices_t, OCC_CH['vec_hive_x']] = vec_x
            final_maps[indices_t, OCC_CH['vec_hive_y']] = vec_y
            
        # "You Are Here" Marker (Vectorized)
        # Convert all positions to map coords
        positions_t = torch.stack(agent_world_positions) # (N, 2)
        # Assumes all agents have same map scale/dims (true for homogeneous agents)
        scale = actor_map_states[0].world_to_map_scale
        map_w_cells = actor_map_states[0].map_width_cells
        map_h_cells = actor_map_states[0].map_height_cells
        
        agent_map_coords = (positions_t / scale).long()
        scaled_x = (agent_map_coords[:, 0].float() * (output_size / map_w_cells)).long()
        scaled_y = (agent_map_coords[:, 1].float() * (output_size / map_h_cells)).long()
        
        # Filter valid
        valid_mask = (scaled_x >= 0) & (scaled_x < output_size) & (scaled_y >= 0) & (scaled_y < output_size)
        
        if valid_mask.any():
            # Set marker
            # We use scatter or simple indexing. Since it's one pixel per batch item:
            valid_indices = torch.where(valid_mask)[0]
            valid_y = scaled_y[valid_mask]
            valid_x = scaled_x[valid_mask]
            
            final_maps[valid_indices, OCC_CH['you_are_here'], valid_y, valid_x] = 1.0
            
        obs_profiler.record("memory_map_batch_gen", time.time() - start_time)
        return final_maps, coverage_maps

    def get_global_context_map(self, agent_world_pos: torch.Tensor, output_size: int, current_step: int, recency_normalization_period: float, env_metadata: Dict) -> torch.Tensor:
        """
        Generates a fixed-size, downsampled global map for the agent.

        This method takes the high-resolution `persistent_map` of timestamps, converts
        it into a map of normalized age values, downsamples it, and adds global context.
        """
        start_time = time.time()
        
        # V7 OPTIMIZATION: Use cached downsampled map if available and recent
        # DISABLED FOR GIF GENERATION - Always regenerate for smooth visualization
        use_cache = False  # (self._downsampled_cache is not None and 
                          #  (current_step - self._cache_step) < self._cache_update_frequency)
        
        if not use_cache:
            # --- New Age-Based Logic ---
            # Create a temporary map to hold calculated ages.
            age_map = torch.full_like(self.persistent_map, 1.1)
        
            # Define which channels from the persistent map should be converted to age.
            timestamp_channels = [
                OCC_CH['obstacle_presence'], OCC_CH['last_seen_resource'], OCC_CH['last_seen_coop_resource'], OCC_CH['last_seen_hive_ally'],
                OCC_CH['last_seen_hive_enemy'], OCC_CH['last_seen_ally'], OCC_CH['last_seen_enemy'], OCC_CH['last_seen_self']
            ]

            # Get the timestamp data for the channels we need to process
            timestamp_data = self.persistent_map[timestamp_channels]
            
            # Calculate the age of each memory (time_delta)
            time_delta = torch.clamp(float(current_step) - timestamp_data, min=0)
            
            # Normalize the age: 0.0 = seen now, 1.0 = seen long ago or never.
            # We normalize by the fixed period to provide a consistent range to the policy.
            normalized_age = torch.clamp(time_delta / max(1.0, recency_normalization_period), 0.0, 1.0)
            
            # Place the calculated ages back into the correct channels, only where a timestamp existed.
            valid_timestamps_mask = timestamp_data > 0
            age_map[timestamp_channels] = torch.where(valid_timestamps_mask, normalized_age, 1.1) # 1.1 = never seen

            # Obstacles do not have an age. Their channel is a binary presence flag.
            # obstacle_presence_data = self.persistent_map[OCC_CH['obstacle_presence']]
            # obstacle_seen_mask = obstacle_presence_data > 0
            # age_map[OCC_CH['obstacle_presence']] = torch.where(obstacle_seen_mask, 1.0, 1.1)
            
            # NEW: Calculate age for 'explored' channel as well to show a path
            explored_timestamps = self.persistent_map[OCC_CH['explored']]
            explored_time_delta = torch.clamp(float(current_step) - explored_timestamps, min=0)
            explored_normalized_age = torch.clamp(explored_time_delta / max(1.0, recency_normalization_period), 0.0, 1.0)
            valid_explored_mask = explored_timestamps > 0
            # V2: Use 1.1 to differentiate "never seen" from "seen long ago" (which clamps to 1.0)
            # This allows the visualization to create a persistent ground layer.
            age_map[OCC_CH['explored']] = torch.where(valid_explored_mask, explored_normalized_age, 1.1)

            # --- V4 REFINED: Create a Coverage Map for Visualization & Policy ---
            # Define which channels actually represent the presence of an entity in the world.
            # This avoids including global context channels (like vec_to_hive) in the density calculation.
            presence_channels_keys = [
                'obstacle_presence', 'last_seen_resource', 'last_seen_coop_resource', 'last_seen_hive_ally',
                'last_seen_hive_enemy', 'last_seen_ally', 'last_seen_enemy', 'last_seen_self'
                # 'explored' is intentionally omitted here to measure entity density, not just visibility.
            ]
            presence_channel_indices = [OCC_CH[key] for key in presence_channels_keys]

            # A cell has "presence" if any of its presence-related channels have a value
            # indicating they have been seen (i.e., the value is not 1.1).
            presence_map = (age_map[presence_channel_indices] < 1.1).any(dim=0).float()
            coverage_map = F.adaptive_avg_pool2d(
                presence_map.unsqueeze(0).unsqueeze(0),
                output_size=(output_size, output_size)
            ).squeeze(0).squeeze(0)
            
            # --- V2: INTELLIGENT DOWNSAMPLING ---
            # Instead of averaging all channels, which washes out small entities, we use
            # min-pooling for age-based channels and max-pooling for presence-based channels.
            downsampled_map = torch.zeros((OCC_CH_COUNT, output_size, output_size), device=self.device, dtype=torch.float32)

            # Channels to downsample using MIN pooling (for ages, we want the smallest age/most recent)
            min_pool_channels_keys = ['last_seen_resource', 'last_seen_coop_resource', 'last_seen_hive_ally', 'last_seen_hive_enemy', 
                                    'last_seen_ally', 'last_seen_enemy', 'last_seen_self', 'explored',
                                    'obstacle_presence'] # Moved here
            min_pool_indices = [OCC_CH[key] for key in min_pool_channels_keys]
            
            if len(min_pool_indices) > 0:
                # Use MIN pooling: min(x) = -max(-x)
                # This ensures we get the most recent (smallest age) value in each downsampled region
                negated_age_data = -age_map[min_pool_indices].unsqueeze(0)
                negated_max_pooled = F.adaptive_max_pool2d(negated_age_data, output_size=(output_size, output_size))
                min_pooled_data = -negated_max_pooled.squeeze(0)
                downsampled_map[min_pool_indices] = min_pooled_data

            # Channels to downsample using MAX pooling (for binary presence, we want to know if it's there at all)
            max_pool_channels_keys = [] # Empty
            max_pool_indices = [OCC_CH[key] for key in max_pool_channels_keys]

            if len(max_pool_indices) > 0:
                max_pooled_data = F.adaptive_max_pool2d(
                    age_map[max_pool_indices].unsqueeze(0),
                    output_size=(output_size, output_size)
                ).squeeze(0)
                downsampled_map[max_pool_indices] = max_pooled_data
            
            # V7: Cache the downsampled map and coverage map
            self._downsampled_cache = (downsampled_map.clone(), coverage_map.clone())
            self._cache_step = current_step
        else:
            # V7: Use cached downsampled map
            downsampled_map, coverage_map = self._downsampled_cache
        
        #if self.agent_id == 0 and current_step % 100 == 0:
        #    explored_count_low_res = (downsampled_map[OCC_CH['explored']] < 0.99).sum().item()
        #    print(f"[MAP DEBUG] Agent {self.agent_id} at step {current_step}: Explored cells in low-res map (< 0.99): {explored_count_low_res}")

        # Initialize the final map to be sent to the policy
        final_map = torch.zeros((OCC_CH_COUNT, output_size, output_size), device=self.device, dtype=torch.float32)

        # Copy the downsampled data into the final map structure
        final_map[:OCC_CH_COUNT] = downsampled_map

        # --- V4 ADD: Add the coverage map as a new observation channel ---
        # This gives the policy direct information about the spatial density of entities
        # in each downsampled grid cell, which was previously only used for visualization.
        final_map[OCC_CH['coverage']] = coverage_map

        # --- V5: Also add 'last_seen_self' to the list of presence channels ---
        presence_channels_keys = [
            'obstacle_presence', 'last_seen_resource', 'last_seen_coop_resource', 'last_seen_hive_ally',
            'last_seen_hive_enemy', 'last_seen_ally', 'last_seen_enemy', 'last_seen_self'
        ]
        
        # Add final global channels (these overwrite the downsampled placeholders)
        max_steps = env_metadata.get('max_steps', 1000) # Get max_steps for step_norm channel
        # current_step is always <= max_steps by design, so normalization stays in [0, 1]
        final_map[OCC_CH['step_norm']] = float(current_step) / max(1.0, float(max_steps))
        
        # Add vector to hive
        vec_hive_x, vec_hive_y, _ = _calculate_common_channels(
            output_size, self.team_id, current_step, env_metadata, self.device
        )
        final_map[OCC_CH['vec_hive_x']] = vec_hive_x
        final_map[OCC_CH['vec_hive_y']] = vec_hive_y
        
        # Add "You Are Here" marker
        agent_map_pos = self._world_to_map_coords(agent_world_pos)
        scaled_x = int(agent_map_pos[0] * (output_size / self.map_width_cells))
        scaled_y = int(agent_map_pos[1] * (output_size / self.map_height_cells))
        
        if 0 <= scaled_y < output_size and 0 <= scaled_x < output_size:
            marker_size = 1 # Draw a 1x1 pixel marker
            y_start, y_end = max(0, scaled_y), min(output_size, scaled_y + marker_size)
            x_start, x_end = max(0, scaled_x), min(output_size, scaled_x + marker_size)
            final_map[OCC_CH['you_are_here'], y_start:y_end, x_start:x_end] = 1.0

        obs_profiler.record("get_global_context_map", time.time() - start_time)
        return final_map, coverage_map
    
    
   



def _calculate_common_channels(
    grid_size: int,
    team_id: int,
    current_step: int,
    env_metadata: Dict,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper to calculate map channels common to different observation types (e.g., critic and actor maps).

    This function generates channels that provide global context, such as:
    - A normalized representation of the current timestep.
    - A vector field pointing from each cell towards the agent's own hive.

    Args:
        grid_size (int): The size (height and width) of the map grid.
        team_id (int): The team ID of the agent for whom the map is being generated.
        current_step (int): The current simulation step.
        env_metadata (Dict): Environment metadata containing hive positions and max steps.
        device (torch.device): The device for tensor creation.

    Returns:
        A tuple of tensors: (vec_hive_x, vec_hive_y, step_norm_channel).
    """
    width = env_metadata.get('width', 1000.0)
    height = env_metadata.get('height', 1000.0)
    d_max = np.sqrt(width**2 + height**2)
    max_steps = env_metadata.get('max_steps', 1000)
    hives_info = env_metadata.get('hives_info_for_critic', {}) # Expect this to be passed in metadata

    # Vector to Hive Channels
    vec_hive_x = torch.zeros((grid_size, grid_size), device=device)
    vec_hive_y = torch.zeros((grid_size, grid_size), device=device)
    
    own_hive_pos = hives_info.get(team_id)
    if own_hive_pos is not None:
        own_hive_pos_t = torch.tensor(own_hive_pos, device=device, dtype=torch.float32)
        
        # Create grid of cell center world coordinates
        map_scale = width / grid_size
        gx_range = torch.arange(grid_size, device=device, dtype=torch.float32)
        gy_range = torch.arange(grid_size, device=device, dtype=torch.float32)
        cell_centers_y, cell_centers_x = torch.meshgrid(
            (gy_range + 0.5) * map_scale,
            (gx_range + 0.5) * map_scale,
            indexing='ij'
        )
        
        # Calculate vector from each cell to the hive and normalize
        # Using d_max / 2.0 as normalization divisor ensures values stay in reasonable range
        # Maximum distance from any cell to hive center is <= d_max / 2, so normalization stays in [-1, 1]
        rel_vec_x = own_hive_pos_t[0] - cell_centers_x
        rel_vec_y = own_hive_pos_t[1] - cell_centers_y
        vec_hive_x = rel_vec_x / (d_max / 2.0)
        vec_hive_y = rel_vec_y / (d_max / 2.0)

    # Step Normalization Channel
    # current_step is always <= max_steps by design, so normalization stays in [0, 1]
    step_norm_channel = torch.full((grid_size, grid_size), float(current_step) / max(1.0, float(max_steps)), device=device)

    return vec_hive_x, vec_hive_y, step_norm_channel


class RawMapObservationManager (nn.Module):
    """
    Manages the efficient, batched generation of egocentric 2D raw map observations.

    This class contains the logic for creating the local, agent-centric 2D maps that
    represent an agent's immediate perception. The core logic is implemented in a
    JIT-compiled PyTorch function for maximum performance, allowing it to process
    the observations for all agents in a single, parallelized batch operation.
    """
    def generate_maps_wrapper(
        self,
        agent_indices_flat: torch.Tensor, visible_entity_pos: torch.Tensor,
        visible_entity_feat: torch.Tensor, visible_entity_types: torch.Tensor,
        visible_entity_teams: torch.Tensor, visible_entity_coop: torch.Tensor,
        visible_entity_radii: torch.Tensor, # ADDED: Radii of visible entities
        observer_pos_batch: torch.Tensor, observer_radii_batch: torch.Tensor,
        observer_teams_batch: torch.Tensor, observer_feat_batch: torch.Tensor,
        batch_size: int, grid_size: int, world_to_map_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Python wrapper for the JIT-compiled map generation function.

        This method acts as an interface to the core JIT function. It handles the
        lookup of constant values (like channel indices and node types) from Python
        dictionaries and passes them as arguments to the compiled function, which
        cannot perform dictionary lookups itself.

        Args:
            agent_indices_flat (torch.Tensor): A tensor indicating which agent each visible entity belongs to.
            visible_entity_*: Tensors containing the properties of all visible entities.
            observer_*_batch: Tensors containing the properties of the observing agents.
            batch_size (int): The number of agents.
            grid_size (int): The desired size of the output maps.

        Returns:
            torch.Tensor: A batch of generated raw maps with shape (batch_size, C, H, W).
        """
        return self._generate_batched_raw_maps_jit(
            agent_indices_flat, visible_entity_pos, visible_entity_feat,
            visible_entity_types, visible_entity_teams,
            visible_entity_radii, # ADDED
            observer_pos_batch, observer_radii_batch, observer_teams_batch, observer_feat_batch,
            batch_size, grid_size, RAW_CH_COUNT, world_to_map_scale,
            NODE_TYPE['agent'], NODE_TYPE['resource'], NODE_TYPE['hive'], NODE_TYPE['obstacle'],
            RAW_CH['ally_presence'], RAW_CH['enemy_presence'], RAW_CH['resource_presence'],
            RAW_CH['coop_resource_presence'],
            RAW_CH['hive_ally_presence'], RAW_CH['hive_enemy_presence'], RAW_CH['obstacle_presence'],
            RAW_CH['self_presence'],
            RAW_CH.get('trace_presence', -1), # Pass trace channel (or -1 if missing)
            NODE_FEATURE_MAP['agent_id'],
            NODE_FEATURE_MAP['is_cooperative']
        )

    @staticmethod
    @torch.jit.script
    def _generate_batched_raw_maps_jit(
        agent_indices_flat: torch.Tensor, visible_entity_pos: torch.Tensor,
        visible_entity_feat: torch.Tensor,
        visible_entity_types: torch.Tensor,
        visible_entity_teams: torch.Tensor,
        visible_entity_radii: torch.Tensor, # ADDED
        observer_pos_batch: torch.Tensor, observer_radii_batch: torch.Tensor,
        observer_teams_batch: torch.Tensor,
        observer_feat_batch: torch.Tensor, # ADDED: Observer features to get agent_id
        batch_size: int, grid_size: int, map_channels: int, world_to_map_scale: torch.Tensor,
        agent_type: int, resource_type: int, hive_type: int, obstacle_type: int,
        ally_presence_ch: int, enemy_presence_ch: int, res_presence_ch: int,
        coop_res_presence_ch: int,
        hive_ally_p_ch: int, hive_enemy_p_ch: int, obstacle_presence_ch: int,
        self_presence_ch: int,
        trace_presence_ch: int, # NEW
        feat_id_idx: int, # ADDED: Index for agent_id in feature tensor
        feat_coop_idx: int
    ) -> torch.Tensor:
        """
        JIT-compiled core logic for generating batched 2D raw observation maps.

        This highly optimized function is the heart of the 2D map generation. It takes
        batched data of all visible entities for all agents and projects them onto
        their respective local egocentric grids. It uses efficient tensor operations
        and `torch.scatter` to "paint" the presence of entities onto the map channels
        in a fully parallelized manner.

        Args:
            (Various Tensors and ints): All inputs are provided as tensors or primitive types
                                        to ensure compatibility with TorchScript JIT compilation.

        Returns:
            torch.Tensor: The final batch of raw observation maps.
        """
        device = observer_pos_batch.device
        H, W, C = grid_size, grid_size, map_channels

        if agent_indices_flat.numel() == 0:
            return torch.zeros((batch_size, C, H, W), device=device, dtype=torch.float32)

        observer_pos_expanded = observer_pos_batch[agent_indices_flat]
        observer_radii_expanded = observer_radii_batch[agent_indices_flat]
        observer_teams_expanded = observer_teams_batch[agent_indices_flat]
        
        relative_pos = visible_entity_pos - observer_pos_expanded
        
        # --- V8: DYNAMIC SCALING REVERT ---
        # The world_to_map_scale is now calculated based on the individual agent's obs_radius,
        # making this a truly dynamic scaling operation.
        cell_dim_expanded = world_to_map_scale[agent_indices_flat]
        
        # --- V8: GAUSSIAN SPLATTING ---
        # Instead of drawing hard circles, we render a smooth 2D Gaussian for each entity.
        # This preserves sub-pixel information and encodes size in the spread of the Gaussian.
        radii_in_cells = (visible_entity_radii / cell_dim_expanded)

        # Define a kernel of offsets around the center point.
        # The kernel size determines how far out the Gaussian will be calculated.
        # A kernel radius of 3*sigma captures >99% of the Gaussian's volume.
        # We'll set sigma proportional to the entity's radius.
        max_entity_radius_in_cells = 8 # Cap to prevent huge kernels for hives
        kernel_radius = max_entity_radius_in_cells * 2 # Heuristic for a safe kernel size
        offset_range = torch.arange(-kernel_radius, kernel_radius + 1, device=device)
        kernel_y, kernel_x = torch.meshgrid(offset_range, offset_range, indexing='ij')
        
        # --- Calculate grid positions and filter points outside the map ---
        local_pos_centered_x = relative_pos[:, 0] + observer_radii_expanded
        local_pos_centered_y = relative_pos[:, 1] + observer_radii_expanded
        center_grid_x = (local_pos_centered_x / cell_dim_expanded)
        center_grid_y = (local_pos_centered_y / cell_dim_expanded)

        # Broadcast to get all potential grid points. Shapes: (N, 1) + (1, K) -> (N, K*K)
        all_grid_x = center_grid_x.unsqueeze(1) + kernel_x.flatten().unsqueeze(0)
        all_grid_y = center_grid_y.unsqueeze(1) + kernel_y.flatten().unsqueeze(0)
        
        valid_grid_mask = (all_grid_x >= 0) & (all_grid_x < W) & (all_grid_y >= 0) & (all_grid_y < H)

        # Get indices of valid points from the 2D mask
        valid_entity_indices, valid_kernel_indices = torch.where(valid_grid_mask)

        # Use the mask to select the final grid coordinates
        grid_x = all_grid_x[valid_grid_mask]
        grid_y = all_grid_y[valid_grid_mask]

        # --- Calculate Gaussian intensity at each valid grid point ---
        # Get the relative positions of the grid points from their entity centers
        kernel_x_flat_valid = kernel_x.flatten()[valid_kernel_indices]
        kernel_y_flat_valid = kernel_y.flatten()[valid_kernel_indices]
        
        # Set sigma to be proportional to the entity's radius in cells.
        # A scaling factor of 0.5 makes the Gaussian's "effective" radius (at ~60% intensity) match the visual radius.
        sigma = radii_in_cells[valid_entity_indices] * 0.5
        # Prevent sigma from being zero to avoid division errors and ensure even tiny objects are visible.
        sigma = torch.clamp(sigma, min=0.3)
        
        # 2D Gaussian formula: I = exp(-((dx^2 + dy^2) / (2 * sigma^2)))
        dist_sq = kernel_x_flat_valid**2 + kernel_y_flat_valid**2
        gaussian_intensity = torch.exp(-dist_sq / (2 * sigma**2))

        # --- V15: REVERTED - Use Gaussian Splatting for Obstacles ---
        # The user correctly observed that hard-edged boxes for obstacles "pop in" too suddenly,
        # depriving the agent of a crucial early warning signal at the edge of its vision.
        # A soft-edged Gaussian, even if not physically perfect, provides a much better
        # perceptual experience for the policy, allowing it to react sooner.
        intensity = gaussian_intensity
        
        # Filter out negligible intensities to save computation in the scatter operation
        significant_intensity_mask = intensity > 0.01
        
        grid_x = grid_x[significant_intensity_mask]
        grid_y = grid_y[significant_intensity_mask]
        intensity = intensity[significant_intensity_mask]
        # We need to re-filter the entity indices as well
        valid_entity_indices = valid_entity_indices[significant_intensity_mask]

        # --- Map intensities to correct channels ---
        agent_indices_flat_final = agent_indices_flat[valid_entity_indices]
        types_final = visible_entity_types[valid_entity_indices]
        teams_final = visible_entity_teams[valid_entity_indices]
        feat_final = visible_entity_feat[valid_entity_indices]
        observer_teams_final = observer_teams_expanded[valid_entity_indices]

        # --- V10 FIX: Move mask creation before usage ---
        is_agent_mask = (types_final == agent_type)
        observer_ids = observer_feat_batch[:, feat_id_idx]
        observer_ids_expanded_final = observer_ids[agent_indices_flat_final]
        entity_ids_final = feat_final[:, feat_id_idx]
        
        is_self_mask = is_agent_mask & (entity_ids_final.long() == observer_ids_expanded_final.long())
        is_ally_mask = is_agent_mask & (teams_final == observer_teams_final) & ~is_self_mask
        is_enemy_mask = is_agent_mask & (teams_final != observer_teams_final)

        is_resource_mask = (types_final == resource_type)
        is_coop_resource_mask = is_resource_mask & (feat_final[:, feat_coop_idx] > 0.5)
        is_resource_mask = is_resource_mask & ~is_coop_resource_mask # Exclude coop from regular resources

        is_ally_hive_mask = (types_final == hive_type) & (teams_final == observer_teams_final)
        is_enemy_hive_mask = (types_final == hive_type) & (teams_final != observer_teams_final)
        is_obstacle_mask = (types_final == obstacle_type)

        flat_indices_for_scatter = agent_indices_flat_final * (H * W) + grid_y * W + grid_x

        num_points_to_draw = valid_entity_indices.shape[0]
        map_features = torch.zeros((num_points_to_draw, C), device=device, dtype=torch.float32)

        # Robustly paint Gaussian intensities onto the correct channels
        # This is more explicit and JIT-friendly than the previous scatter_add approach.
        # We need to map our flat indices (which are for the N points) into the B*C*H*W flattened space
        
        # Calculate indices into the 'map_features' tensor (N, C)
        # We need to scatter 'intensity' into map_features at the correct column indices.
        # But wait, map_features is already (N, C). We can just direct assign.
        
        map_features[is_self_mask, self_presence_ch] = intensity[is_self_mask]
        map_features[is_ally_mask, ally_presence_ch] = intensity[is_ally_mask]
        map_features[is_enemy_mask, enemy_presence_ch] = intensity[is_enemy_mask]
        map_features[is_resource_mask, res_presence_ch] = intensity[is_resource_mask]
        map_features[is_coop_resource_mask, coop_res_presence_ch] = intensity[is_coop_resource_mask]
        map_features[is_ally_hive_mask, hive_ally_p_ch] = intensity[is_ally_hive_mask]
        map_features[is_enemy_hive_mask, hive_enemy_p_ch] = intensity[is_enemy_hive_mask]
        map_features[is_obstacle_mask, obstacle_presence_ch] = intensity[is_obstacle_mask]
        
        # --- NEW: TRACE RENDERING ---
        if trace_presence_ch >= 0:
            map_features[is_self_mask, trace_presence_ch] = intensity[is_self_mask]

        # Initialize the output tensor (reuse if possible)
        flat_map_grid = torch.zeros((batch_size * H * W, C), device=device, dtype=torch.float32)

        # Add floor and clamp for robust float-to-long index conversion
        safe_grid_x = torch.clamp(grid_x.floor(), 0, W - 1)
        safe_grid_y = torch.clamp(grid_y.floor(), 0, H - 1)
        
        flat_indices_for_scatter = agent_indices_flat_final * (H * W) + safe_grid_y * W + safe_grid_x
        
        # Use scatter with 'max' to handle overlapping Gaussians: the brightest value wins.
        flat_map_grid = scatter(
            src=map_features,
            index=flat_indices_for_scatter.long(),
            dim=0,
            dim_size=flat_map_grid.shape[0],
            reduce='max'
        )
        
        # Reshape the flat grid back into the final batched map tensor
        batched_raw_maps = flat_map_grid.view(batch_size, H, W, C).permute(0, 3, 1, 2)
        return batched_raw_maps









def create_graph_edge_features(graph_data: Data, edge_mlp: nn.Module, obs_radius: float) -> torch.Tensor:
    """
    Creates rich edge features for a graph network, processing them with a provided MLP.

    This is a utility function used in graph-based policies. It takes a graph's basic
    structure and computes detailed features for each edge. These features include
    relative position, distance, relative velocity, and whether the connected nodes
    are on the same team. These raw edge features are then passed through a small
    neural network (`edge_mlp`) to create a learned edge embedding.

    Args:
        graph_data (torch_geometric.data.Data): The input graph data object.
        edge_mlp (torch.nn.Module): A neural network (MLP) to process the raw edge features.
        obs_radius (float): The observation radius, used for normalizing distance features.

    Returns:
        torch.Tensor: A tensor of final edge features/embeddings for the graph.
    """
    if graph_data.edge_index is None or graph_data.num_edges == 0:
        # If there are no edges, return an empty tensor with the correct feature dimension.
        # This requires knowing the output dimension of the edge_mlp.
        output_dim = -1
        try:
            # First, try to access as if it's our custom MLP class
            last_layer = edge_mlp.mlp[-1]
            if isinstance(last_layer, nn.Linear):
                output_dim = last_layer.out_features
        except (AttributeError, IndexError):
            # If that fails, try to access as if it's a standard nn.Sequential
            try:
                last_layer = edge_mlp[-1]
                if isinstance(last_layer, nn.Linear):
                    output_dim = last_layer.out_features
            except (AttributeError, IndexError, TypeError):
                # If both fail, print a warning and fallback.
                print("Warning: Could not determine output dimension of edge_mlp. Falling back to a hardcoded value.")
                output_dim = 16 # Fallback to a common default

        # Ensure output_dim was found, otherwise use the fallback
        if output_dim == -1:
            output_dim = 16

        return torch.empty((0, output_dim), device=graph_data.x.device, dtype=graph_data.x.dtype)

    edge_index, pos, node_features_raw = graph_data.edge_index, graph_data.pos, graph_data.x
    row, col = edge_index

    source_node_feats = node_features_raw[row]
    target_node_feats = node_features_raw[col]

    rel_pos = pos[col] - pos[row]
    distance = torch.linalg.norm(rel_pos, dim=1, keepdim=True)
    
    # Normalize distance by the passed-in obs_radius
    dist_norm = (distance / max(obs_radius, 1e-6)).clamp(-2.0, 2.0)

    # Velocity and Team features
    vel_indices = [NODE_FEATURE_MAP['vel_x_norm'], NODE_FEATURE_MAP['vel_y_norm']]
    vel_i = source_node_feats[:, vel_indices]
    vel_j = target_node_feats[:, vel_indices]
    rel_vel = vel_j - vel_i

    team_id_idx = NODE_FEATURE_MAP['team_id']
    team_i = source_node_feats[:, team_id_idx]
    team_j = target_node_feats[:, team_id_idx]
    same_team = (team_i == team_j).float().unsqueeze(-1)

    # Certainty features (handle both live and memory graphs)
    certainty_i = torch.ones_like(distance)
    certainty_j = torch.ones_like(distance)
    if 'certainty' in MEM_NODE_FEAT_IDX and node_features_raw.shape[1] > MEM_NODE_FEAT_IDX['certainty']:
        certainty_idx = MEM_NODE_FEAT_IDX['certainty']
        certainty_i = source_node_feats[:, certainty_idx].unsqueeze(-1)
        certainty_j = target_node_feats[:, certainty_idx].unsqueeze(-1)

    # Concatenate all raw features for the MLP
    edge_attr_raw = torch.cat([
        rel_pos, dist_norm, rel_vel, same_team, certainty_i, certainty_j,
        source_node_feats, target_node_feats
    ], dim=1)

    return edge_mlp(edge_attr_raw)



