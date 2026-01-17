import torch
from typing import List, Dict, Optional, Tuple
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import scatter

from constants import MEM_NODE_FEAT_IDX, NODE_FEATURE_DIM, MEM_NODE_FEATURE_DIM, NODE_FEATURE_MAP, NODE_TYPE, MAX_STEPS
from env.observations import cluster_nodes_by_voxel

def _find_root(parent: torch.Tensor, i: int) -> int:
    """Helper function for the disjoint set union (DSU) data structure to find the root of an element."""
    root = i
    path_to_root = []
    while parent[root] != root:
        path_to_root.append(root)
        root = parent[root].item()
    
    # Path compression
    for node in path_to_root:
        parent[node] = root
        
    return root

def _tensor_to_str(tensor: torch.Tensor, top_k=3):
    """Helper to create a concise string representation of a tensor."""
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


# This will be the new home for our batched memory manager.

class BatchedPersistentGraphMemory:
    """
    A memory manager that stores and updates graph-based memories for all agents
    in a batched, GPU-optimized manner. It replaces the inefficient agent-by-agent
    loop with vectorized tensor operations.
    """
    def __init__(self,
                 num_agents: int,
                 device: torch.device):
        """
        Initializes the unified tensors that will hold memory data for all agents.

        Args:
            num_agents: The total number of agents in the simulation.
            device: The torch device (e.g., 'cuda') for tensor operations.
        """
        self.num_agents = num_agents
        self.device = device
        self.debug_mode = False
        
        # --- Unified Memory Tensors ---
        # These tensors store data for all nodes across all agents.
        # They are initialized as empty and will grow dynamically.

        # Core features of each memory node.
        # Shape: [N_total_mem_nodes, D_feat]
        self.mem_features: Optional[torch.Tensor] = None

        # Position (x, y) of each memory node.
        # Shape: [N_total_mem_nodes, 2]
        self.mem_pos: Optional[torch.Tensor] = None

        # The key to batching: This tensor maps every memory node to its owner agent.
        # Shape: [N_total_mem_nodes]
        self.mem_agent_idx: Optional[torch.Tensor] = None
        
        # The unique environment ID for each entity represented by a memory node.
        # Used for matching new observations to existing memories.
        # Shape: [N_total_mem_nodes]
        self.mem_env_id: Optional[torch.Tensor] = None

        # This pointer is no longer needed with the optimized get_graph_batch, so it's removed.
        # self.mem_ptr: Optional[torch.Tensor] = None
        
        self.d_feat_in = -1 # Input feature dim from live obs
        self.d_feat_mem = MEM_NODE_FEATURE_DIM # Stored feature dim
        
        # For convenience, store the indices of special features
        self.last_seen_idx = MEM_NODE_FEAT_IDX['last_observed_step']
        self.status_idx = MEM_NODE_FEAT_IDX['node_status']
        self.env_id_idx = NODE_FEATURE_MAP['agent_id']
        self.is_ego_idx = NODE_FEATURE_MAP['is_ego']
        
        # --- Per-Agent Caching for Clustered Periphery ---
        # To replicate the original logic, each agent needs its own cache.
        self.periphery_graph_cache: Dict[int, Optional[Data]] = {i: None for i in range(num_agents)}
        self.last_cluster_step: Dict[int, int] = {i: -1 for i in range(num_agents)}
        self.last_cluster_pos: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(num_agents)}

    def _create_global_keys(self, agent_indices: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Creates unique 64-bit keys by combining agent and environment IDs."""
        # Shift agent_idx to the high 32 bits and env_id to the low 32 bits
        return (agent_indices.long() << 32) | env_ids.long()
    
    @staticmethod
    def _fast_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
        """
        Fast replacement for torch.isin using searchsorted.
        Returns a boolean mask indicating which elements are in test_elements.
        O(N log M) instead of O(N*M).
        """
        if elements.numel() == 0 or test_elements.numel() == 0:
            return torch.zeros(elements.shape[0], dtype=torch.bool, device=elements.device)
        
        # Sort test_elements for binary search
        sorted_test, _ = torch.sort(test_elements)
        
        # Use searchsorted to find positions
        positions = torch.searchsorted(sorted_test, elements)
        
        # Clamp to valid range
        positions = torch.clamp(positions, 0, sorted_test.shape[0] - 1)
        
        # Check if elements at positions match
        return sorted_test[positions] == elements

    def update_memory_nodes(self, indices: torch.Tensor, features: torch.Tensor, pos: torch.Tensor, current_step: int):
        """A dedicated (and currently unused) function to update memory nodes."""
        # This function is a placeholder for potential future use cases where memory
        # nodes might be updated based on model predictions (e.g., fading out nodes
        # that are predicted to be gone). It is not used in the current observation-
        # based memory update flow.
        self.mem_features[indices] = features
        self.mem_pos[indices] = pos
        self.mem_features[indices, self.last_seen_idx] = current_step

    def _sort_and_find_matches(self, mem_keys: torch.Tensor, obs_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Efficiently finds matching, new, and unmatched nodes using SORTED MERGE.
        This is O(N log N + M log M) instead of O(N*M) for torch.isin.

        Returns:
            mem_update_indices: Indices in memory for nodes that have a match in observations.
            obs_update_indices: Corresponding indices in observations for the matched nodes.
            obs_add_indices: Indices in observations for nodes that are new and need to be added.
        """
        
        # Sort both key arrays
        sorted_mem_keys, sorted_mem_indices = torch.sort(mem_keys)
        sorted_obs_keys, sorted_obs_indices = torch.sort(obs_keys)
        
        # Use searchsorted to find where each obs key would be inserted in sorted mem keys
        positions = torch.searchsorted(sorted_mem_keys, sorted_obs_keys)
        
        # Clamp positions to valid range
        positions = torch.clamp(positions, 0, sorted_mem_keys.shape[0] - 1)
        
        # Check if the keys at those positions actually match
        matches = (sorted_mem_keys[positions] == sorted_obs_keys)
        
        # Get the original indices for matches
        obs_update_indices = sorted_obs_indices[matches]
        mem_update_indices = sorted_mem_indices[positions[matches]]
        
        # New observations are those that don't match
        obs_add_indices = sorted_obs_indices[~matches]

        return mem_update_indices, obs_update_indices, obs_add_indices

    def update_batch(self, obs_batch: Batch, current_step: int):
        """
        The core method. Updates the memory for all agents in a single batched call.
        """
        # --- EGO DEBUG ---
        if self.debug_mode and current_step < 5:
            agent_0_batch_mask = (obs_batch.batch == 0)
            if torch.any(agent_0_batch_mask):
                agent_0_features = obs_batch.x[agent_0_batch_mask]
                agent_0_ego_sum = torch.sum(agent_0_features[:, self.is_ego_idx]).item()
                print(f"[TRACEPOINT 2 @ Step {current_step}, MemManager.update] Obs received. Ego nodes in 'x': {agent_0_ego_sum}.")
        # --- END EGO DEBUG ---

        # --- Pre-computation: Get env_id from features if not present ---
        # This makes the function more robust.
        if not hasattr(obs_batch, 'env_id'):
            obs_batch.env_id = obs_batch.x[:, self.env_id_idx].long()
        else:
            obs_batch.env_id = obs_batch.env_id.long()

        # --- Step 0: Handle Initial Case ---
        if self.mem_features is None:
            self.d_feat_in = obs_batch.x.shape[1]
            if self.d_feat_in != NODE_FEATURE_DIM:
                 raise ValueError(f"Initial observation feature dim {self.d_feat_in} does not match NODE_FEATURE_DIM {NODE_FEATURE_DIM}")
            
            num_new_nodes = obs_batch.x.shape[0]
            
            # Directly initialize memory from the first observation batch
            self.mem_features = torch.zeros((num_new_nodes, self.d_feat_mem), device=self.device)
            self.mem_features[:, :self.d_feat_in] = obs_batch.x
            self.mem_features[:, self.last_seen_idx] = current_step
            self.mem_features[:, self.status_idx] = 0.0 # 0.0 for normal status
            
            self.mem_pos = obs_batch.pos
            self.mem_agent_idx = obs_batch.batch
            self.mem_env_id = obs_batch.env_id
            return

        # --- Step 2: Prepare Keys for Matching ---
        mem_keys = self._create_global_keys(self.mem_agent_idx, self.mem_env_id)
        obs_keys = self._create_global_keys(obs_batch.batch, obs_batch.env_id)

        # --- Step 3: Sort Keys and Find Matches ---
        mem_update_indices, obs_update_indices, obs_add_indices = self._sort_and_find_matches(mem_keys, obs_keys)

        # --- Step 4: Perform Batched Updates ---
        if mem_update_indices.numel() > 0:
            # This preserves the original identity (e.g., team_id, node_type) of the memory node
            # while updating its state (position, last_seen_step).

            # 1. Update the position of the memory node to the new observation's position.
            self.mem_pos[mem_update_indices] = obs_batch.pos[obs_update_indices]

            # 2. Update the 'last_seen_step' to the current step.
            self.mem_features[mem_update_indices, self.last_seen_idx] = current_step

            # 3. Explicitly set 'is_ego' to 0.0, as an updated memory node cannot be the current step's 'self' node.
            self.mem_features[mem_update_indices, self.is_ego_idx] = 0.0

        # Add new, unmatched observations to memory
        new_mask = torch.ones(obs_batch.x.shape[0], dtype=torch.bool, device=self.device)
        new_mask[obs_update_indices] = False # Mark matched observations as not new
        new_obs_indices = torch.where(new_mask)[0]

        if new_obs_indices.numel() > 0:
            num_added_nodes = new_obs_indices.shape[0]
            
            # Create feature tensor for new nodes
            new_features = torch.zeros((num_added_nodes, self.d_feat_mem), device=self.device)
            new_features[:, :self.d_feat_in] = obs_batch.x[new_obs_indices]

            new_features[:, self.last_seen_idx] = current_step
            new_features[:, self.status_idx] = 0.0 # 0.0 for normal status

            # Append new data to the unified tensors
            self.mem_features = torch.cat([self.mem_features, new_features], dim=0)
            self.mem_pos = torch.cat([self.mem_pos, obs_batch.pos[new_obs_indices]], dim=0)
            self.mem_agent_idx = torch.cat([self.mem_agent_idx, obs_batch.batch[new_obs_indices]], dim=0)
            self.mem_env_id = torch.cat([self.mem_env_id, obs_batch.env_id[new_obs_indices]], dim=0)
            
            # Also append radii ---
            if hasattr(self, 'mem_radii') and self.mem_radii is not None:
                self.mem_radii = torch.cat([self.mem_radii, obs_batch.radii[new_obs_indices]], dim=0)
            else:
                # If mem_radii doesn't exist yet, this is the first time we're adding radii.
                # We need to create a placeholder for all existing memory nodes.
                num_existing_nodes = self.mem_features.shape[0] - num_added_nodes
                placeholder_radii = torch.zeros(num_existing_nodes, device=self.device)
                self.mem_radii = torch.cat([placeholder_radii, obs_batch.radii[new_obs_indices]], dim=0)


    def get_graph_batch(self,
                        fovea_agent_positions: torch.Tensor,
                        fovea_agent_radii: torch.Tensor,
                        live_fovea_graph_list: List[Data],
                        current_step: int,
                        min_cluster_size: float,
                        max_cluster_size: float,
                        graph_connection_radius_factor: float,
                        cluster_aggressiveness: float,
                        cluster_exclusion_radius_factor: float,
                        detailed_clustering_radius_factor: float,
                        mem_skeleton_connection_factor: float,
                        clustering_frequency: int = 5,
                        max_steps: int = 1000,
                        debug_mode: bool = False) -> Optional[Batch]:
        """
        Eliminates the per-agent loop entirely by processing
        all agents simultaneously in a unified graph structure.
        """
        self.debug_mode = debug_mode
        if self.mem_pos is None or self.mem_pos.shape[0] == 0:
            # If no memory, just return the live graphs
            if not live_fovea_graph_list:
                return None
            return Batch.from_data_list(live_fovea_graph_list)

        num_agents_in_batch = fovea_agent_positions.shape[0]

        # --- Overall Memory State Debug (Print every 10 steps) ---
        if current_step % 10 == 0:
            total_mem_nodes = self.mem_pos.shape[0]
            avg_mem_per_agent = total_mem_nodes / num_agents_in_batch if num_agents_in_batch > 0 else 0
            print(f"[Memory Stats @ Step {current_step}] Total nodes: {total_mem_nodes}, Avg/agent: {avg_mem_per_agent:.1f}, Agents: {num_agents_in_batch}")
        
        if debug_mode and current_step < 5:
            print(f"\n[DEBUG @ Step {current_step}, BatchedGraphMemory V6]")
            print(f"  - Total nodes in memory: {self.mem_pos.shape[0]}")
            if self.mem_agent_idx is not None and self.mem_agent_idx.numel() > 0:
                unique_agents, counts = torch.unique(self.mem_agent_idx, return_counts=True)
                mem_dist_str = ", ".join([f"A{a.item()}:{c.item()}" for a, c in zip(unique_agents[:5], counts[:5])])
                print(f"  - Memory node distribution: {mem_dist_str}" + ("..." if len(unique_agents) > 5 else ""))

        # Get clustered periphery graphs for all agents
        clustered_periphery_graphs = self._get_clustered_periphery_batched(
            agent_positions=fovea_agent_positions,
            agent_fovea_radii=fovea_agent_radii,
            live_fovea_graph_list=live_fovea_graph_list,
            current_step=current_step,
            clustering_frequency=clustering_frequency,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_aggressiveness=cluster_aggressiveness,
            mem_skeleton_connection_factor=mem_skeleton_connection_factor,
            cluster_exclusion_radius_factor=cluster_exclusion_radius_factor,
            detailed_clustering_radius_factor=detailed_clustering_radius_factor
        )

        # =================================================================
        # BATCHED PROCESSING - Process all agents simultaneously
        # =================================================================
        
        # Step 1: Batch all live foveal nodes
        all_live_nodes_list = []
        all_live_pos_list = []
        all_live_radii_list = []
        all_live_batch_list = []
        node_offset = 0
        node_offsets_per_agent = torch.zeros(num_agents_in_batch + 1, dtype=torch.long, device=self.device)
        
        for i in range(num_agents_in_batch):
            foveal_nodes_live = self._get_foveal_nodes(live_fovea_graph_list[i], current_step)
            num_live = foveal_nodes_live['pos'].shape[0]
            
            all_live_nodes_list.append(foveal_nodes_live['x'])
            all_live_pos_list.append(foveal_nodes_live['pos'])
            all_live_radii_list.append(foveal_nodes_live['radii'])
            all_live_batch_list.append(torch.full((num_live,), i, dtype=torch.long, device=self.device))
            
            node_offset += num_live
            node_offsets_per_agent[i + 1] = node_offset
        
        # Concatenate all live nodes
        if all_live_nodes_list:
            all_live_features = torch.cat(all_live_nodes_list, dim=0) if any(x.numel() > 0 for x in all_live_nodes_list) else torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
            all_live_pos = torch.cat(all_live_pos_list, dim=0) if any(x.numel() > 0 for x in all_live_pos_list) else torch.empty((0, 2), device=self.device)
            all_live_radii = torch.cat(all_live_radii_list, dim=0) if any(x.numel() > 0 for x in all_live_radii_list) else torch.empty(0, device=self.device)
            all_live_batch = torch.cat(all_live_batch_list, dim=0) if any(x.numel() > 0 for x in all_live_batch_list) else torch.empty(0, dtype=torch.long, device=self.device)
        else:
            all_live_features = torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
            all_live_pos = torch.empty((0, 2), device=self.device)
            all_live_radii = torch.empty(0, device=self.device)
            all_live_batch = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Step 2: Batch all memory foveal nodes (after deduplication)
        all_mem_fov_features_list = []
        all_mem_fov_pos_list = []
        all_mem_fov_radii_list = []
        all_mem_fov_batch_list = []
        mem_fov_offset_start = node_offset
        
        for i in range(num_agents_in_batch):
            live_fovea_env_ids = live_fovea_graph_list[i].x[:, self.env_id_idx].long() if live_fovea_graph_list[i].num_nodes > 0 else torch.tensor([], device=self.device, dtype=torch.long)
            
            # Get all memory nodes for this agent
            agent_mem_mask = self.mem_agent_idx == i
            agent_mem_indices = torch.where(agent_mem_mask)[0]
            
            # Filter out duplicates
            if agent_mem_indices.numel() > 0 and live_fovea_env_ids.numel() > 0:
                agent_mem_env_ids = self.mem_env_id[agent_mem_indices]
                is_duplicate_mask = self._fast_isin(agent_mem_env_ids, live_fovea_env_ids)
                agent_mem_indices = agent_mem_indices[~is_duplicate_mask]
            
            # Partition into fovea
            if agent_mem_indices.numel() > 0:
                agent_mem_pos = self.mem_pos[agent_mem_indices]
                dists_sq = torch.sum((agent_mem_pos - fovea_agent_positions[i])**2, dim=1)
                radius_sq = fovea_agent_radii[i]**2
                
                fovea_mem_mask_local = dists_sq < radius_sq
                fovea_mem_indices = agent_mem_indices[fovea_mem_mask_local]
            else:
                fovea_mem_indices = torch.tensor([], dtype=torch.long, device=self.device)
            
            num_mem_fov = fovea_mem_indices.numel()
            if num_mem_fov > 0:
                mem_fov_feat = self.mem_features[fovea_mem_indices]
                mem_fov_feat_with_count = torch.zeros(num_mem_fov, MEM_NODE_FEATURE_DIM + 1, device=self.device)
                mem_fov_feat_with_count[:, :MEM_NODE_FEATURE_DIM] = mem_fov_feat
                mem_fov_feat_with_count[:, -1] = 1.0
                
                all_mem_fov_features_list.append(mem_fov_feat_with_count)
                all_mem_fov_pos_list.append(self.mem_pos[fovea_mem_indices])
                all_mem_fov_radii_list.append(torch.zeros(num_mem_fov, device=self.device))
                all_mem_fov_batch_list.append(torch.full((num_mem_fov,), i, dtype=torch.long, device=self.device))
            
            node_offset += num_mem_fov
            node_offsets_per_agent[i + 1] = node_offset
        
        # Concatenate all memory foveal nodes
        if all_mem_fov_features_list:
            all_mem_fov_features = torch.cat(all_mem_fov_features_list, dim=0)
            all_mem_fov_pos = torch.cat(all_mem_fov_pos_list, dim=0)
            all_mem_fov_radii = torch.cat(all_mem_fov_radii_list, dim=0)
            all_mem_fov_batch = torch.cat(all_mem_fov_batch_list, dim=0)
        else:
            all_mem_fov_features = torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
            all_mem_fov_pos = torch.empty((0, 2), device=self.device)
            all_mem_fov_radii = torch.empty(0, device=self.device)
            all_mem_fov_batch = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Step 3: Batch all peripheral nodes
        all_periph_features_list = []
        all_periph_pos_list = []
        all_periph_radii_list = []
        all_periph_batch_list = []
        all_periph_edge_list = []
        periph_offset_start = node_offset
        periph_node_offsets = torch.zeros(num_agents_in_batch + 1, dtype=torch.long, device=self.device)
        
        for i in range(num_agents_in_batch):
            peripheral_nodes = clustered_periphery_graphs[i]
            num_periph = peripheral_nodes.num_nodes
            
            if num_periph > 0:
                all_periph_features_list.append(peripheral_nodes.x)
                all_periph_pos_list.append(peripheral_nodes.pos)
                all_periph_radii_list.append(peripheral_nodes.radii if hasattr(peripheral_nodes, 'radii') and peripheral_nodes.radii is not None else torch.empty(num_periph, device=self.device))
                all_periph_batch_list.append(torch.full((num_periph,), i, dtype=torch.long, device=self.device))
                
                # Adjust edge indices
                if peripheral_nodes.edge_index.numel() > 0:
                    adjusted_edges = peripheral_nodes.edge_index + node_offset
                    all_periph_edge_list.append(adjusted_edges)
            
            node_offset += num_periph
            periph_node_offsets[i + 1] = node_offset
            node_offsets_per_agent[i + 1] = node_offset
        
        # Concatenate all peripheral nodes
        if all_periph_features_list:
            all_periph_features = torch.cat(all_periph_features_list, dim=0)
            all_periph_pos = torch.cat(all_periph_pos_list, dim=0)
            all_periph_radii = torch.cat(all_periph_radii_list, dim=0)
            all_periph_batch = torch.cat(all_periph_batch_list, dim=0)
            all_periph_edges = torch.cat(all_periph_edge_list, dim=1) if all_periph_edge_list else torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            all_periph_features = torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
            all_periph_pos = torch.empty((0, 2), device=self.device)
            all_periph_radii = torch.empty(0, device=self.device)
            all_periph_batch = torch.empty(0, dtype=torch.long, device=self.device)
            all_periph_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Step 4: Combine all nodes into unified graph
        final_features = torch.cat([all_live_features, all_mem_fov_features, all_periph_features], dim=0)
        final_pos = torch.cat([all_live_pos, all_mem_fov_pos, all_periph_pos], dim=0)
        final_radii = torch.cat([all_live_radii, all_mem_fov_radii, all_periph_radii], dim=0)
        final_batch = torch.cat([all_live_batch, all_mem_fov_batch, all_periph_batch], dim=0)
        
        total_nodes = final_pos.shape[0]
        
        if total_nodes == 0:
            return None
        
        # Step 5: Build edges in batched manner
        all_edges_list = []
        
        # 5a. Foveal edges (within each agent's fovea)
        for i in range(num_agents_in_batch):
            start_idx = node_offsets_per_agent[i].item()
            # Find where memory foveal nodes start for this agent
            num_live_i = (all_live_batch == i).sum().item()
            live_start = (all_live_batch[:start_idx] < i).sum().item() if start_idx > 0 else 0
            live_end = live_start + num_live_i
            
            # Memory foveal nodes for this agent
            num_mem_fov_i = (all_mem_fov_batch == i).sum().item()
            mem_fov_start = all_live_features.shape[0] + (all_mem_fov_batch[:] < i).sum().item()
            mem_fov_end = mem_fov_start + num_mem_fov_i
            
            # Get indices for this agent's foveal nodes (live + memory)
            foveal_mask = (final_batch == i) & (torch.arange(total_nodes, device=self.device) < (all_live_features.shape[0] + all_mem_fov_features.shape[0]))
            foveal_indices = torch.where(foveal_mask)[0]
            
            if foveal_indices.numel() > 1:
                foveal_pos = final_pos[foveal_indices]
                foveal_radius = fovea_agent_radii[i] * graph_connection_radius_factor
                foveal_edges_local = pyg_nn.radius_graph(foveal_pos, r=foveal_radius, max_num_neighbors=16)
                
                # Convert local indices to global
                if foveal_edges_local.numel() > 0:
                    foveal_edges_global = foveal_indices[foveal_edges_local]
                    all_edges_list.append(foveal_edges_global)
        
        # 5b. Peripheral edges (already computed and adjusted)
        if all_periph_edges.numel() > 0:
            all_edges_list.append(all_periph_edges)
        
        # 5c. Inter-edges (fovea to periphery) - Batched landmark connections
        for i in range(num_agents_in_batch):
            # Find ego node for this agent
            agent_mask = final_batch == i
            agent_foveal_mask = agent_mask & (torch.arange(total_nodes, device=self.device) < (all_live_features.shape[0] + all_mem_fov_features.shape[0]))
            
            if agent_foveal_mask.any():
                agent_foveal_features = final_features[agent_foveal_mask]
                ego_mask_local = agent_foveal_features[:, self.is_ego_idx] > 0.5
                
                if ego_mask_local.any():
                    agent_foveal_indices = torch.where(agent_foveal_mask)[0]
                    ego_idx_global = agent_foveal_indices[ego_mask_local][0]
                    
                    # Find peripheral nodes for this agent
                    periph_mask = agent_mask & (torch.arange(total_nodes, device=self.device) >= (all_live_features.shape[0] + all_mem_fov_features.shape[0]))
                    periph_indices = torch.where(periph_mask)[0]
                    
                    if periph_indices.numel() > 0:
                        periph_pos = final_pos[periph_indices]
                        periph_features = final_features[periph_indices]
                        
                        # Calculate distances
                        dists_to_agent = torch.norm(periph_pos - fovea_agent_positions[i], dim=1)
                        
                        # Select bridge nodes using simplified criteria
                        bridge_indices_local = []
                        
                        # 1. Proximity: Closest clusters
                        num_prox = min(4, periph_indices.numel())
                        if num_prox > 0:
                            prox_local = torch.topk(dists_to_agent, k=num_prox, largest=False).indices
                            bridge_indices_local.append(prox_local)
                        
                        # 2. Get agent team for content-aware connections
                        agent_team_id = -1
                        if live_fovea_graph_list[i].num_nodes > 0:
                            agent_team_id = live_fovea_graph_list[i].x[0, MEM_NODE_FEAT_IDX['team_id']]
                        
                        if agent_team_id != -1:
                            node_types_periph = periph_features[:, MEM_NODE_FEAT_IDX['node_type_encoded']]
                            
                            # Enemy clusters
                            enemy_mask = (node_types_periph == NODE_TYPE['agent']) & (periph_features[:, MEM_NODE_FEAT_IDX['team_id']] != agent_team_id)
                            if enemy_mask.any():
                                enemy_local = torch.where(enemy_mask)[0]
                                num_enemy = min(2, enemy_local.numel())
                                enemy_dists = dists_to_agent[enemy_local]
                                closest_enemies = enemy_local[torch.topk(enemy_dists, k=num_enemy, largest=False).indices]
                                bridge_indices_local.append(closest_enemies)
                            
                            # Ally clusters
                            ally_mask = (node_types_periph == NODE_TYPE['agent']) & (periph_features[:, MEM_NODE_FEAT_IDX['team_id']] == agent_team_id)
                            if ally_mask.any():
                                ally_local = torch.where(ally_mask)[0]
                                num_ally = min(2, ally_local.numel())
                                ally_dists = dists_to_agent[ally_local]
                                closest_allies = ally_local[torch.topk(ally_dists, k=num_ally, largest=False).indices]
                                bridge_indices_local.append(closest_allies)
                            
                            # Resource clusters
                            resource_mask = (node_types_periph == NODE_TYPE['resource'])
                            if resource_mask.any():
                                res_local = torch.where(resource_mask)[0]
                                num_res = min(2, res_local.numel())
                                res_dists = dists_to_agent[res_local]
                                closest_res = res_local[torch.topk(res_dists, k=num_res, largest=False).indices]
                                bridge_indices_local.append(closest_res)
                            
                            # Hives - connect to all
                            hive_mask = (node_types_periph == NODE_TYPE['hive'])
                            if hive_mask.any():
                                hive_local = torch.where(hive_mask)[0]
                                bridge_indices_local.append(hive_local)
                        
                        # Combine all bridge nodes
                        if bridge_indices_local:
                            all_bridge_local = torch.cat(bridge_indices_local).unique()
                            bridge_indices_global = periph_indices[all_bridge_local]
                            
                            # Create edges from ego to bridges
                            inter_edges = torch.stack([
                                ego_idx_global.expand(bridge_indices_global.shape[0]),
                                bridge_indices_global
                            ], dim=0)
                            all_edges_list.append(inter_edges)
        
        # 5d. Combine all edges
        if all_edges_list:
            final_edge_index = torch.cat(all_edges_list, dim=1)
            final_edge_index = torch_geometric.utils.coalesce(final_edge_index)
        else:
            final_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Step 6: Create edge attributes
        edge_attr = torch.empty((final_edge_index.shape[1], 4), dtype=torch.float32, device=self.device)
        if final_edge_index.shape[1] > 0:
            row, col = final_edge_index
            pos_src = final_pos[row]
            pos_dst = final_pos[col]
            
            # Relative position
            relative_pos = pos_dst - pos_src
            
            # Normalize by agent's observation radius (use batch index to get correct radius)
            agent_ids_edges = final_batch[row]
            obs_radii_edges = fovea_agent_radii[agent_ids_edges]
            normalized_relative_pos = relative_pos / (obs_radii_edges.unsqueeze(1) + 1e-6)
            
            # Normalized distance
            distance = torch.norm(relative_pos, dim=1, keepdim=True)
            normalized_distance = distance / (obs_radii_edges.unsqueeze(1) + 1e-6)
            
            # Connection strength
            connection_strength = torch.ones(row.shape[0], 1, device=self.device)
            
            edge_attr = torch.cat([
                normalized_relative_pos,
                normalized_distance,
                connection_strength
            ], dim=1)
        
        # Step 7: Create final masks
        ego_mask = final_features[:, NODE_FEATURE_MAP['is_ego']] > 0.5 if final_features.numel() > 0 else torch.empty((0,), dtype=torch.bool, device=self.device)
        is_live_mask = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
        is_live_mask[:all_live_features.shape[0]] = True
        
        # Step 8: Create unified data object
        unified_data = Data(
            x=final_features,
            pos=final_pos,
            edge_index=final_edge_index,
            edge_attr=edge_attr,
            is_ego=ego_mask,
            is_live=is_live_mask,
            radii=final_radii,
            batch=final_batch
        )
        
        # Convert to batch format for compatibility
        # Create individual graphs from the unified graph
        final_graphs = []
        for i in range(num_agents_in_batch):
            agent_mask = final_batch == i
            agent_node_indices = torch.where(agent_mask)[0]
            
            if agent_node_indices.numel() == 0:
                continue
            
            # Extract subgraph for this agent
            agent_features = final_features[agent_node_indices]
            agent_pos = final_pos[agent_node_indices]
            agent_radii = final_radii[agent_node_indices]
            agent_ego_mask = ego_mask[agent_node_indices]
            agent_is_live = is_live_mask[agent_node_indices]
            
            # Remap edges
            # Create mapping from global to local indices
            node_mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=self.device)
            node_mapping[agent_node_indices] = torch.arange(agent_node_indices.numel(), device=self.device)
            
            # Find edges where both endpoints belong to this agent
            edge_mask = agent_mask[final_edge_index[0]] & agent_mask[final_edge_index[1]]
            agent_edges_global = final_edge_index[:, edge_mask]
            agent_edge_attr = edge_attr[edge_mask] if edge_attr.numel() > 0 else torch.empty((0, 4), device=self.device)
            
            # Remap to local indices
            agent_edges_local = node_mapping[agent_edges_global]
            
            final_graphs.append(Data(
                x=agent_features,
                pos=agent_pos,
                edge_index=agent_edges_local,
                edge_attr=agent_edge_attr,
                is_ego=agent_ego_mask,
                is_live=agent_is_live,
                radii=agent_radii,
                agent_id=torch.tensor([i], dtype=torch.long, device=self.device)
            ))
        
        if not final_graphs:
            return None
        
        return Batch.from_data_list(final_graphs)
    
    # Keep all the helper methods unchanged
    def _get_foveal_nodes(self, live_fovea_graph: Data, current_step: int) -> Dict[str, torch.Tensor]:
        """
        Pads the features of a live fovea graph to match the memory graph format.
        This logic is identical to the foveal processing in the original implementation.
        """
        TARGET_FEAT_DIM = MEM_NODE_FEATURE_DIM + 1
        live_features = live_fovea_graph.x
        num_foveal_nodes = live_features.shape[0]

        if num_foveal_nodes == 0:
             return {'x': torch.empty(0, TARGET_FEAT_DIM, device=self.device),
                     'pos': torch.empty(0, 2, device=self.device),
                     'radii': torch.empty(0, device=self.device)}

        if live_features.shape[1] == TARGET_FEAT_DIM:
            # Already has the full dimension, update last_observed_step for live nodes
            foveal_feat_with_count = live_features.clone()
            step_value = torch.full((num_foveal_nodes,), float(current_step), device=self.device, dtype=foveal_feat_with_count.dtype)
            foveal_feat_with_count[:, self.last_seen_idx] = step_value
        elif live_features.shape[1] == NODE_FEATURE_DIM:
            foveal_feat_with_count = torch.zeros(num_foveal_nodes, TARGET_FEAT_DIM, device=self.device, dtype=live_features.dtype)
            foveal_feat_with_count[:, :NODE_FEATURE_DIM] = live_features
            step_value = torch.full((num_foveal_nodes,), float(current_step), device=self.device, dtype=foveal_feat_with_count.dtype)
            foveal_feat_with_count[:, self.last_seen_idx] = step_value
            foveal_feat_with_count[:, -1] = 1.0  # Count feature
        else:
            raise ValueError(f"generate_foveated_graph received live_features with unexpected shape: {live_features.shape}")

        return {'x': foveal_feat_with_count, 'pos': live_fovea_graph.pos, 'radii': live_fovea_graph.radii}


    def _update_peripheral_cache_for_agent(self, agent_idx: int, fovea_agent_pos: torch.Tensor,
                                           fovea_agent_radius: torch.Tensor, mid_periphery_scale: float,
                                           mid_cluster_cell_size: float, far_cluster_cell_size: float):
        """
        Performs clustering of peripheral nodes for a single agent and caches the result.
        This logic is extracted and adapted from the original `_update_peripheral_cache`.
        """
        agent_mem_mask = (self.mem_agent_idx == agent_idx)
        if not agent_mem_mask.any():
            self.periphery_graph_cache[agent_idx] = {'x': torch.empty(0, MEM_NODE_FEATURE_DIM + 1, device=self.device),
                                                     'pos': torch.empty(0, 2, device=self.device),
                                                     'radii': torch.empty(0, device=self.device)}
            return

        agent_mem_pos = self.mem_pos[agent_mem_mask]
        agent_mem_features = self.mem_features[agent_mem_mask]
        
        # Retrieve stored radii instead of creating a placeholder
        if hasattr(self, 'mem_radii') and self.mem_radii is not None:
            agent_mem_radii = self.mem_radii[agent_mem_mask]
        else:
            # Fallback for safety if radii haven't been stored for some reason
            agent_mem_radii = torch.zeros(agent_mem_pos.shape[0], device=self.device)


        # --- Determine which memory nodes are in the fovea vs. periphery ---
        is_in_fovea_mask = torch.zeros(agent_mem_pos.shape[0], dtype=torch.bool, device=self.device)
        if agent_mem_pos.numel() > 0 and fovea_agent_pos.numel() > 0:
            dists = torch.cdist(agent_mem_pos, fovea_agent_pos)
            is_in_range = dists < fovea_agent_radius
            is_in_fovea_mask = torch.any(is_in_range, dim=1)

        # --- Partition and Cluster Peripheral Nodes ---
        peripheral_mask = ~is_in_fovea_mask
        peripheral_pos = agent_mem_pos[peripheral_mask]
        
        mid_super_node_pos = torch.empty((0, 2), device=self.device)
        mid_super_node_feat = torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
        mid_super_node_radii = torch.empty((0,), device=self.device)
        far_super_node_pos = torch.empty((0, 2), device=self.device)
        far_super_node_feat = torch.empty((0, MEM_NODE_FEATURE_DIM + 1), device=self.device)
        far_super_node_radii = torch.empty((0,), device=self.device)

        if peripheral_pos.numel() > 0:
            fovea_center_point = fovea_agent_pos.mean(dim=0)
            distances_sq_peripheral = torch.sum((peripheral_pos - fovea_center_point) ** 2, dim=1)

            mid_periphery_radius = fovea_agent_radius.max() * mid_periphery_scale
            mid_mask = distances_sq_peripheral <= mid_periphery_radius ** 2
            far_mask = ~mid_mask

            peripheral_indices = torch.where(peripheral_mask)[0]
            mid_indices = peripheral_indices[mid_mask]
            far_indices = peripheral_indices[far_mask]

            if mid_indices.numel() > 0:
                mid_super_node_pos, mid_super_node_feat, mid_super_node_radii, _ = cluster_nodes_by_voxel(
                    agent_mem_pos[mid_indices], agent_mem_features[mid_indices],
                    agent_mem_radii[mid_indices], mid_cluster_cell_size
                )

            if far_indices.numel() > 0:
                far_super_node_pos, far_super_node_feat, far_super_node_radii, _ = cluster_nodes_by_voxel(
                    agent_mem_pos[far_indices], agent_mem_features[far_indices],
                    agent_mem_radii[far_indices], far_cluster_cell_size
                )
        
        self.periphery_graph_cache[agent_idx] = {
            'x': torch.cat([mid_super_node_feat, far_super_node_feat], dim=0),
            'pos': torch.cat([mid_super_node_pos, far_super_node_pos], dim=0),
            'radii': torch.cat([mid_super_node_radii, far_super_node_radii], dim=0)
        }

    def _partition_memory_batched(self, agent_positions: torch.Tensor, agent_fovea_radii: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For a batch of agents, partitions the entire memory bank into fovea and periphery nodes
        for each agent.
        
        Returns:
            fovea_mask_dense: A [num_agents, num_memory_nodes] boolean tensor. True if mem_node j is in agent i's fovea.
            periphery_mask_dense: The inverse of the fovea mask for nodes OWNED by the agent.
            agent_indices: A [num_agents] tensor mapping back to the original batch order.
            mem_indices: A [num_memory_nodes] tensor.
        """
        num_agents = agent_positions.shape[0]
        num_memory_nodes = self.mem_pos.shape[0]

        # Create a mask to ensure agents only access their OWN memory ---
        # self.mem_agent_idx contains the owning agent's index for each memory node.
        # We broadcast it against the agent indices to create a [N_agents, N_mem] ownership mask.
        agent_indices_col = torch.arange(num_agents, device=self.device).unsqueeze(1)
        owns_mask = (self.mem_agent_idx == agent_indices_col)

        # Expand dims to enable broadcasting
        agent_pos_expanded = agent_positions.unsqueeze(1) # [N_agents, 1, 2]
        mem_pos_expanded = self.mem_pos.unsqueeze(0) # [1, N_mem, 2]
        
        # Calculate squared distances for efficiency
        dist_sq = torch.sum((agent_pos_expanded - mem_pos_expanded) ** 2, dim=-1) # [N_agents, N_mem]
        
        # Fovea mask based on distance
        radii_sq = agent_fovea_radii.unsqueeze(1) ** 2 # [N_agents, 1]
        in_radius_mask = dist_sq < radii_sq
        
        # Combine distance mask with ownership mask ---
        fovea_mask_dense = in_radius_mask & owns_mask
        
        # Periphery mask: Nodes must be owned by the agent AND outside its fovea
        periphery_mask_dense = ~in_radius_mask & owns_mask
        
        # Agent and memory indices for reference
        agent_indices = torch.arange(num_agents, device=self.device)
        mem_indices = torch.arange(num_memory_nodes, device=self.device)
        
        return fovea_mask_dense, periphery_mask_dense, agent_indices, mem_indices

    def _get_clustered_periphery_batched(self, agent_positions: torch.Tensor, agent_fovea_radii: torch.Tensor,
                                        live_fovea_graph_list: List[Data],
                                        current_step: int,
                                        clustering_frequency: int,
                                        min_cluster_size: float,
                                        max_cluster_size: float,
                                        cluster_aggressiveness: float,
                                        mem_skeleton_connection_factor: float,
                                        cluster_exclusion_radius_factor: float,
                                        detailed_clustering_radius_factor: float) -> List[Data]:
        """
        Generates clustered periphery graphs for a batch of agents, with caching.
        """
        num_agents = agent_positions.shape[0]
        
        if self.mem_pos is None or self.mem_pos.numel() == 0:
            return [Data(x=torch.empty((0, self.d_feat_mem + 1), device=self.device),
                pos=torch.empty((0, 2), device=self.device),
                         edge_index=torch.empty((2,0), dtype=torch.long, device=self.device),
                         radii=torch.empty(0, device=self.device))
                    for _ in range(num_agents)]

        _, periphery_mask_dense, _, _ = self._partition_memory_batched(agent_positions, agent_fovea_radii)
        
        clustered_graphs = []
        for i in range(num_agents):
            agent_pos = agent_positions[i]

            # --- Caching Logic ---
            use_cache = False
            cached_graph = self.periphery_graph_cache.get(i)
            last_step = self.last_cluster_step.get(i)
            last_pos = self.last_cluster_pos.get(i)

            if cached_graph is not None and last_step is not None and last_pos is not None:
                steps_since_cluster = current_step - last_step
                pos_diff = torch.norm(agent_pos - last_pos)
                
                # Invalidate if too much time has passed OR agent moved significantly
                if steps_since_cluster < clustering_frequency and pos_diff < (agent_fovea_radii[i] * 0.25):
                    use_cache = True

            if use_cache:
                clustered_graphs.append(cached_graph)
                continue

            # --- Re-clustering Logic (if cache not used) ---
            self.last_cluster_step[i] = current_step
            self.last_cluster_pos[i] = agent_pos.clone()

            periphery_node_indices = torch.where(periphery_mask_dense[i])[0]

            # The global de-duplication happens *after* clustering is called.
            # To prevent live foveal nodes from being clustered, we must de-duplicate here.
            if periphery_node_indices.numel() > 0:
                live_fovea_graph = live_fovea_graph_list[i]
                if live_fovea_graph.num_nodes > 0:
                    live_env_ids = live_fovea_graph.x[:, MEM_NODE_FEAT_IDX['agent_id']].long()
                    periphery_env_ids = self.mem_env_id[periphery_node_indices]
                    
                    is_duplicate_mask = self._fast_isin(periphery_env_ids, live_env_ids)
                    periphery_node_indices = periphery_node_indices[~is_duplicate_mask]

            if periphery_node_indices.numel() == 0:
                empty_graph = Data(
                    x=torch.empty((0, self.d_feat_mem + 1), device=self.device), 
                    pos=torch.empty((0, 2), device=self.device), 
                    edge_index=torch.empty((2,0), dtype=torch.long, device=self.device),
                    radii=torch.empty(0, device=self.device)
                )
                clustered_graphs.append(empty_graph)
                self.periphery_graph_cache[i] = empty_graph
                continue

            periphery_pos = self.mem_pos[periphery_node_indices]
            periphery_features = self.mem_features[periphery_node_indices]
            if hasattr(self, 'mem_radii') and self.mem_radii is not None:
                periphery_radii = self.mem_radii[periphery_node_indices]
            else:
                periphery_radii = torch.zeros(periphery_pos.shape[0], device=self.device)
            
            # --- EXCLUSION RADIUS & SMOOTHER SCALING ---
            dists = torch.norm(periphery_pos - agent_pos, dim=1)
            
            # 1. Apply an exclusion radius where no clustering occurs.
            exclusion_radius = agent_fovea_radii[i] * cluster_exclusion_radius_factor
            clusterable_mask = dists > exclusion_radius
            
            # --- LEVEL OF DETAIL (LOD) CLUSTERING ---
            detailed_radius = agent_fovea_radii[i] * detailed_clustering_radius_factor
            
            unclusterable_mask = ~clusterable_mask
            mid_periphery_mask = clusterable_mask & (dists <= detailed_radius)
            far_periphery_mask = clusterable_mask & (dists > detailed_radius)

            unclusterable_indices = periphery_node_indices[unclusterable_mask]
            mid_periphery_indices = periphery_node_indices[mid_periphery_mask]
            far_periphery_indices = periphery_node_indices[far_periphery_mask]

            # Zone 1: Unclustered nodes (pass through)
            unclustered_pos = self.mem_pos[unclusterable_indices]
            unclustered_features = self.mem_features[unclusterable_indices]
            unclustered_radii = periphery_radii[unclusterable_mask]
            unclustered_count = torch.ones(unclustered_pos.shape[0], 1, device=self.device)
            unclustered_final_features = torch.cat([unclustered_features, unclustered_count], dim=1)

            all_clustered_pos_list = []
            all_clustered_features_list = []
            all_clustered_radii_list = []

            # --- Zone 2: Mid-Periphery (Detailed Clustering) ---
            if mid_periphery_indices.numel() > 0:
                clusterable_features_mid = self.mem_features[mid_periphery_indices]
                node_types_mid = clusterable_features_mid[:, MEM_NODE_FEAT_IDX['node_type_encoded']]
                unique_node_types_mid = torch.unique(node_types_mid)

                for node_type in unique_node_types_mid:
                    if node_type == NODE_TYPE['agent']:
                        type_mask_mid = (node_types_mid == node_type)
                        agent_indices_local_mid = torch.where(type_mask_mid)[0]
                        agent_features_subset = clusterable_features_mid[agent_indices_local_mid]
                        team_ids_subset = agent_features_subset[:, MEM_NODE_FEAT_IDX['team_id']]
                        unique_teams = torch.unique(team_ids_subset)
                        
                        for team_id in unique_teams:
                            team_mask_local = (team_ids_subset == team_id)
                            subset_indices_local = agent_indices_local_mid[team_mask_local]
                            self._perform_clustering_on_subset(
                                subset_indices_local, mid_periphery_indices, dists, mid_periphery_mask,
                                exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                                all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                            )
                    elif node_type == NODE_TYPE['hive']:
                        type_mask_mid = (node_types_mid == node_type)
                        hive_indices_local_mid = torch.where(type_mask_mid)[0]
                        hive_features_subset = clusterable_features_mid[hive_indices_local_mid]
                        team_ids_subset = hive_features_subset[:, MEM_NODE_FEAT_IDX['team_id']]
                        unique_teams = torch.unique(team_ids_subset)
                        
                        for team_id in unique_teams:
                            team_mask_local = (team_ids_subset == team_id)
                            subset_indices_local = hive_indices_local_mid[team_mask_local]
                            self._perform_clustering_on_subset(
                                subset_indices_local, mid_periphery_indices, dists, mid_periphery_mask,
                                exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                                all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                            )
                    elif node_type == NODE_TYPE['resource']:
                        type_mask_mid = (node_types_mid == node_type)
                        res_indices_local_mid = torch.where(type_mask_mid)[0]
                        res_features_subset = clusterable_features_mid[res_indices_local_mid]
                        is_coop_feature = res_features_subset[:, MEM_NODE_FEAT_IDX['is_cooperative']] > 0.5

                        # Coop resources
                        self._perform_clustering_on_subset(
                            res_indices_local_mid[is_coop_feature], mid_periphery_indices, dists, mid_periphery_mask,
                            exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                            all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                        )
                        # Single-agent resources
                        self._perform_clustering_on_subset(
                            res_indices_local_mid[~is_coop_feature], mid_periphery_indices, dists, mid_periphery_mask,
                            exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                            all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                        )
                    else: # Other types (obstacles)
                        type_mask_mid = (node_types_mid == node_type)
                        subset_indices_local = torch.where(type_mask_mid)[0]
                        self._perform_clustering_on_subset(
                            subset_indices_local, mid_periphery_indices, dists, mid_periphery_mask,
                            exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                            all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                        )
            
            # --- Zone 3: Far-Periphery (Generic Clustering) ---
            if far_periphery_indices.numel() > 0:
                clusterable_features_far = self.mem_features[far_periphery_indices]
                node_types_far = clusterable_features_far[:, MEM_NODE_FEAT_IDX['node_type_encoded']]
                unique_node_types_far = torch.unique(node_types_far)

                for node_type in unique_node_types_far:
                    # Group only by node_type, ignoring team or coop status
                    type_mask_far = (node_types_far == node_type)
                    subset_indices_local = torch.where(type_mask_far)[0]
                    self._perform_clustering_on_subset(
                        subset_indices_local, far_periphery_indices, dists, far_periphery_mask,
                        exclusion_radius, min_cluster_size, max_cluster_size, cluster_aggressiveness,
                        all_clustered_pos_list, all_clustered_features_list, all_clustered_radii_list, node_type
                    )

            # Combine clustered results
            clustered_pos = torch.cat(all_clustered_pos_list, dim=0) if all_clustered_pos_list else torch.empty((0, 2), device=self.device)
            clustered_features = torch.cat(all_clustered_features_list, dim=0) if all_clustered_features_list else torch.empty((0, self.d_feat_mem + 1), device=self.device)
            clustered_radii = torch.cat(all_clustered_radii_list, dim=0) if all_clustered_radii_list else torch.empty(0, device=self.device)

            # --- COMBINE UNCLUSTERED AND CLUSTERED NODES ---
            final_pos = torch.cat([unclustered_pos, clustered_pos], dim=0)
            final_features = torch.cat([unclustered_final_features, clustered_features], dim=0)
            final_radii = torch.cat([unclustered_radii, clustered_radii], dim=0)

            if final_pos.numel() > 0:
                # --- LOCALIZED RADIUS CONNECTIONS ---
                if final_pos.shape[0] > 1:
                    connection_radius_local = min_cluster_size * mem_skeleton_connection_factor 
                    edge_index = pyg_nn.radius_graph(final_pos, r=connection_radius_local, max_num_neighbors=16)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                
                graph_data = Data(x=final_features, pos=final_pos, edge_index=edge_index, radii=final_radii)
                clustered_graphs.append(graph_data)
                self.periphery_graph_cache[i] = graph_data
            else:
                empty_graph = Data(
                    x=torch.empty((0, self.d_feat_mem + 1), device=self.device), 
                    pos=torch.empty((0, 2), device=self.device), 
                    edge_index=torch.empty((2,0), dtype=torch.long, device=self.device),
                    radii=torch.empty(0, device=self.device)
                )
                clustered_graphs.append(empty_graph)
                self.periphery_graph_cache[i] = empty_graph

        return clustered_graphs
        
    def _perform_clustering_on_subset(self,
                                      subset_indices_local: torch.Tensor,
                                      original_indices_global: torch.Tensor,
                                      dists_full_periphery: torch.Tensor,
                                      zone_mask: torch.Tensor,
                                      exclusion_radius: float,
                                      min_cluster_size: float,
                                      max_cluster_size: float,
                                      cluster_aggressiveness: float,
                                      all_clustered_pos_list: list,
                                      all_clustered_features_list: list,
                                      all_clustered_radii_list: list,
                                      node_type: int):
        """
        Refactored helper to run the full clustering pipeline on a subset of nodes.
        """
        if subset_indices_local.numel() == 0:
            return

        type_indices_global = original_indices_global[subset_indices_local]
        clusterable_pos_type = self.mem_pos[type_indices_global]
        clusterable_features_type = self.mem_features[type_indices_global]
        if hasattr(self, 'mem_radii') and self.mem_radii is not None:
            clusterable_radii_type = self.mem_radii[type_indices_global]
        else:
            clusterable_radii_type = torch.zeros(clusterable_pos_type.shape[0], device=self.device)

        # Correctly slice the distances for the current subset
        dists_zone = dists_full_periphery[zone_mask]
        clusterable_dists_type = dists_zone[subset_indices_local]

        current_aggressiveness = cluster_aggressiveness
        if node_type == NODE_TYPE['resource']:
            current_aggressiveness *= 2.5
        
        distance_scale = (clusterable_dists_type / exclusion_radius).clamp(min=1.0)
        cluster_sizes = (min_cluster_size * distance_scale * current_aggressiveness).clamp(max=max_cluster_size)

        if cluster_sizes.numel() > 0:
            avg_cluster_size = torch.mean(cluster_sizes)
            cluster_assignment = pyg_nn.pool.voxel_grid(clusterable_pos_type, batch=torch.zeros(clusterable_pos_type.shape[0], dtype=torch.long, device=self.device), size=avg_cluster_size)
        else:
            return

        # --- HYBRID MERGE (applied per-type) ---
        unique_clusters, inverse_indices = torch.unique(cluster_assignment, return_inverse=True)
        num_clusters_before_merge = unique_clusters.shape[0]

        if num_clusters_before_merge > 1:
            cluster_counts = scatter(torch.ones_like(inverse_indices, dtype=torch.float), inverse_indices, dim=0, reduce='sum')
            cluster_pos_sum = scatter(clusterable_pos_type, inverse_indices, dim=0, reduce='sum')
            cluster_centers = cluster_pos_sum / cluster_counts.unsqueeze(1)
            adaptive_merge_threshold = avg_cluster_size * 1.5
            dist_matrix = torch.cdist(cluster_centers, cluster_centers)
            merge_pairs = torch.where((dist_matrix > 0) & (dist_matrix < adaptive_merge_threshold))
            parent = torch.arange(num_clusters_before_merge, device=self.device)
            indices_i, indices_j = merge_pairs[0], merge_pairs[1]
            for i_idx, j_idx in zip(indices_i, indices_j):
                if i_idx < j_idx:
                    root_i = _find_root(parent, i_idx.item())
                    root_j = _find_root(parent, j_idx.item())
                    if root_i != root_j:
                        if cluster_counts[root_i] < cluster_counts[root_j]:
                            parent[root_i] = root_j
                            cluster_counts[root_j] += cluster_counts[root_i]
                        else:
                            parent[root_j] = root_i
                            cluster_counts[root_i] += cluster_counts[root_j]
            for k in range(num_clusters_before_merge):
                _find_root(parent, k)
            final_roots, new_inverse_indices = torch.unique(parent[inverse_indices], return_inverse=True)
            inverse_indices = new_inverse_indices
        
        # --- AGGREGATION (applied per-type) ---
        clustered_pos_type = scatter(clusterable_pos_type, inverse_indices, dim=0, reduce='mean')
        clustered_radii_type = scatter(clusterable_radii_type, inverse_indices, dim=0, reduce='mean')

        max_last_seen_per_cluster = scatter(clusterable_features_type[:, self.last_seen_idx], inverse_indices, dim=0, reduce='max')
        global_max_last_seen = max_last_seen_per_cluster[inverse_indices]
        is_max_node = (clusterable_features_type[:, self.last_seen_idx] == global_max_last_seen)
        
        max_indices = torch.where(is_max_node)[0]
        max_inverse = inverse_indices[is_max_node]
        sorted_max_inverse, sorted_indices = torch.sort(max_inverse)
        sorted_max_indices = max_indices[sorted_indices]
        
        first_occurrence_mask = torch.cat([torch.tensor([True], device=self.device), sorted_max_inverse[1:] != sorted_max_inverse[:-1]])
        first_occurrence_indices = torch.where(first_occurrence_mask)[0]
        
        original_indices = sorted_max_indices[first_occurrence_indices]
        clustered_features_base_type = clusterable_features_type[original_indices]
        
        count_feature = scatter(torch.ones(clusterable_features_type.shape[0], 1, device=self.device), inverse_indices, dim=0, reduce='sum')
        clustered_final_features_type = torch.cat([clustered_features_base_type, count_feature], dim=1)

        all_clustered_pos_list.append(clustered_pos_type)
        all_clustered_features_list.append(clustered_final_features_type)
        all_clustered_radii_list.append(clustered_radii_type)

    def reset(self):
        """
        Resets the memory state, clearing all tensors.
        """
        self.mem_features = None
        self.mem_pos = None
        self.mem_agent_idx = None
        self.mem_env_id = None
        self.mem_radii = None # Reset radii as well
        self.d_feat_in = -1

        # Reset caches
        self.periphery_graph_cache = {i: None for i in range(self.num_agents)}
        self.last_cluster_step = {i: -1 for i in range(self.num_agents)}
        self.last_cluster_pos = {i: None for i in range(self.num_agents)}

