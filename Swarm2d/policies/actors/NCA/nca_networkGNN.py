# nca_network.py
#!/usr/bin/env python3
"""
Agent Policy Implementation - Featuring PINSAN-NCA

This file implements the PINSANPolicy, a novel actor architecture based on the
Physics-Informed Neuro-Symbolic Adaptive Neighborhood Neural Cellular Automata
(PINSAN-NCA) concept combined with an internally generated, decentralized belief state
and attentive neighbor processing.

The belief state is updated using both self-observations and features
extracted from the agent's local map observation.
Includes conditional logic based on ablation flags.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from typing import Optional, Dict, Tuple, Union, List
import traceback
import copy # For deepcopying roles if needed
import torch.distributions as D
import math # For sqrt in attention
# Calculate the path to the project root (assuming this script is in src/agents/)
project_root_nca = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_nca not in sys.path:
    sys.path.insert(0, project_root_nca)
from constants import (
    MEM_NODE_FEAT_IDX, MEM_NODE_FEATURE_DIM, OCC_CH_COUNT, NODE_FEATURE_DIM, NODE_FEATURE_MAP,
    NODE_TYPE, NUM_NODE_TYPES, RAW_CH_COUNT, RAW_CH, SELF_OBS_MAP
)
from env.observations import create_graph_edge_features

# SAC specific constants
LOG_STD_MAX = 2
LOG_STD_MIN = -5 # Often -20, but -5 can be more stable
epsilon = 1e-6

# --- PyG Imports ---
try:
    import torch_geometric
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_batch
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn import radius_graph # Crucial for AN-NCA
    # Use torch_scatter directly for attention softmax and aggregation
    from torch_scatter import scatter_add, scatter_mean, scatter

except ImportError:
    print("ERROR: PyTorch Geometric or torch_scatter not found. PINSANPolicy requires them.")
    print("Please install PyG (pyg.org) and torch_scatter (`pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html`)")
    sys.exit(1)

# Set device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchedExternalMemory(nn.Module):
    """
    (V2 - Fully Batched) External Memory module with batched read and write operations.
    Manages memories for all agents in a team with a single tensor.
    """
    def __init__(self, num_agents, num_slots=16, memory_dim=128, query_dim=128, num_read_heads=1):
        super().__init__()
        self.num_agents = num_agents
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.num_read_heads = num_read_heads
        
        # Memory is now a single batched tensor: [NumAgents, NumSlots, MemDim]
        self.memory = nn.Parameter(torch.empty(num_agents, num_slots, memory_dim))
        nn.init.xavier_uniform_(self.memory)

        self.query_proj = nn.Linear(query_dim, memory_dim * num_read_heads)
        self.erase_layer = nn.Linear(query_dim, memory_dim)
        self.add_layer = nn.Linear(query_dim, memory_dim)
        self.read_fusion = nn.Sequential(
            nn.Linear(memory_dim * num_read_heads, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim)
        )
        self.write_gate = nn.Linear(query_dim, 1)

    def read(self, queries: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:
        """
        Batched read from the memories of specified agents.
        Args:
            queries (Tensor): Query vectors for reading. Shape: [B, query_dim].
            agent_indices (Tensor): The indices of the agents whose memories to read from. Shape: [B].
        Returns:
            Fused read vectors. Shape: [B, query_dim].
        """
        B = queries.size(0)
        # Select the memory banks for the agents in the batch
        agent_memories = self.memory[agent_indices] # [B, NumSlots, MemDim]

        processed_queries = self.query_proj(queries).view(B, self.num_read_heads, self.memory_dim)
        
        # Normalize memories and queries for cosine similarity
        mem_norm = F.normalize(agent_memories, p=2, dim=2) # [B, NumSlots, MemDim]
        
        read_vecs_list = []
        for h in range(self.num_read_heads):
            query_h = F.normalize(processed_queries[:, h, :], p=2, dim=1).unsqueeze(1) # [B, 1, MemDim]
            
            # Batched matrix multiply for attention scores
            attn_scores_h = torch.bmm(query_h, mem_norm.transpose(1, 2)).squeeze(1) # [B, NumSlots]
            attn_weights_h = F.softmax(attn_scores_h / math.sqrt(self.memory_dim), dim=1) # [B, NumSlots]
            
            # Batched matrix multiply to get read vectors
            read_vec_h = torch.bmm(attn_weights_h.unsqueeze(1), agent_memories).squeeze(1) # [B, MemDim]
            read_vecs_list.append(read_vec_h)
            
        read_vec_concat = torch.cat(read_vecs_list, dim=1) # [B, MemDim * NumHeads]
        fused_read = self.read_fusion(read_vec_concat)
        return fused_read

    def write(self, write_queries: torch.Tensor, agent_indices: torch.Tensor):
        """
        Batched write to the memories of specified agents.
        Args:
            write_queries (Tensor): Query vectors for writing. Shape: [B, query_dim].
            agent_indices (Tensor): The indices of the agents whose memories to write to. Shape: [B].
        """
        B = write_queries.size(0)
        if B == 0: return

        # --- Calculate Write Parameters (Batched) ---
        processed_query = self.query_proj(write_queries).view(B, self.num_read_heads, self.memory_dim)[:, 0, :]
        query_norm = F.normalize(processed_query, p=2, dim=1)
        
        erase_signal = torch.sigmoid(self.erase_layer(write_queries)) # [B, MemDim]
        add_signal = torch.tanh(self.add_layer(write_queries))       # [B, MemDim]
        write_intensity = torch.sigmoid(self.write_gate(write_queries)) # [B, 1]

        # --- Scatter-based Aggregation ---
        # We need to perform an aggregated write, where all writes destined for the
        # same memory bank (agent_idx) are averaged together.
        unique_agent_indices, inverse_indices = torch.unique(agent_indices, return_inverse=True)
        num_unique_agents = unique_agent_indices.size(0)

        # Select the memory banks for the unique agents that need updating
        mem_to_update = self.memory.detach()[unique_agent_indices] # [U, NumSlots, MemDim]
        mem_norm = F.normalize(mem_to_update, p=2, dim=2)

        # Scatter queries to align with unique agent memories for bmm
        scattered_query_norm = torch.zeros(num_unique_agents, B, self.memory_dim, device=self.memory.device)
        scattered_query_norm.index_add_(0, inverse_indices, query_norm.unsqueeze(0).expand(num_unique_agents, -1, -1))
        
        # This part is complex to fully vectorize correctly. A simpler, robust approach is to
        # aggregate the write signals first, then apply to memory.
        
        # 1. Calculate attention weights for all queries against their target memories
        # This requires expanding memories to match the query batch size before bmm
        target_mems_expanded = mem_to_update[inverse_indices] # [B, NumSlots, MemDim]
        target_mems_norm_expanded = F.normalize(target_mems_expanded, p=2, dim=2)
        
        attn_scores = torch.bmm(query_norm.unsqueeze(1), target_mems_norm_expanded.transpose(1, 2)).squeeze(1) # [B, NumSlots]
        write_weights = F.softmax(attn_scores / math.sqrt(self.memory_dim), dim=1) # [B, NumSlots]

        # 2. Compute erase/add signals per slot for each write operation
        erase_per_slot = write_weights.unsqueeze(2) * erase_signal.unsqueeze(1) # [B, NumSlots, MemDim]
        add_per_slot = write_weights.unsqueeze(2) * add_signal.unsqueeze(1)     # [B, NumSlots, MemDim]
        
        # 3. Apply write intensity
        erase_per_slot *= write_intensity.unsqueeze(1)
        add_per_slot *= write_intensity.unsqueeze(1)

        # 4. Aggregate these signals per unique memory bank using scatter_add
        final_erase = torch.zeros_like(mem_to_update) # [U, NumSlots, MemDim]
        final_add = torch.zeros_like(mem_to_update)   # [U, NumSlots, MemDim]
        
        final_erase.scatter_add_(0, inverse_indices.view(-1, 1, 1).expand_as(erase_per_slot), erase_per_slot)
        final_add.scatter_add_(0, inverse_indices.view(-1, 1, 1).expand_as(add_per_slot), add_per_slot)

        # 5. Update the memory banks
        new_memory_subset = mem_to_update * (1 - final_erase) + final_add
        
        # 6. Normalize and write back to the main memory parameter
        with torch.no_grad():
            self.memory[unique_agent_indices] = F.normalize(new_memory_subset, p=2, dim=2)


def scatter_softmax_manual(src, index, dim=-1, dim_size=None, epsilon=1e-12):
    """Manually compute softmax using scatter ops for stability."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if dim_size == 0: # Handle empty index case
        return torch.empty_like(src)

    with torch.no_grad():
        src_max = scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
        src_max[torch.isneginf(src_max)] = 0
    src_max_expanded = src_max.gather(dim, index)
    out = torch.exp(src - src_max_expanded)
    out_sum = scatter_add(out, index, dim=dim, dim_size=dim_size)
    out_sum_expanded = out_sum.gather(dim, index)
    return out / (out_sum_expanded + epsilon)


class ExternalMemory(nn.Module):
    # Neural Turing Machine
    """External Memory module with read and write operations."""
    def __init__(self, num_slots=16, memory_dim=128, query_dim=128, num_read_heads=1):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.num_read_heads = num_read_heads
        self.register_parameter('memory', nn.Parameter(torch.empty(num_slots, memory_dim)))
        nn.init.xavier_uniform_(self.memory)
        self.query_proj = nn.Linear(query_dim, memory_dim * num_read_heads)
        self.erase_layer = nn.Linear(query_dim, memory_dim)
        self.add_layer = nn.Linear(query_dim, memory_dim)
        self.read_fusion = nn.Sequential(
            nn.Linear(memory_dim * num_read_heads, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim)
        )
        self.write_gate = nn.Linear(query_dim, 1)

    def read(self, query):
        current_device = self.memory.device
        query = query.to(current_device)
        B = query.size(0)
        processed_queries = self.query_proj(query).view(B, self.num_read_heads, self.memory_dim)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        attn_weights_list = []
        for h in range(self.num_read_heads):
            query_h = F.normalize(processed_queries[:, h, :], p=2, dim=1)
            attn_scores_h = torch.matmul(query_h, mem_norm.t())
            attn_weights_h = torch.softmax(attn_scores_h / math.sqrt(self.memory_dim), dim=1)
            attn_weights_list.append(attn_weights_h)
        read_vecs = [torch.matmul(attn_weights_list[h], self.memory) for h in range(self.num_read_heads)]
        read_vec_concat = torch.cat(read_vecs, dim=1)
        fused_read = self.read_fusion(read_vec_concat)
        return fused_read

    def write(self, write_query):
        current_device = self.memory.device
        write_query = write_query.to(current_device)
        B = write_query.size(0)
        if B == 0: return # Avoid processing empty batches

        processed_query = self.query_proj(write_query).view(B, self.num_read_heads, self.memory_dim)[:, 0, :]
        query_norm = F.normalize(processed_query, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        attn_scores = torch.matmul(query_norm, mem_norm.t())
        write_weights = torch.softmax(attn_scores / math.sqrt(self.memory_dim), dim=1)

        erase_signal = torch.sigmoid(self.erase_layer(write_query))
        add_signal = torch.tanh(self.add_layer(write_query))
        write_intensity = torch.sigmoid(self.write_gate(write_query))

        # --- Aggregated Write ---
        erase_per_slot = torch.einsum('bs,bd->sd', write_weights, erase_signal)
        add_per_slot = torch.einsum('bs,bd->sd', write_weights, add_signal)
        slot_weights_sum = write_weights.sum(dim=0).unsqueeze(1).clamp(min=1e-6)
        avg_erase = erase_per_slot / slot_weights_sum
        avg_add = add_per_slot / slot_weights_sum
        avg_intensity = torch.einsum('bs,b->s', write_weights, write_intensity.squeeze(-1)) / slot_weights_sum.squeeze(-1)
        avg_intensity = avg_intensity.unsqueeze(1)

        # --- Update memory (non-inplace) ---
        current_memory_detached = self.memory.detach()
        new_memory = current_memory_detached * (1 - avg_intensity * avg_erase) + avg_intensity * avg_add
        with torch.no_grad():
            self.memory.copy_(F.normalize(new_memory, p=2, dim=1))


class MLP(nn.Module):
    """ Simple Multi-Layer Perceptron with LayerNorm and GELU """
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=2, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim, 32) # Ensure hidden_dim is reasonable, min 32
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CNNMapEncoder(nn.Module):
    """ Encodes grid maps using simple strided convolutions. """
    def __init__(self, input_channels, output_dim=64, base_filters=32, num_layers=2, grid_size=32, verbose: bool = True):
        super().__init__()
        layers = []
        current_channels = input_channels
        current_filters = base_filters
        self.total_stride = 1
        for i in range(num_layers):
            stride = 2 if i < 2 else 1 # Downsample twice initially
            layers.append(nn.Conv2d(current_channels, current_filters, kernel_size=3, stride=stride, padding=1))
            num_groups = max(1, min(current_filters // 4, 32))
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=current_filters))
            layers.append(nn.GELU())
            current_channels = current_filters
            if stride == 2:
                # Don't double filters on every downsample to keep the network smaller
                if i == 0:
                    current_filters = min(current_filters * 2, 512)
                self.total_stride *= 2

        # Add a final conv layer to get to the desired output dimension.
        layers.append(nn.Conv2d(current_channels, output_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim

        if verbose:
            print(f"CNNMapEncoder: Input Channels={input_channels}, Output Dim={output_dim}, Total Stride={self.total_stride}")

    def forward(self, x):
        # The forward pass is now just the encoder
        return self.encoder(x)
    

class BeliefMapProcessor(nn.Module):
    """ Processes the agent's local map_obs to extract context features for belief update. """
    def __init__(self, input_channels: int, output_dim: int = 32, map_grid_size: int = 32, base_filters: int = 16, verbose: bool = True):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.map_grid_size = map_grid_size

        layers = []
        current_channels = input_channels
        current_filters = base_filters
        # Example CNN structure: 3 conv layers with decreasing spatial size
        # Layer 1: 32x32 -> 16x16
        layers.append(nn.Conv2d(current_channels, current_filters, kernel_size=3, stride=2, padding=1))
        layers.append(nn.GroupNorm(max(1, current_filters // 4), current_filters))
        layers.append(nn.GELU())
        current_channels = current_filters
        current_filters *= 2 # e.g., 16 -> 32

        # Layer 2: 16x16 -> 8x8
        layers.append(nn.Conv2d(current_channels, current_filters, kernel_size=3, stride=2, padding=1))
        layers.append(nn.GroupNorm(max(1, current_filters // 4), current_filters))
        layers.append(nn.GELU())
        current_channels = current_filters
        current_filters *= 2 # e.g., 32 -> 64

        # Layer 3: 8x8 -> 4x4
        layers.append(nn.Conv2d(current_channels, current_filters, kernel_size=3, stride=2, padding=1))
        layers.append(nn.GroupNorm(max(1, current_filters // 4), current_filters))
        layers.append(nn.GELU())
        current_channels = current_filters

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.map_grid_size, self.map_grid_size)
            flattened_size = self.conv_layers(dummy_input).flatten(1).shape[1]

        self.fc = MLP(flattened_size, output_dim, hidden_dim=max(output_dim * 2, flattened_size // 2, 64))
        if verbose:
            print(f"BeliefMapProcessor: Input {input_channels}x{map_grid_size}x{map_grid_size}, CNN Out Features={flattened_size}, Final Output Dim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H_map, W_map)
        if x.shape[1] != self.input_channels or x.shape[2] != self.map_grid_size or x.shape[3] != self.map_grid_size:
            # This indicates a mismatch between expected and actual map_obs dimensions
            raise ValueError(f"BeliefMapProcessor input shape mismatch. Expected C={self.input_channels}, H/W={self.map_grid_size}. Got {x.shape}")
        x = self.conv_layers(x)
        x = x.flatten(1) # Flatten all dimensions except batch
        x = self.fc(x)
        return x


class PhysicsFeatureExtractor(nn.Module):
    """ Calculates basic physics-based edge features. """
    def __init__(self, output_dim=8, verbose: bool = True):
        super().__init__()
        self.output_dim = output_dim
        # MLP input: dist_norm (1), rel_pos_norm (2), approach_speed_norm (1) = 4
        self.physics_mlp = MLP(4, output_dim, hidden_dim=max(16, output_dim*2), num_layers=2)
        if verbose:
            print(f"PhysicsFeatureExtractor: Output Dim={output_dim}")

    def forward(self, pos_i, pos_j, vel_i, vel_j, max_dist):
        max_dist = max_dist.clamp(min=1e-6) # Clamp the tensor
        rel_pos = pos_j - pos_i
        dist = torch.linalg.norm(rel_pos, dim=1, keepdim=True).clamp(min=1e-6)
        dist_norm = (dist / max_dist).clamp(0.0, 1.0) # Element-wise division
        rel_pos_norm = rel_pos / dist # Unit vector
        rel_vel = vel_j - vel_i
        # Project rel_vel onto the negative direction of rel_pos_norm (i.e., direction from j to i)
        approach_speed_comp = torch.sum(rel_vel * (-rel_pos_norm), dim=1, keepdim=True)
        # Normalize approach speed: assume max closing speed is roughly 2 * agent_max_speed.
        # For simplicity, normalize by a constant, e.g., 2.0 (can be tuned)
        approach_speed_norm = torch.clamp(approach_speed_comp / 2.0, -1.0, 1.0)
        raw_physics = torch.cat([dist_norm, rel_pos_norm, approach_speed_norm], dim=1)
        physics_features = self.physics_mlp(raw_physics)
        return physics_features


class NeighborAttention(nn.Module):
    """ Simplified Scaled Dot-Product Attention over neighbors using torch_scatter. """
    def __init__(self, query_dim, key_dim, value_dim, out_dim, dropout=0.1, verbose: bool = True):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, query_dim, bias=False)
        self.value_proj = nn.Linear(value_dim, value_dim, bias=False)
        self.scale = query_dim ** -0.5
        self.out_proj = nn.Linear(value_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        if verbose:
            print(f"NeighborAttention: Q/K Proj Dim={query_dim}, V Proj Dim={value_dim}, Out Dim={out_dim}")

    def forward(self, query_node: torch.Tensor, key_edge: torch.Tensor, value_edge: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            query_node (Tensor): Query vectors for each node in the graph. Shape: [num_nodes, query_dim].
            key_edge (Tensor): Key vector for each edge. Shape: [num_edges, query_dim].
            value_edge (Tensor): Value vector for each edge. Shape: [num_edges, value_dim].
            edge_index (Tensor): The graph's edge_index.
        """
        num_nodes = query_node.size(0)
        target_nodes_idx = edge_index[1]

        # Project inputs
        q_i = self.query_proj(query_node)
        k_ji = self.key_proj(key_edge)
        v_ji = self.value_proj(value_edge)

        # Expand query to match the number of edges
        q_i_edge = q_i[target_nodes_idx]

        # Calculate attention
        attn_scores = torch.sum(q_i_edge * k_ji, dim=-1) * self.scale
        attn_weights = scatter_softmax_manual(attn_scores, target_nodes_idx, dim=0, dim_size=num_nodes)

        # Apply attention to values and aggregate
        weighted_values = attn_weights.unsqueeze(-1) * v_ji
        aggregated_attended_values = scatter_add(weighted_values, target_nodes_idx, dim=0, dim_size=num_nodes)
        
        # Project output and apply norm
        projected_output = self.out_proj(aggregated_attended_values)
        output = self.norm(self.dropout(projected_output))
        return output, attn_weights


class NeuralLogicNet(nn.Module):
    """ Small MLP to learn modulation factors based on symbolic and belief context. """
    def __init__(self,
                 self_sym_dim: int,
                 neighbor_sym_dim: int,
                 belief_dim: int,
                 output_dim: int,
                 hidden_dim: int = 32,
                 verbose: bool = True):
        super().__init__()
        input_dim = self_sym_dim + neighbor_sym_dim + belief_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=2)
        if verbose:
            print(f"NeuralLogicNet Initialized: Input Dim={input_dim} (SelfSym={self_sym_dim}, NeighSym={neighbor_sym_dim}, Belief={belief_dim}), Output Dim={output_dim}")

    def forward(self, sym_input_i, agg_sym_neighbors_i, belief_state_input):
        inputs_to_cat = []
        if sym_input_i is not None and sym_input_i.shape[-1] > 0: inputs_to_cat.append(sym_input_i)
        if agg_sym_neighbors_i is not None and agg_sym_neighbors_i.shape[-1] > 0: inputs_to_cat.append(agg_sym_neighbors_i)
        if belief_state_input is not None and belief_state_input.shape[-1] > 0: inputs_to_cat.append(belief_state_input)

        if not inputs_to_cat:
            print("Warning: NeuralLogicNet received no valid inputs.")
            output_features = self.mlp.mlp[-1].out_features
            bs = 1
            if sym_input_i is not None: bs = sym_input_i.size(0)
            elif agg_sym_neighbors_i is not None: bs = agg_sym_neighbors_i.size(0)
            elif belief_state_input is not None: bs = belief_state_input.size(0)
            return torch.zeros(bs, output_features, device=self.mlp.mlp[-1].weight.device)

        combined_input = torch.cat(inputs_to_cat, dim=1)
        modulation_params = self.mlp(combined_input)
        return modulation_params


##################################################
# === PINSAN-NCA Network === #####################
##################################################
class PINSAN_Network(nn.Module):
    """
    (V6) PINSAN-NCA Core Network.
    - Uses two separate MLPs for live vs. memory graph edge features.
    - Extracts ego position from the structured `env_graph` for robust perception sampling.
    """
    def __init__(self,
                 self_feature_dim: int, raw_map_channels: int, memory_map_channels: int,
                 action_dim: int, pickup_dim: int, reward_dim: int, num_discrete_roles: int, role_emb_dim: int,
                 map_grid_size: int, map_embed_dim: int, cnn_layers: int, h_dim: int, nca_iterations: int,
                 msg_dim: int, sym_dim: int, sym_emb_dim: int, physics_feat_dim: int,
                 gnn_hidden_dim: int, gnn_layers: int, gnn_heads: int, edge_feature_dim: int,
                 logic_net_hidden_dim: int, modulation_dim: int, modulation_type: str, comm_radius: float,
                 belief_dim: int, map_context_for_belief_dim: int, readout_hidden_dim: int,
                 dynamics_pred_dim: Optional[int],
                 use_decentralized_memory: bool,
                 decentralized_memory_dim: int,
                 query_input_dim: int,
                 map_width: float, map_height: float, dropout_rate: float,
                 role_entropy_coef: float, aux_loss_coef: float,
                 obs_radius: float,
                 mem_connection_radius: float,
                 use_neighbor_attention: bool, use_neural_logic: bool,
                 use_symbolic_layer: bool, use_physics_features: bool,
                 verbose: bool = True):
        super().__init__()
        super().__init__()
        # Store Config
        self.h_dim = h_dim; self.nca_iterations = nca_iterations; self.use_symbolic_layer = use_symbolic_layer
        self.use_physics_features = use_physics_features; self.use_neighbor_attention = use_neighbor_attention
        self.use_neural_logic = use_neural_logic; self.sym_dim = sym_dim; self.sym_emb_dim = sym_emb_dim
        self.physics_feat_dim = physics_feat_dim; self.modulation_type = modulation_type
        self.belief_dim = belief_dim; self.comm_radius = comm_radius
        self.map_context_for_belief_dim = map_context_for_belief_dim
        self.map_width = map_width
        self.map_height = map_height
        self.memory_map_channels = OCC_CH_COUNT # No longer +1
        self.use_decentralized_memory = use_decentralized_memory
        self.decentralized_memory_dim = query_input_dim if self.use_decentralized_memory else 0
        self.obs_radius = obs_radius
        self.mem_connection_radius = mem_connection_radius
        self.map_grid_size = map_grid_size # Store for belief processor

        # --- Perception Modules ---
        self.self_obs_fc = MLP(self_feature_dim, h_dim // 2, hidden_dim=h_dim)
        self.raw_map_encoder = CNNMapEncoder(raw_map_channels, output_dim=map_embed_dim, num_layers=cnn_layers, grid_size=map_grid_size, verbose=verbose)
        memory_map_channels_input = self.memory_map_channels
        self.memory_map_encoder = CNNMapEncoder(memory_map_channels_input, output_dim=map_embed_dim, num_layers=cnn_layers, grid_size=map_grid_size, verbose=verbose)

        self.perception_mlp = MLP((h_dim // 2) + (map_embed_dim * 2), h_dim, hidden_dim=h_dim)

        # --- Symbolic & Physics Feature Modules ---
        if self.use_symbolic_layer:
            self.symbolic_head = MLP(h_dim, sym_dim)
            self.symbolic_embedding = nn.Embedding(sym_dim, sym_emb_dim)
        if self.use_physics_features:
            self.physics_extractor = PhysicsFeatureExtractor(physics_feat_dim, verbose=verbose)

        # --- Communication Modules ---
        self.message_mlp = MLP(h_dim + (sym_emb_dim if use_symbolic_layer else 0) + (physics_feat_dim if use_physics_features else 0), msg_dim)
        if self.use_neighbor_attention:
            self.neighbor_attention = NeighborAttention(h_dim, msg_dim, msg_dim, h_dim, dropout=dropout_rate, verbose=verbose)
        else:
            self.neighbor_attention = None
            self.simple_agg_proj = MLP(msg_dim, h_dim)

        # --- UNIFIED GNN & Edge Feature Modules ---
        # The unified graph has memory features, so we only need one set of modules.
        # The edge MLP input dimension is based on the richer memory graph features.
        unified_edge_raw_dim = 8 + (2 * (MEM_NODE_FEATURE_DIM + 1))
        self.gnn_edge_mlp = MLP(unified_edge_raw_dim, edge_feature_dim, hidden_dim=gnn_hidden_dim)

        # GNN for processing the unified persistent memory graph
        self.gnn_node_encoder = MLP(MEM_NODE_FEATURE_DIM + 1, gnn_hidden_dim) # +1 for cluster count
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=gnn_heads, concat=False, dropout=dropout_rate, edge_dim=edge_feature_dim) 
            for _ in range(gnn_layers)
        ])
        self.gnn_output_dim = gnn_hidden_dim

        # --- Belief State & Logic Modules ---
        if self.belief_dim > 0:
            self.belief_input_feature_dim = 10 + map_context_for_belief_dim
            self.belief_map_processor = BeliefMapProcessor(raw_map_channels, map_context_for_belief_dim, map_grid_size, verbose=verbose)
            self.belief_update_gru = nn.GRUCell(self.belief_input_feature_dim, self.belief_dim)
        if self.use_neural_logic:
            self.neural_logic_net = NeuralLogicNet(sym_dim, sym_dim, self.belief_dim, modulation_dim, logic_net_hidden_dim, verbose=verbose)

        # --- External Memory (NTM) Modules ---
        if self.use_decentralized_memory:
            self.read_query_mlp = MLP(query_input_dim, query_input_dim)
            gating_input_dim = h_dim * 2 + physics_feat_dim
            self.memory_gating = nn.Sequential(MLP(gating_input_dim, 16, hidden_dim=32), nn.Linear(16,1), nn.Sigmoid())

        # --- Core Recurrent Update ---
        gru_input_dim = h_dim + self.gnn_output_dim + self.decentralized_memory_dim
        self.gru_cell = nn.GRUCell(gru_input_dim, h_dim)

        # --- Output Readout Heads ---
        readout_input_dim = h_dim + h_dim
        self.readout_mlp = MLP(readout_input_dim, readout_hidden_dim)
        self.movement_head = nn.Linear(readout_hidden_dim, action_dim * 2)
        self.pickup_head = nn.Linear(readout_hidden_dim, pickup_dim)
        self.role_head_discrete = nn.Linear(readout_hidden_dim, num_discrete_roles)
        self.role_head_continuous = nn.Linear(readout_hidden_dim, role_emb_dim)
        self.reward_weight_head_state_dependent = nn.Sequential(nn.Linear(readout_hidden_dim, readout_hidden_dim//2), nn.GELU(), nn.Linear(readout_hidden_dim//2, reward_dim))
        self.dynamics_head = MLP(readout_hidden_dim, dynamics_pred_dim) if dynamics_pred_dim else None
        
        self.apply(self._init_weights)

    def forward(self,
                personal_ntm_memories: BatchedExternalMemory, # Changed type hint
                self_obs: torch.Tensor, raw_map: torch.Tensor,
                graph: Optional[Data], # This is the unified graph
                memory_map: torch.Tensor,
                h_prev_tuple: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]],
                agent_policy_indices: Optional[torch.Tensor] = None, # Added
                **kwargs):
        N = self_obs.size(0)
        device = self_obs.device

        if N > 0 and torch.randint(0, 20, (1,)).item() == 0: # Print sporadically for one agent in the batch
            print(f"\n--- DEBUG: PINSAN_Network Input ---")
            print(f"  - Batch Size: {N}")
            if graph is not None and hasattr(graph, 'num_nodes'):
                print(f"  - Unified Graph: {graph.num_nodes} nodes, {graph.num_edges} edges.")
            else:
                print(f"  - Unified Graph: None")
            print("-" * 20)

        h_prev, belief_prev = self._prepare_hidden_input_internal(h_prev_tuple, N, device)

        # 1. Perception
        self_emb = self.self_obs_fc(self_obs)
        
        # --- ROBUST GRID SAMPLING ---
        agent_pos_norm = self_obs[:, [NODE_FEATURE_MAP['pos_x_norm'], NODE_FEATURE_MAP['pos_y_norm']]]
        if graph is not None and hasattr(graph, 'x') and graph.num_nodes > 0 and hasattr(graph, 'batch'):
            ego_mask = graph.is_ego if hasattr(graph, 'is_ego') else (graph.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5)
            if ego_mask.any():
                agent_pos_dense, valid_mask = to_dense_batch(
                    # Use base features from the unified graph
                    graph.x[ego_mask][:, [NODE_FEATURE_MAP['pos_x_norm'], NODE_FEATURE_MAP['pos_y_norm']]],
                    graph.batch[ego_mask], batch_size=N
                )
                valid_batch_items_mask = valid_mask[:, 0]
                if valid_batch_items_mask.any():
                    agent_pos_norm[valid_batch_items_mask] = agent_pos_dense[valid_batch_items_mask, 0, :]

        grid_coords = (agent_pos_norm * 2.0 - 1.0).view(N, 1, 1, 2)
        raw_map_spatial_feat = self.raw_map_encoder(raw_map)
        raw_map_feat = F.grid_sample(raw_map_spatial_feat, grid_coords, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).squeeze(-1)
        
        mem_map_spatial_feat = self.memory_map_encoder(memory_map)
        mem_map_feat = F.grid_sample(mem_map_spatial_feat, grid_coords, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).squeeze(-1)
        perception_feat = self.perception_mlp(torch.cat([self_emb, raw_map_feat, mem_map_feat], dim=1))

        # 2. Belief Update
        current_belief_state = self._update_belief_state(self_obs, raw_map, belief_prev, N, device)
        # 3. GNN on Persistent Memory Graph
        gnn_embedding = self._process_memory_graph(graph, N, device) # This now processes the unified graph
        
        # 4. NCA Iterations
        h_i = h_prev
        for _ in range(self.nca_iterations):
            h_in = h_i.detach() # Detach for stability
            sym_probs_i = F.softmax(self.symbolic_head(h_in), dim=1) if self.use_symbolic_layer else torch.zeros(N, self.sym_dim, device=device)
            
            # Extract communication messages from neighbors in the live graph
            att_msg_i, agg_sym_neighbors_i, physics_ji, agent_edge_index, batch_vec_egos = self._extract_communication_edges(h_in, sym_probs_i, graph)
            
            # Perform memory operations (read from own, gated read from neighbors)
            swapped_memory_vec = self._perform_memory_swapping(
                h_in, personal_ntm_memories, physics_ji, agent_edge_index,
                batch_vec_egos, agent_policy_indices, N
            )
            
            # Get modulation factors from the logic net
            mod_factors = self.neural_logic_net(sym_probs_i, agg_sym_neighbors_i, current_belief_state) if self.use_neural_logic and self.neural_logic_net else None
            
            # Core recurrent update
            gru_input = torch.cat([perception_feat, gnn_embedding, swapped_memory_vec], dim=1)
            h_candidate = self.gru_cell(gru_input, h_in)
            
            # Apply modulation
            if mod_factors is not None:
                h_i = h_candidate * torch.sigmoid(mod_factors) if self.modulation_type == 'gate' else h_candidate + mod_factors
            else:
                h_i = h_candidate


        # 5. Readout Heads
        final_h_i = h_i
        readout_input = torch.cat([final_h_i, perception_feat], dim=1)
        readout_feat = self.readout_mlp(readout_input)
        
        mean, log_std = self.movement_head(readout_feat).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        pickup_logits = self.pickup_head(readout_feat)
        
        action_distribution_params = {"movement_mean": mean, "movement_std": log_std.exp(), "pickup_logits": pickup_logits}
        role_discrete_logits = self.role_head_discrete(readout_feat)
        role_continuous_emb = self.role_head_continuous(readout_feat)
        predicted_next = self.dynamics_head(readout_feat) if self.dynamics_head is not None else None
        hybrid_role = {"discrete": role_discrete_logits, "continuous": role_continuous_emb}
        reward_weights_state_dependent = F.softmax(self.reward_weight_head_state_dependent(readout_feat), dim=-1)
        
        final_hidden_state_tuple = (final_h_i, current_belief_state)
        
        # Standardize the return signature to include raw_action and contrastive_embedding
        mean, std = action_distribution_params["movement_mean"], action_distribution_params["movement_std"]
        pickup_logits = action_distribution_params["pickup_logits"]
        movement_sampled = torch.tanh(mean) # Use mean for raw action
        pickup_sampled = torch.argmax(pickup_logits, dim=-1)
        raw_action = torch.cat([movement_sampled, pickup_sampled.unsqueeze(-1).float()], dim=1)
        
        return (action_distribution_params, hybrid_role, final_hidden_state_tuple,
                predicted_next, reward_weights_state_dependent, current_belief_state,
                None, raw_action, None)
    
    def _process_memory_graph(self, graph: Optional[Data], N: int, device: torch.device) -> torch.Tensor:
        """
        (UNIFIED) Processes the batched graph (live fovea + memory periphery) using GNN layers
        and extracts a feature vector for each agent's ego node.
        """
        gnn_embedding = torch.zeros(N, self.gnn_output_dim, device=device)
        if graph is None or not hasattr(graph, 'x') or graph.num_nodes == 0:
            return gnn_embedding

        graph_data = graph.to(device)
        if not hasattr(graph_data, 'batch') or graph_data.batch is None:
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

        node_feat_unified = self.gnn_node_encoder(graph_data.x)

        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None and graph_data.num_edges > 0:
            # Use the larger connection radius suitable for the combined memory graph
            edge_attr_unified = create_graph_edge_features(graph_data, self.gnn_edge_mlp, self.mem_connection_radius)

            for gnn_layer in self.gnn_layers:
                node_feat_unified = F.gelu(gnn_layer(node_feat_unified, graph_data.edge_index, edge_attr=edge_attr_unified))
        
        is_ego_mask = graph_data.is_ego if hasattr(graph_data, 'is_ego') else (graph_data.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5)
        if not is_ego_mask.any():
            return gnn_embedding

        # Extract the final feature vector for each agent's ego node
        ego_features_dense, valid_mask = to_dense_batch(
            node_feat_unified[is_ego_mask], graph_data.batch[is_ego_mask], batch_size=N
        )
        if ego_features_dense.numel() > 0:
            first_ego_features = ego_features_dense[:, 0, :]
            valid_batch_items_mask = valid_mask[:, 0]
            gnn_embedding[valid_batch_items_mask] = first_ego_features[valid_batch_items_mask]
            
        return gnn_embedding
    
    def _extract_communication_edges(self,
                                     h_in: torch.Tensor,
                                     sym_probs_i: torch.Tensor,
                                     unified_graph: Optional[Data]
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        (V2 - CORRECTED INDEXING) Extracts communication messages by finding ego-to-ego
        connections and correctly mapping graph indices to batch indices.
        """
        N = h_in.shape[0]
        device = h_in.device

        # Default return values
        att_msg_i = torch.zeros_like(h_in)
        agg_sym_neighbors_i = torch.zeros_like(sym_probs_i)
        physics_ji = torch.empty((0, self.physics_feat_dim if self.use_physics_features else 0), device=device)
        agent_comm_edge_index = None
        ego_nodes_batch_vec = None

        if unified_graph is None or not hasattr(unified_graph, 'x') or unified_graph.num_nodes < 2:
            return att_msg_i, agg_sym_neighbors_i, physics_ji, agent_comm_edge_index, ego_nodes_batch_vec

        is_ego_mask = unified_graph.x[:, MEM_NODE_FEAT_IDX['is_ego']] > 0.5
        
        if is_ego_mask.sum() < 2:
             return att_msg_i, agg_sym_neighbors_i, physics_ji, agent_comm_edge_index, ego_nodes_batch_vec

        ego_indices_in_unified = torch.where(is_ego_mask)[0]
        ego_nodes_x = unified_graph.x[ego_indices_in_unified, :NODE_FEATURE_DIM] # Use base features
        ego_nodes_pos = unified_graph.pos[ego_indices_in_unified]
        ego_nodes_batch_vec = unified_graph.batch[ego_indices_in_unified]

        # Map from ego_nodes index back to the batch index (0 to N-1)
        # `ego_nodes_batch_vec` already contains the correct batch index for each ego node.
        # We need to map the local indices (0 to num_egos-1) used in the new radius_graph
        # back to the global batch indices they correspond to.
        
        # Build the temporary communication graph using `obs_radius`
        agent_comm_edge_index_local = radius_graph(ego_nodes_pos, r=self.obs_radius, batch=ego_nodes_batch_vec, max_num_neighbors=32)

        if agent_comm_edge_index_local is not None and agent_comm_edge_index_local.numel() > 0:
            row_local, col_local = agent_comm_edge_index_local
            
            # Convert local indices (in the ego-only graph) to global batch indices
            row_global_batch = ego_nodes_batch_vec[row_local]
            col_global_batch = ego_nodes_batch_vec[col_local]
            
            # We now use these global batch indices to correctly access `h_in` and `sym_probs_i`
            msg_inputs_list = [h_in[row_global_batch]]
            
            if self.use_symbolic_layer and self.symbolic_embedding is not None:
                embedded_sym_neighbor = torch.matmul(sym_probs_i[row_global_batch], self.symbolic_embedding.weight)
                msg_inputs_list.append(embedded_sym_neighbor)

            if self.use_physics_features and self.physics_extractor is not None:
                vel_indices = [NODE_FEATURE_MAP['vel_x_norm'], NODE_FEATURE_MAP['vel_y_norm']]
                vel_egos = ego_nodes_x[:, vel_indices]
                physics_ji = self.physics_extractor(
                    pos_i=ego_nodes_pos[col_local], pos_j=ego_nodes_pos[row_local],
                    vel_i=vel_egos[col_local], vel_j=vel_egos[row_local],
                    max_dist=torch.tensor(self.comm_radius, device=device)
                )
                msg_inputs_list.append(physics_ji)
            
            raw_msg_ji = self.message_mlp(torch.cat(msg_inputs_list, dim=1))

            if self.use_neighbor_attention and self.neighbor_attention is not None:
                # The query node is the target of the communication edge
                attended_msg, att_weights_ji = self.neighbor_attention(
                    query_node=h_in, key_edge=raw_msg_ji, value_edge=raw_msg_ji,
                    edge_index=torch.stack([row_global_batch, col_global_batch]) # Pass global indices to attention
                )
                att_msg_i = h_in + attended_msg
                if self.use_symbolic_layer:
                    agg_sym_neighbors_i = scatter_add(att_weights_ji.unsqueeze(-1) * sym_probs_i[row_global_batch], col_global_batch, dim=0, dim_size=N)
            
            elif self.simple_agg_proj is not None:
                mean_msg = scatter(raw_msg_ji, col_global_batch, dim=0, dim_size=N, reduce='mean')
                att_msg_i = h_in + self.simple_agg_proj(mean_msg)
                if self.use_symbolic_layer:
                    agg_sym_neighbors_i = scatter(sym_probs_i[row_global_batch], col_global_batch, dim=0, dim_size=N, reduce='mean')
        
        # Return the original local-indexed graph, as it's used by the DecentralizedCoordinator which operates on the local ego graph
        return att_msg_i, agg_sym_neighbors_i, physics_ji, agent_comm_edge_index_local, ego_nodes_batch_vec

    def _perform_memory_swapping(self,
                                 h_in: torch.Tensor,
                                 batched_ntm: BatchedExternalMemory,
                                 physics_ji: torch.Tensor,
                                 agent_edge_index: Optional[torch.Tensor],
                                 batch_vec_for_egos: Optional[torch.Tensor],
                                 agent_policy_indices: Optional[torch.Tensor],
                                 N: int) -> torch.Tensor:
        """
        (V3 - Fully Batched) Orchestrates memory swapping using the batched memory module.
        """
        if not self.use_decentralized_memory or self.read_query_mlp is None or batched_ntm is None:
            return torch.zeros(N, self.decentralized_memory_dim, device=h_in.device)

        # 1. Every agent generates a query and reads from its OWN memory.
        # This is a single, batched read operation.
        own_read_queries = self.read_query_mlp(h_in)
        own_memory_reads = batched_ntm.read(own_read_queries, agent_policy_indices)

        # 2. If there are neighbors, perform gated reads from them.
        if agent_edge_index is not None and agent_edge_index.numel() > 0 and batch_vec_for_egos is not None:
            row, col = agent_edge_index  # Source (neighbor), Target (ego)

            # Map local ego indices to global batch indices
            row_batch_idx = batch_vec_for_egos[row]
            col_batch_idx = batch_vec_for_egos[col]
            
            # Map from local ego-graph indices to global policy indices
            neighbor_policy_indices = agent_policy_indices[row_batch_idx]

            # Batched calculation of gating values
            gating_input = torch.cat([h_in[col_batch_idx], h_in[row_batch_idx], physics_ji], dim=1)
            gating_values = self.memory_gating(gating_input)

            # Batched read from neighbor memories
            queries_for_neighbor_mem = own_read_queries[col_batch_idx]
            neighbor_reads = batched_ntm.read(queries_for_neighbor_mem, neighbor_policy_indices)
            
            gated_neighbor_reads = gating_values * neighbor_reads
            
            # Aggregate results back to the original batch agents
            social_memory_agg = scatter(gated_neighbor_reads, col_batch_idx, dim=0, dim_size=N, reduce='sum')

            return own_memory_reads + social_memory_agg
        else:
            return own_memory_reads

    def _init_weights(self, module):
        if isinstance(module, nn.LazyLinear):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if hasattr(module, 'weight') and module.weight is not None: nn.init.constant_(module.weight, 1.0)
            if hasattr(module, 'bias') and module.bias is not None: nn.init.constant_(module.bias, 0.0)

    def init_hidden(self, batch_size=1):
        current_device = next(self.parameters()).device
        initial_nca_h = torch.zeros(batch_size, self.h_dim, device=current_device)
        initial_belief = torch.zeros(batch_size, self.belief_dim, device=current_device) if self.belief_dim > 0 else None
        return (initial_nca_h, initial_belief)

    def _prepare_hidden_input_internal(self, h_prev_tuple, N, device):
        if h_prev_tuple is None or not isinstance(h_prev_tuple, tuple) or len(h_prev_tuple) != 2: return self.init_hidden(N)
        h_prev, belief_prev = h_prev_tuple
        if h_prev is None or h_prev.shape[0] != N: h_prev = torch.zeros(N, self.h_dim, device=device)
        if self.belief_dim > 0:
            if belief_prev is None or belief_prev.shape[0] != N: belief_prev = torch.zeros(N, self.belief_dim, device=device)
        else: belief_prev = None
        return h_prev, belief_prev

    def _update_belief_state(self, self_obs, raw_map, belief_prev, N, device):
        if self.belief_update_gru is None: return belief_prev

        # Use the SELF_OBS_MAP to access features by name, making it robust to changes.
        s = SELF_OBS_MAP
        required_keys = ['energy_norm', 'health_norm', 'vel_x_norm', 'vel_y_norm', 'is_carrying', 'strength_norm', 'radius_norm', 'rel_hive_x_norm', 'rel_hive_y_norm']
        if all(k in s for k in required_keys):
            # Safe to access all keys
            feats = [
                self_obs[:, s['energy_norm']],
                self_obs[:, s['health_norm']],
                self_obs[:, s['vel_x_norm']],
                self_obs[:, s['vel_y_norm']],
                self_obs[:, s['is_carrying']],
                self_obs[:, s['strength_norm']],
                self_obs[:, s['radius_norm']],
                self_obs[:, s['rel_hive_x_norm']],
                self_obs[:, s['rel_hive_y_norm']],
                self_obs[:, s['hive_dist_norm']]
            ]
            self_obs_derived_features = torch.stack(feats, dim=1)
        else:
            # Fallback for safety if the map is incomplete
            print("Warning: SELF_OBS_MAP is missing required keys for belief update. Using zeros.")
            self_obs_derived_features = torch.zeros(N, 10, device=device)

        map_context_features = torch.zeros(N, self.map_context_for_belief_dim, device=device)
        if self.belief_map_processor is not None:
            map_context_features = self.belief_map_processor(raw_map)

        belief_input_features = torch.cat([self_obs_derived_features, map_context_features], dim=1)

        if belief_input_features.shape[1] != self.belief_input_feature_dim:
            diff = self.belief_input_feature_dim - belief_input_features.shape[1]
            if diff > 0: belief_input_features = F.pad(belief_input_features, (0, diff))
            else: belief_input_features = belief_input_features[:, :self.belief_input_feature_dim]

        return self.belief_update_gru(belief_input_features, belief_prev) if belief_prev is not None else belief_prev


class NCA_PINSANPolicy(nn.Module):
    def __init__(self, num_agents_on_team: int,
                 action_dim=2, pickup_dim=3, self_feature_dim=22, grid_size=32, map_channels=RAW_CH_COUNT,
                 memory_map_channels=OCC_CH_COUNT,
                 # Parameters for the internal PINSAN_Network
                 h_dim=64, nca_iterations=2, msg_dim=32,
                 map_embed_dim=32, cnn_layers=2, 
                 sym_dim=4, sym_emb_dim=8, physics_feat_dim=16,
                 gnn_hidden_dim=48, gnn_layers=2, gnn_heads=2, edge_feature_dim=16,
                 logic_net_hidden_dim=24, modulation_dim=64, modulation_type='bias',
                 belief_dim=10, map_context_for_belief_dim=16,
                 readout_hidden_dim=128,
                 # Parameters for this wrapper class
                 num_discrete_roles=4,
                 role_emb_dim=16,
                 num_reward_components=18,
                 use_decentralized_memory: bool = True,
                 decentralized_memory_slots: int = 16,
                 decentralized_memory_dim: int = 32,
                 query_input_dim: int = 64,
                 use_dynamics_prediction: bool = True,
                 use_contrastive: bool = True,
                 obs_radius: float = 50.0,
                 mem_connection_radius: float = 100.0,
                 policy_lr=1e-3,
                 max_grad_norm=1.0,
                 aux_loss_coef=0.1,
                 role_entropy_coef=0.01,
                 contrastive_loss_coef=0.1,
                 contrastive_tau=0.07,
                 dropout_rate=0.0,
                 verbose: bool = True,
                 **kwargs):
        super().__init__()
        # --- Store self attributes ---
        self.num_reward_components = num_reward_components
        self.use_decentralized_memory = use_decentralized_memory
        self.use_dynamics_prediction = use_dynamics_prediction
        self.use_contrastive = use_contrastive
        self.contrastive_loss_coef = contrastive_loss_coef
        self.contrastive_tau = contrastive_tau
        self.aux_loss_coef = aux_loss_coef
        self.role_entropy_coef = role_entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # --- Instantiate the Core Network ---
        self.network = PINSAN_Network(
            self_feature_dim=self_feature_dim, raw_map_channels=map_channels,
            memory_map_channels=memory_map_channels, action_dim=action_dim, pickup_dim=pickup_dim,
            reward_dim=num_reward_components, num_discrete_roles=num_discrete_roles, role_emb_dim=role_emb_dim,
            map_grid_size=grid_size, map_embed_dim=map_embed_dim, cnn_layers=cnn_layers, h_dim=h_dim,
            nca_iterations=nca_iterations, msg_dim=msg_dim, sym_dim=sym_dim, sym_emb_dim=sym_emb_dim,
            physics_feat_dim=physics_feat_dim, gnn_hidden_dim=gnn_hidden_dim, gnn_layers=gnn_layers,
            gnn_heads=gnn_heads, edge_feature_dim=edge_feature_dim, logic_net_hidden_dim=logic_net_hidden_dim,
            modulation_dim=modulation_dim, modulation_type=modulation_type, comm_radius=obs_radius,
            belief_dim=belief_dim, map_context_for_belief_dim=map_context_for_belief_dim,
            readout_hidden_dim=readout_hidden_dim,
            dynamics_pred_dim=self_feature_dim if use_dynamics_prediction else None,
            use_decentralized_memory=use_decentralized_memory,
            decentralized_memory_dim=decentralized_memory_dim, query_input_dim=query_input_dim,
            map_width=kwargs.get('map_width', 1000.0), map_height=kwargs.get('map_height', 1000.0),
            dropout_rate=dropout_rate, role_entropy_coef=role_entropy_coef, aux_loss_coef=aux_loss_coef,
            obs_radius=obs_radius, mem_connection_radius=mem_connection_radius,
            use_neighbor_attention=kwargs.get('use_neighbor_attention', True),
            use_neural_logic=kwargs.get('use_neural_logic', True),
            use_symbolic_layer=kwargs.get('use_symbolic_layer', True),
            use_physics_features=kwargs.get('use_physics_features', True),
            verbose=verbose
        )

        # --- Internal, Stateful Memory Management ---
        if self.use_decentralized_memory:
            if verbose:
                print("Info: Initializing internal NTMs for NCA_PINSANPolicy.")
            self.personal_ntms = BatchedExternalMemory(
                num_agents=num_agents_on_team,
                num_slots=decentralized_memory_slots,
                memory_dim=decentralized_memory_dim,
                query_dim=query_input_dim
            )
        else:
            self.personal_ntms = None
            
        # Initialize the momentum encoder for contrastive learning if used
        self.key_contrastive_proj_head = None
        if self.use_contrastive and hasattr(self.network, 'contrastive_proj_head') and self.network.contrastive_proj_head is not None:
            if verbose:
                print("Info: Initializing Momentum Key Encoder for Contrastive Learning.")
            self.key_contrastive_proj_head = copy.deepcopy(self.network.contrastive_proj_head)
            for param in self.key_contrastive_proj_head.parameters():
                param.requires_grad = False

    def evaluate_actions(self,
                         obs_batch: Dict[str, Union[torch.Tensor, Data]],
                         hidden_state: Optional[Tuple] = None,
                         agent_policy_indices: Optional[torch.Tensor] = None,
                         deterministic_role: bool = False
                         ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict, Dict, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        final_device = next(self.network.parameters()).device
        N = obs_batch['self'].shape[0]
        h_in_tuple = self._prepare_hidden_input(hidden_state, N)

        # The batched memory module is passed directly. Agent indexing is handled inside its methods.
        ntms_for_batch = self.personal_ntms

        # Call the core network, passing the selected NTMs for this batch
        (action_dist_params, hybrid_role_batch, h_next_tuple,
         predicted_next, reward_weights, belief_state_pred,
         contrastive_embedding, raw_action, _) = self.network(
            personal_ntm_memories=ntms_for_batch, # This is now the single batched module
            self_obs=obs_batch['self'].to(final_device),
            raw_map=obs_batch['map'].to(final_device),
            graph=obs_batch.get('graph'), # This is the new unified graph
            memory_map=obs_batch.get('memory_map'),
            h_prev_tuple=h_in_tuple,
            agent_policy_indices=agent_policy_indices, # Pass indices to the network
            return_contrastive_embedding=self.use_contrastive
        )

        # Process outputs for SAC
        mean, std, pickup_logits = action_dist_params["movement_mean"], action_dist_params["movement_std"], action_dist_params["pickup_logits"]
        movement_dist = D.Normal(mean, std)
        pickup_dist = D.Categorical(logits=pickup_logits)

        movement_sampled_pretanh = movement_dist.rsample()
        movement_sampled = torch.tanh(movement_sampled_pretanh)
        pickup_sampled = pickup_dist.sample()

        log_pi_movement = (movement_dist.log_prob(movement_sampled_pretanh) - torch.log((1 - movement_sampled.pow(2)).clamp(min=epsilon))).sum(dim=-1, keepdim=True)
        log_pi_pickup = pickup_dist.log_prob(pickup_sampled).unsqueeze(-1)
        joint_log_pi = (log_pi_movement + log_pi_pickup)
        
        # Assemble final outputs into standardized format
        action_dict = {"movement": movement_sampled, "pickup": pickup_sampled, "joint_action": raw_action}
        next_hidden = {'nca_h': h_next_tuple[0], 'belief': h_next_tuple[1]} if h_next_tuple is not None else None
        
        role_info = {'continuous': None, 'discrete': None, 'discrete_idx': None}
        if isinstance(hybrid_role_batch, dict):
            role_logits, role_continuous = hybrid_role_batch.get('discrete'), hybrid_role_batch.get('continuous')
            role_info.update({'continuous': role_continuous, 'discrete': role_logits})
            if role_logits is not None:
                if deterministic_role:
                    role_discrete_idx = torch.argmax(role_logits, dim=1)
                else:
                    role_dist = D.Categorical(logits=role_logits)
                    role_discrete_idx = role_dist.sample()
                role_info['discrete_idx'] = role_discrete_idx.long()
        elif isinstance(hybrid_role_batch, torch.Tensor):
            role_info['continuous'] = hybrid_role_batch
                    
        return (action_dict,           # 1. Action dictionary
                joint_log_pi,          # 2. Log probability of the action
                next_hidden,           # 3. Next hidden state
                role_info,             # 4. Role information
                reward_weights,        # 5. Predicted reward weights
                predicted_next,        # 6. Predicted next obs
                raw_action,            # 7. Raw action tensor
                contrastive_embedding, # 8. Contrastive embedding
                belief_state_pred)     # 9. Belief state

    def train(self, mode: bool = True):
        super().train(mode); return self
    def eval(self):
        super().eval(); return self

    def init_hidden(self, batch_size=1) -> Dict:
        """Standardized: Returns a dictionary of initial hidden states."""
        # The internal network still returns a tuple, but we wrap it in a dictionary.
        h_tensor, belief_tensor = self.network.init_hidden(batch_size)
        return {'nca_h': h_tensor, 'belief': belief_tensor}
    
    def _prepare_hidden_input(self, hidden_state_dict: Optional[Dict], expected_num_nodes: int) -> Tuple:
        """ Prepares the internal tuple format from the standardized input dictionary. """
        current_device = next(self.network.parameters()).device
        if hidden_state_dict is None or not isinstance(hidden_state_dict, dict):
            return self.network.init_hidden(expected_num_nodes)

        h_prev = hidden_state_dict.get('nca_h')
        belief_prev = hidden_state_dict.get('belief')
        return self.network._prepare_hidden_input_internal((h_prev, belief_prev), expected_num_nodes, current_device)
    
    @torch.no_grad()
    def act(self,
            obs: Dict[str, Union[np.ndarray, torch.Tensor, Data]],
            hidden_state: Optional[Dict] = None,
            personal_ntm_memory: Optional[ExternalMemory] = None,
            noise_scale: float = 0.0
            ) -> Tuple[Dict[str, Union[np.ndarray, int]], Dict, Dict, Dict]:
        """
        (V5 Standardized) Generates a single action for one agent.
        Returns a standardized 4-tuple: (action_dict, next_hidden_state, role_info, aux_outputs)
        """
        self.eval()
        current_device = next(self.network.parameters()).device

        # Batch the single observation dictionary
        obs_batch = {}
        for key, val in obs.items():
            if isinstance(val, Data):
                obs_batch[key] = Batch.from_data_list([val])
            elif isinstance(val, (np.ndarray, torch.Tensor)):
                tensor_val = torch.as_tensor(val, dtype=torch.float32)
                obs_batch[key] = tensor_val.unsqueeze(0)
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=1)
        mem_list = [personal_ntm_memory] if personal_ntm_memory is not None else []
        # Use the standardized act_batch for the core logic
        action_batch_dict, next_hidden_batch, role_batch_dict, aux_outputs_batch = self.act_batch(
            obs_batch=obs_batch,
            hidden_state=hidden_state,
            personal_ntm_memories=mem_list,
            noise_scale=noise_scale
        )

        # Un-batch results for a single agent
        movement_np = action_batch_dict['movement'][0].cpu().numpy()
        pickup_val = action_batch_dict['pickup'][0].item()
        action_env_dict = {"movement": movement_np, "pickup": int(np.clip(round(pickup_val), 0, 2))}       
        # Detach and convert to numpy where applicable
        next_hidden_single = {k: v[0].cpu().numpy() if hasattr(v, 'cpu') and v is not None else v for k, v in next_hidden_batch.items()}
        role_info_single = {k: v[0].cpu().numpy() if hasattr(v, 'cpu') and v is not None and v.numel() > 0 else v for k, v in role_batch_dict.items()}
        aux_outputs_single = {k: v[0].cpu().numpy() if hasattr(v, 'cpu') and v is not None and v.numel() > 0 else v for k, v in aux_outputs_batch.items()}

        return action_env_dict, next_hidden_single, role_info_single, aux_outputs_single

    def act_batch(self,
                  obs_batch: Dict[str, torch.Tensor],
                  hidden_state: Optional[Dict] = None,
                  agent_policy_indices: Optional[torch.Tensor] = None, 
                  noise_scale: float = 0.0,
                  deterministic_role: bool = False,
                  **kwargs # Absorb unused kwargs
                  ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict, Dict]:
        """
        (V6 Standardized) Selects actions for a batch of agents.
        Returns a standardized 4-tuple: (action_dict, next_hidden, role_info, aux_outputs)
        """
        self.network.eval()

        # 1. Evaluate actions using the full observation context
        with torch.no_grad():
            output_tuple = self.evaluate_actions(
                obs_batch=obs_batch,
                hidden_state=hidden_state,
                agent_policy_indices=agent_policy_indices,
                deterministic_role=deterministic_role
            )
            sampled_action_dict, _, next_hidden_state_dict, role_info, predicted_next, _, _, contrastive_embedding, belief_state_pred = output_tuple

        # 2. Perform the NTM write operation (NOW BATCHED)
        if self.use_decentralized_memory and self.personal_ntms is not None and agent_policy_indices is not None and agent_policy_indices.numel() > 0:
            h_final_nca = next_hidden_state_dict.get('nca_h')
            if h_final_nca is not None and h_final_nca.shape[0] > 0:
                with torch.no_grad():
                    # Generate all write queries in one batch
                    write_queries = self.network.read_query_mlp(h_final_nca)
                    # Perform a single, batched write operation
                    self.personal_ntms.write(write_queries, agent_policy_indices)

        # 3. Add noise for exploration
        if noise_scale > 0.0:
            noise = torch.randn_like(sampled_action_dict['movement']) * noise_scale
            sampled_action_dict['movement'] = torch.clamp(sampled_action_dict['movement'] + noise, -1.0, 1.0)
            sampled_action_dict['joint_action'] = torch.cat([
                sampled_action_dict['movement'], 
                sampled_action_dict['pickup'].unsqueeze(-1).float()
            ], dim=1)
        
        # 4. Prepare detached outputs
        action_output_dict_final = {k: v.detach() for k, v in sampled_action_dict.items()}
        role_info_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in role_info.items()}
        
        next_hidden_detached = {k: v.detach() if v is not None else None for k, v in next_hidden_state_dict.items()}
        
        aux_outputs = {
            'predicted_next_obs': predicted_next.detach() if predicted_next is not None else None,
            'belief_state': belief_state_pred.detach() if belief_state_pred is not None else None
        }
        
        return action_output_dict_final, next_hidden_detached, role_info_detached, aux_outputs

    def compute_role_entropy_loss(self, role_logits):
        if role_logits is None: return torch.tensor(0.0, device=next(self.parameters()).device)
        role_probs = F.softmax(role_logits, dim=1)
        entropy = -torch.sum(role_probs * torch.log(role_probs.clamp(min=1e-10)), dim=1)
        return -abs(self.role_entropy_coef) * torch.mean(entropy)

    def compute_imagination_loss(self, predicted_next_obs, target_next_obs):
        if not self.use_dynamics_prediction or predicted_next_obs is None or target_next_obs is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.aux_loss_coef * F.mse_loss(predicted_next_obs, target_next_obs.to(predicted_next_obs.device))

    def clip_gradients(self):
        if self.max_grad_norm > 0:
            params_to_clip = [p for p in self.parameters() if p.grad is not None]
            if params_to_clip: torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)

    def to(self, device):
        super().to(device)
        self.network.to(device)
        if self.use_decentralized_memory and self.personal_ntms is not None:
            self.personal_ntms.to(device)
        if self.use_contrastive and self.key_contrastive_proj_head is not None:
            self.key_contrastive_proj_head.to(device)
        return self




#####################################
### TEST LOOP #######################
#####################################
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Testing Refined PINSANPolicy with Correct Batched Communication")
    print("="*60)
    print(f"Using device: {device}")

    # --- Test Parameters (Unchanged) ---
    B = 4; SELF_DIM = 24; RAW_MAP_CH = RAW_CH_COUNT; MEM_MAP_CH = OCC_CH_COUNT
    MAP_GRID = 32; ACTION_DIM = 2; PICKUP_DIM = 3; REWARD_DIM = 18
    N_DISCRETE_ROLES = 4; ROLE_EMB_DIM = 16; WORLD_WIDTH = 1000.0; WORLD_HEIGHT = 1000.0
    MAP_EMBED_DIM = 32; CNN_LAYERS = 2; H_DIM = 64; NCA_ITER = 2; MSG_DIM = 32
    SYM_DIM = N_DISCRETE_ROLES; SYM_EMB_DIM = 8; PHYSICS_FEAT_DIM = 16
    LOGIC_NET_HIDDEN = 24; BELIEF_DIM_TEST = 10; MAP_CONTEXT_FOR_BELIEF_DIM_TEST = 16
    MODULATION_DIM = H_DIM; MODULATION_TYPE = 'bias'; READOUT_HIDDEN = 128
    DYNAMICS_PRED_DIM = SELF_DIM; COMM_RADIUS = 150.0
    GNN_HIDDEN_DIM = 48; GNN_LAYERS = 2; GNN_HEADS = 2; EDGE_FEATURE_DIM = 16
    DECENTRALIZED_MEM_DIM = 32; QUERY_INPUT_DIM = H_DIM; DECENTRALIZED_MEM_SLOTS = 16

    # --- 1. Instantiation ---
    print("\n--- 1. Instantiating Policy ---")
    try:
        policy = NCA_PINSANPolicy(
            num_agents_on_team=20, # Added dummy value for testing
            self_feature_dim=SELF_DIM, map_channels=RAW_MAP_CH, memory_map_channels=MEM_MAP_CH,
            action_dim=ACTION_DIM, pickup_dim=PICKUP_DIM, reward_dim=REWARD_DIM,
            num_discrete_roles=N_DISCRETE_ROLES, role_emb_dim=ROLE_EMB_DIM,
            map_grid_size=MAP_GRID, map_embed_dim=MAP_EMBED_DIM, cnn_layers=CNN_LAYERS,
            h_dim=H_DIM, nca_iterations=NCA_ITER, comm_radius=COMM_RADIUS, msg_dim=MSG_DIM,
            sym_dim=SYM_DIM, sym_emb_dim=SYM_EMB_DIM, physics_feat_dim=PHYSICS_FEAT_DIM,
            gnn_hidden_dim=GNN_HIDDEN_DIM, gnn_layers=GNN_LAYERS, gnn_heads=GNN_HEADS,
            edge_feature_dim=EDGE_FEATURE_DIM,
            logic_net_hidden_dim=LOGIC_NET_HIDDEN, modulation_dim=MODULATION_DIM, modulation_type=MODULATION_TYPE,
            belief_dim=BELIEF_DIM_TEST, map_context_for_belief_dim=MAP_CONTEXT_FOR_BELIEF_DIM_TEST,
            readout_hidden_dim=READOUT_HIDDEN, dynamics_pred_dim=DYNAMICS_PRED_DIM,
            decentralized_memory_slots=DECENTRALIZED_MEM_SLOTS,
            decentralized_memory_dim=DECENTRALIZED_MEM_DIM,
            query_input_dim=QUERY_INPUT_DIM,
            map_width=WORLD_WIDTH, map_height=WORLD_HEIGHT,
            dropout_rate=0.0, role_entropy_coef=0.01, aux_loss_coef=0.1,
            obs_radius=COMM_RADIUS,
            use_neighbor_attention=True, use_neural_logic=True, use_symbolic_layer=True, use_physics_features=True
        )
        print("  Policy instantiated successfully.")
    except Exception as e:
        print(f"!!!!!! Policy Init Failed: {e}"); traceback.print_exc(); sys.exit(1)
    
    # --- 2. Generating Realistic Dummy Batched Observation ---
    print("\n--- 2. Generating Realistic Dummy Batched Observation & Memories ---")
    env_graphs_list = []
    mem_graphs_list = []
    for i in range(B):
        # Env graph
        num_nodes_in_graph = random.randint(5, 10)
        x_single = torch.zeros(num_nodes_in_graph, MEM_NODE_FEATURE_DIM + 1)
        x_single_raw = torch.randn(num_nodes_in_graph, NODE_FEATURE_DIM)
        x_single[:, :NODE_FEATURE_DIM] = x_single_raw

        x_single[:, NODE_FEATURE_MAP['team_id']] = torch.randint(0, 2, (num_nodes_in_graph,))
        x_single[:, NODE_FEATURE_MAP['node_type_encoded']] = NODE_TYPE['agent']
        x_single[:, NODE_FEATURE_MAP['is_ego']] = 0.0
        x_single[0, NODE_FEATURE_MAP['is_ego']] = 1.0
        x_single[0, NODE_FEATURE_MAP['team_id']] = 0
        pos_single = torch.rand(num_nodes_in_graph, 2) * WORLD_WIDTH
        env_graphs_list.append(Data(x=x_single, pos=pos_single, radii=torch.rand(num_nodes_in_graph) * 5))
        
        # Memory graph
        num_mem_nodes = random.randint(15, 25)
        mem_pos = torch.rand(num_mem_nodes, 2) * 1000
        mem_edge_index = radius_graph(mem_pos, r=250, max_num_neighbors=16)
        mem_graphs_list.append(Data(
            x=torch.randn(num_mem_nodes, MEM_NODE_FEATURE_DIM + 1),
            pos=mem_pos,
            radii=torch.rand(num_mem_nodes)*5,
            is_ego=torch.tensor([True]+[False]*(num_mem_nodes-1)),
            edge_index=mem_edge_index
        ))

    batched_env_graph = Batch.from_data_list(env_graphs_list).to(device)
    batched_mem_graph = Batch.from_data_list(mem_graphs_list).to(device)

    obs_batch = {
        'self': torch.randn(B, SELF_DIM, device=device),
        'map': torch.randn(B, RAW_MAP_CH, MAP_GRID, MAP_GRID, device=device),
        'graph': batched_env_graph, # Keep this for legacy compatibility if something needs it
        'memory_map': torch.randn(B, MEM_MAP_CH, MAP_GRID, MAP_GRID, device=device),
        'memory_graph': batched_mem_graph
    }
    
    personal_ntms = [ExternalMemory(memory_dim=DECENTRALIZED_MEM_DIM, query_dim=H_DIM).to(device) for _ in range(B)]
    print(f"  Dummy observation batch created. `env_graph` contains {B} distinct graphs.")
    
    # --- 3. Executing `evaluate_actions` to test the full pipeline (Unchanged) ---
    print("\n--- 3. Executing `evaluate_actions` and Verifying Outputs ---")
    try:
        policy.train()
        initial_hidden_state = policy.init_hidden(B)
        # We need to create dummy agent indices for the call
        dummy_agent_indices = torch.arange(B, device=device)

        output_tuple = policy.evaluate_actions(
            obs_batch=obs_batch,
            hidden_state=initial_hidden_state,
            agent_policy_indices=dummy_agent_indices # Pass the dummy indices
        )
        print("  `evaluate_actions` executed successfully without errors.")

        # --- 4. Verifying Output Shapes (CORRECTED) ---
        print("\n--- 4. Verifying Output Shapes ---")
        # Unpack the standardized 9-tuple
        action_dict, log_pi, h_next_dict, role_info, reward_weights, _, joint_action, _, belief_state = output_tuple

        assert joint_action.shape == (B, ACTION_DIM + 1), "Joint Action shape incorrect."
        print(f"  - Joint Action shape OK: {joint_action.shape}")
        
        assert log_pi.shape == (B, 1), f"Log Pi shape incorrect. Expected ({B}, 1), got {log_pi.shape}"
        print(f"  - Log Pi shape OK: {log_pi.shape}")

        # <<< CORRECTED ASSERTION: Check the dictionary and the tensor inside it >>>
        assert isinstance(h_next_dict, dict) and h_next_dict['nca_h'].shape == (B, H_DIM), "Hidden state shape incorrect."
        print(f"  - Next Hidden State structure and shapes OK.")

    except Exception as e:
        print(f"!!!!!! Forward Pass or Verification Failed: {e}"); traceback.print_exc()

    print("\n" + "="*60)
    print("  NCA Policy Refinement Test Completed Successfully.")
    print("="*60 + "\n")
    

