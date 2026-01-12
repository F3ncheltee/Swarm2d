# maac_attention.py
#!/usr/bin/env python3


import sys, os, time, random, copy
import numpy as np
import gc # Garbage collector
from collections import deque # Added missing import
# --- Imports ---
import scipy.spatial # For KDTree
import cProfile # For profiling
import pstats # For reading profile stats
import io # For reading profile stats
import traceback

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import random as random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import traceback
import torch.distributions as D
from typing import Optional, Dict, Tuple, List, Union
import traceback # <<< ADD THIS IMPORT
from torch_geometric.data import Data, Batch

# Helper function that might have been in `otherhelpers`
def get_data_list_from_graph(graph_batch: Optional[Batch]) -> List[Data]:
    if graph_batch is None:
        return []
    if isinstance(graph_batch, Data):
        return [graph_batch]
    return graph_batch.to_data_list()

from env.env import Swarm2DEnv
from constants import (
    WIDTH, HEIGHT, FPS, OBS_RADIUS, COMM_RADIUS, AGENT_RADIUS, MAX_STEPS,
    HIVE_MAX_HEALTH, AGENT_MAX_HEALTH, AGENT_MAX_ENERGY, RESOURCE_MAX_SIZE,
    BEE_SPEED, MEM_NODE_FEATURE_DIM, MEM_NODE_FEAT_IDX, NODE_FEATURE_DIM,
    NODE_FEATURE_MAP, NODE_TYPE, NUM_NODE_TYPES, RAW_CH_COUNT, RAW_CH,
    REWARD_COMPONENT_KEYS, CLUSTER_CELL_SIZE, NODE_FEATURE_MAP, 
    MEM_NODE_FEATURE_DIM, MEM_NODE_FEAT_IDX, NODE_FEATURE_DIM, 
    OCC_CH, RAW_CH, OCC_CH_COUNT, RAW_CH_COUNT, RAW_CH_IDX_TO_NAME, GLOBAL_CUE_DIM
)
from env.observations import create_graph_edge_features, generate_foveated_graph, OccMapObservationManager

import torch.distributions as D
import math

# SAC specific constants
LOG_STD_MAX = 2
LOG_STD_MIN = -5 # Often -20, but -5 can be more stable
epsilon = 1e-6

try:
    import torch_geometric
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    radius_graph = pyg_nn.radius_graph
    from torch_scatter import scatter_mean

except ImportError:
    print("Error: PyTorch Geometric core components not found.")
    print("Please install it: pip install torch-geometric ...")
    # Fallback...
    class Batch: pass
    pyg_nn = nn
    # Define fallback radius_graph that returns None
    def radius_graph(*args, **kwargs):
        # print("Warning: PyG not found, using fallback radius_graph (returns None).") # Optional warning
        return None
    # Ensure GATv2Conv fallback if needed within the except block
    if not hasattr(pyg_nn, 'GATv2Conv'):
        pyg_nn.GATv2Conv = nn.Identity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAAC_ROLES_GLOBAL = ["scout", "collector", "defender", "attacker"]

# --- Constants ---
GRAPH_PROX_RADIUS = COMM_RADIUS # Proximity for GNN communication graph

# --- Helper Modules (Unchanged) ---
class TitanTransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder layer with retrieval-augmented attention (Titan).
    The layer can attend to an external, growing episodic memory cache.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Standard Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # LayerNorms and Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, episodic_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            src (Tensor): The sequence from the current call (Batch, SeqLen, Dim).
            episodic_memory (Tuple[Tensor, Tensor]): A tuple of (keys, values) from previous steps.
        """
        query = src 
        # Combine current sequence with episodic memory for attention
        if episodic_memory is not None and episodic_memory[0].numel() > 0:
            mem_k, mem_v = episodic_memory
            # Keys and values are a mix of current sequence and past memories
            key = torch.cat([mem_k, src], dim=1)
            value = torch.cat([mem_v, src], dim=1)
        else:
            # First step of an episode, no memory yet
            key, value = src, src
        # Standard multi-head attention over the combined context
        attn_output, _ = self.self_attn(query, key, value)
        # Standard transformer block operations (residual connections, layer norm)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=2, dropout=0.0):
        super().__init__()
        if hidden_dim is None: hidden_dim = max(input_dim, output_dim, 32)
        layers = []; in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim)); layers.append(nn.LayerNorm(hidden_dim)); layers.append(nn.GELU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim)); self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class CNNMapEncoder(nn.Module):
    """ Standard CNN for processing grid-based observations. """
    def __init__(self, input_channels, output_dim=64, base_filters=32, num_layers=2):
        super().__init__()
        layers = []; current_channels = input_channels; current_filters = base_filters
        for i in range(num_layers):
            stride = 2 if i < 2 else 1
            out_filters = current_filters * 2 if stride == 2 else current_filters
            out_filters = min(out_filters, 512)
            layers.append(nn.Conv2d(current_channels, out_filters, 3, stride=stride, padding=1))
            layers.append(nn.GroupNorm(max(1, out_filters // 8), out_filters)); layers.append(nn.GELU())
            current_channels = out_filters
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        # Use a lazy linear layer to avoid manual calculation errors
        layers.append(nn.LazyLinear(output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # One-time initialization for LazyLinear
        if isinstance(self.encoder[-1], nn.LazyLinear):
            with torch.no_grad():
                self.encoder(x)
        return self.encoder(x)

class MultiScaleCandidateExtractor(nn.Module):
    """ Extracts candidate tokens from the map using multiple convolutional scales. """
    def __init__(self, raw_map_channels, candidate_out_dim=8, num_tokens=8, grid_size=32):
        super().__init__()
        self.num_tokens = num_tokens
        mid_channels = 32
        norm_dims = [grid_size, grid_size]
        ch1, ch3, ch5 = 10, 12, 10 # Ensure they sum to mid_channels

        self.branch1 = nn.Sequential(nn.Conv2d(raw_map_channels, ch1, 1, padding=0), nn.LayerNorm([ch1] + norm_dims), nn.GELU())
        self.branch3 = nn.Sequential(nn.Conv2d(raw_map_channels, ch3, 3, padding=1), nn.LayerNorm([ch3] + norm_dims), nn.GELU())
        self.branch5 = nn.Sequential(nn.Conv2d(raw_map_channels, ch5, 5, padding=2), nn.LayerNorm([ch5] + norm_dims), nn.GELU())
        
        self.conv_reduce = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 1), nn.GELU())
        self.adaptive_pool = nn.AdaptiveAvgPool2d((num_tokens, 1))
        
        # Use LazyLinear to avoid manual calculation errors
        self.fc = nn.LazyLinear(candidate_out_dim)

    def forward(self, x):
        # x shape: (B, raw_map_channels, H, W) e.g., (B, 29, 32, 32)
        if x.dim() != 4 or x.size(1) != self.branch1[0].in_channels:
            raise ValueError(f"Input map shape mismatch. Expected (B, {self.branch1[0].in_channels}, H, W), got {x.shape}")
        out1 = self.branch1(x)
        out3 = self.branch3(x)
        out5 = self.branch5(x)
        out = torch.cat([out1, out3, out5], dim=1)  # (B, mid_channels, H, W)
        out = self.conv_reduce(out)                # (B, mid_channels, H, W)
        out = self.adaptive_pool(out)              # (B, mid_channels, num_tokens, 1)
        out = out.squeeze(-1)                      # (B, mid_channels, num_tokens)
        out = out.permute(0, 2, 1)                 # (B, num_tokens, mid_channels)
        out = self.fc(out)                         # (B, num_tokens, candidate_out_dim)
        return out



###############################################################################
# MAACActorSOTA: The Core Actor Network (REVISED w/ Communication)
###############################################################################
class MAACActorSOTA_V4(nn.Module):
    """
    (V7) Final, self-contained, decentralized actor.
    - Stores both obs_radius and mem_connection_radius for correct edge feature normalization.
    """
    def __init__(self,
                 self_feature_dim: int, raw_map_channels: int, map_grid_size: int,
                 d_model: int, candidate_in_dim: int,
                 memory_map_channels: int,
                 gnn_hidden_dim: int, gnn_layers: int, gnn_heads: int,
                 temporal_hidden_dim: int, titan_nhead: int, memory_length: int,
                 role_embed_dim: int, movement_dim: int, pickup_dim: int, num_reward_components: int,
                 obs_radius: float,
                 mem_connection_radius: float,
                 dropout_rate: float):
        super().__init__()
        # Store key dimensions and configs
        self.memory_length = memory_length
        self.gnn_hidden_dim = gnn_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.role_embed_dim = role_embed_dim
        self.movement_dim = movement_dim
        self.pickup_dim = pickup_dim
        self.num_reward_components = num_reward_components
        self.obs_radius = obs_radius # For live perception graph
        self.mem_connection_radius = mem_connection_radius # For larger memory graph

        # A single, consistent edge dimension source of truth
        self.gnn_edge_dim = 16

        # --- 1. Perception Encoders (Unchanged) ---
        self.self_obs_mlp = MLP(self_feature_dim, d_model)
        self.candidate_extractor = MultiScaleCandidateExtractor(raw_map_channels, candidate_in_dim, grid_size=map_grid_size)
        self.fc_candidate = MLP(candidate_in_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=gnn_heads, dropout=dropout_rate, batch_first=True)
        self.perception_fusion = MLP(d_model * 2, d_model)

        # --- 2. GNN Components ---
        self.comm_edge_mlp = nn.Sequential(nn.LazyLinear(gnn_hidden_dim // 2), nn.GELU(), nn.Linear(gnn_hidden_dim // 2, self.gnn_edge_dim))
        self.mem_edge_mlp = nn.Sequential(nn.LazyLinear(gnn_hidden_dim // 2), nn.GELU(), nn.Linear(gnn_hidden_dim // 2, self.gnn_edge_dim))
        
        self.comm_gnn_node_encoder = MLP(NODE_FEATURE_DIM, gnn_hidden_dim)
        self.comm_gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=gnn_heads, concat=False, dropout=dropout_rate, edge_dim=self.gnn_edge_dim)
            for _ in range(gnn_layers)
        ])
        self.comm_embedding_proj = MLP(gnn_hidden_dim, d_model)
        
        self.mem_gnn_node_encoder = MLP(MEM_NODE_FEATURE_DIM + 1, gnn_hidden_dim) # +1 for cluster count
        self.mem_gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=gnn_heads, concat=False, dropout=dropout_rate, edge_dim=self.gnn_edge_dim)
            for _ in range(gnn_layers)
        ])
        self.mem_graph_embedding_proj = MLP(gnn_hidden_dim, d_model)

        self.memory_map_cnn = CNNMapEncoder(memory_map_channels, d_model)
        self.fusion_cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=gnn_heads, dropout=dropout_rate, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_post_attn_mlp = MLP(d_model, temporal_hidden_dim, hidden_dim=temporal_hidden_dim*2)
        self.gru = nn.GRU(temporal_hidden_dim, temporal_hidden_dim, num_layers=1, batch_first=True)
        # This call ensures the weights are stored contiguously in memory for cuDNN efficiency.
        self.gru.flatten_parameters() 
        self.titan_transformer = TitanTransformerEncoderLayer(d_model=temporal_hidden_dim, nhead=titan_nhead)
        self.temporal_gate = nn.Sequential(nn.Linear(temporal_hidden_dim * 2, 1), nn.Sigmoid())
        self.role_head_continuous = nn.Linear(temporal_hidden_dim, role_embed_dim)
        self.fc_movement = nn.Linear(temporal_hidden_dim, movement_dim * 2)
        self.fc_pickup = nn.Linear(temporal_hidden_dim, pickup_dim)
        self.reward_weight_head = nn.Sequential(nn.Linear(temporal_hidden_dim, 64), nn.GELU(), nn.Linear(64, self.num_reward_components))

    def get_role_reward_weights(self, final_context: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.reward_weight_head(final_context), dim=-1)

    
    def forward(self,
                self_obs: torch.Tensor, raw_map: torch.Tensor, env_graph: Optional[Batch],
                memory_map: torch.Tensor, memory_graph: Optional[Batch],
                hidden_state: Dict[str, Optional[torch.Tensor]]
               ) -> Tuple[Dict, Dict, Dict, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # This forward pass now returns the standard 9-item tuple
        B = self_obs.shape[0]; device = self_obs.device

        # --- Perception Stream ---
        baseline_emb = self.self_obs_mlp(self_obs)
        candidate_tokens = self.candidate_extractor(raw_map)
        candidate_emb = self.fc_candidate(candidate_tokens)
        cross_feat, _ = self.cross_attn(query=baseline_emb.unsqueeze(1), key=candidate_emb, value=candidate_emb)
        perception_embedding = self.perception_fusion(torch.cat([baseline_emb, cross_feat.squeeze(1)], dim=1))

        # --- GNN Stream 1: Live Communication Graph ---
        comm_embedding_raw = torch.zeros(B, self.gnn_hidden_dim, device=device)
        if env_graph is not None and env_graph.num_nodes > 0:
            env_graph = env_graph.to(device)
            if not hasattr(env_graph, 'batch') or env_graph.batch is None:
                env_graph.batch = torch.zeros(env_graph.num_nodes, dtype=torch.long, device=device)
            node_feat_comm = self.comm_gnn_node_encoder(env_graph.x)
            
            # ### Use self.obs_radius for live communication graph edge features ###
            edge_attr_comm = create_graph_edge_features(env_graph, self.comm_edge_mlp, self.obs_radius)
            
            for layer in self.comm_gnn_layers:
                node_feat_comm = F.gelu(layer(node_feat_comm, env_graph.edge_index, edge_attr=edge_attr_comm))
            ego_mask = env_graph.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5
            ego_features = node_feat_comm[ego_mask]
            ego_batch_indices = env_graph.batch[ego_mask]
            if ego_features.numel() > 0:
                comm_embedding_raw.index_copy_(0, ego_batch_indices, ego_features)
        comm_embedding = self.comm_embedding_proj(comm_embedding_raw)

        # --- GNN Stream 2: Internal Memory Graph Reasoning ---
        mem_graph_feat_raw = torch.zeros(B, self.gnn_hidden_dim, device=device)
        if memory_graph is not None and memory_graph.num_nodes > 0:
            memory_graph = memory_graph.to(device)
            if not hasattr(memory_graph, 'batch') or memory_graph.batch is None:
                memory_graph.batch = torch.zeros(memory_graph.num_nodes, dtype=torch.long, device=device)
            node_feat_mem = self.mem_gnn_node_encoder(memory_graph.x)

            # ### Use the larger self.mem_connection_radius for internal memory graph edge features ###
            edge_attr_mem = create_graph_edge_features(memory_graph, self.mem_edge_mlp, self.mem_connection_radius)

            for layer in self.mem_gnn_layers:
                node_feat_mem = F.gelu(layer(node_feat_mem, memory_graph.edge_index, edge_attr=edge_attr_mem))
            pooled_feat = pyg_nn.global_mean_pool(node_feat_mem, memory_graph.batch)
            if pooled_feat.shape[0] < B: mem_graph_feat_raw[:pooled_feat.shape[0]] = pooled_feat
            else: mem_graph_feat_raw = pooled_feat
        mem_graph_feat = self.mem_graph_embedding_proj(mem_graph_feat_raw)

        mem_map_feat = self.memory_map_cnn(memory_map)
        
        # --- [Unchanged] Fusion, Temporal, and Output streams ---
        query = perception_embedding.unsqueeze(1)
        context_stack = torch.stack([comm_embedding, mem_map_feat, mem_graph_feat], dim=1)
        attn_output, _ = self.fusion_cross_attention(query=query, key=context_stack, value=context_stack)
        fused_state = self.fusion_norm(perception_embedding + attn_output.squeeze(1))
        z_t = self.fusion_post_attn_mlp(fused_state)
        
        gru_h_prev = hidden_state.get('gru_hidden')
        episodic_cache = hidden_state.get('episodic_cache')
        gru_out, gru_h_next = self.gru(z_t.unsqueeze(1), gru_h_prev)
        titan_out = self.titan_transformer(z_t.unsqueeze(1), episodic_cache).squeeze(1)
        
        with torch.no_grad():
            new_z = z_t.unsqueeze(1).detach()
            if episodic_cache and episodic_cache[0].numel() > 0:
                new_k = torch.cat([episodic_cache[0], new_z], dim=1)[:, -self.memory_length:]
                new_v = torch.cat([episodic_cache[1], new_z], dim=1)[:, -self.memory_length:]
            else: new_k, new_v = new_z, new_z
        new_hidden_state = {'gru_hidden': gru_h_next.detach(), 'episodic_cache': (new_k, new_v)}

        gate = self.temporal_gate(torch.cat([gru_out.squeeze(1), titan_out], dim=1))
        final_context = gate * titan_out + (1 - gate) * gru_out.squeeze(1)
        
        mean, log_std = self.fc_movement(final_context).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        pickup_logits = self.fc_pickup(final_context)
        action_dist_params = {"movement_mean": mean, "movement_std": log_std.exp(), "pickup_logits": pickup_logits}
        role_output = {"continuous": self.role_head_continuous(final_context), "discrete_logits": None}
        reward_weights = self.get_role_reward_weights(final_context)
        
        movement_sampled = torch.tanh(mean)
        pickup_sampled = torch.argmax(pickup_logits, dim=-1)
        joint_action_tensor = torch.cat([movement_sampled, pickup_sampled.unsqueeze(-1).float()], dim=1)

        return (action_dist_params,           # 1. Action distribution params
                new_hidden_state,             # 2. Next hidden state
                role_output,                  # 3. Role info
                reward_weights,               # 4. Reward weights
                None,                         # 5. Predicted next obs (not used)
                None,                         # 6. Belief state (not used)
                joint_action_tensor,          # 7. Raw action tensor
                None,                         # 8. Contrastive embedding (not used)
                None)                         # 9. Global context (not used)

    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h_gru = torch.zeros(1, batch_size, self.temporal_hidden_dim, device=device)
        episodic_cache = (
            torch.empty(batch_size, 0, self.temporal_hidden_dim, device=device),
            torch.empty(batch_size, 0, self.temporal_hidden_dim, device=device)
        )
        return {'gru_hidden': h_gru, 'episodic_cache': episodic_cache}

from torch_scatter import scatter_mean

###############################################################################
# MAACPolicy: Manages Multiple Actors (FINAL SIMPLIFIED WRAPPER)
###############################################################################
class MAACPolicy(nn.Module):
    """
    (V6 Wrapper - Corrected Optimizers) A router that holds role-specific actors.
    """
    def __init__(self,
                 self_feature_dim: int, raw_map_channels: int, map_grid_size: int,
                 movement_dim: int, num_reward_components: int, agent_types: List[str],
                 d_model: int, candidate_in_dim: int,
                 temporal_hidden_dim: int, role_embed_dim: int, gnn_hidden_dim: int,
                 gnn_layers: int, gnn_heads: int, titan_nhead: int, memory_length: int,
                 obs_radius: float,
                 mem_connection_radius: float,
                 actor_lr: float, max_grad_norm: float, dropout_rate: float,
                 memory_map_channels: int = OCC_CH_COUNT,
                 **kwargs):
        
        super(MAACPolicy, self).__init__()
        self.agent_types = agent_types
        self.num_reward_components = num_reward_components
        self.max_grad_norm = max_grad_norm
        self.role_name_to_idx = {name: idx for idx, name in enumerate(agent_types)}
        self.movement_dim = movement_dim
        self.pickup_dim = 3

        self.actors = nn.ModuleDict()
        self.optimizers = {} # <<< This will now store one optimizer per role actor.

        for typ in agent_types:
            actor = MAACActorSOTA_V4(
                self_feature_dim=self_feature_dim, raw_map_channels=raw_map_channels,
                map_grid_size=map_grid_size, d_model=d_model, candidate_in_dim=candidate_in_dim,
                memory_map_channels=memory_map_channels + 1,
                gnn_hidden_dim=gnn_hidden_dim, gnn_layers=gnn_layers, gnn_heads=gnn_heads,
                temporal_hidden_dim=temporal_hidden_dim, titan_nhead=titan_nhead,
                memory_length=memory_length, role_embed_dim=role_embed_dim,
                obs_radius=obs_radius,
                mem_connection_radius=mem_connection_radius,
                dropout_rate=dropout_rate,
                movement_dim=self.movement_dim,
                pickup_dim=self.pickup_dim,
                num_reward_components=self.num_reward_components
            )
            
            # --- Create a SINGLE optimizer for all parameters of this role-specific actor ---
            self.optimizers[typ] = torch.optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=1e-4)
            self.actors[typ] = actor
            
    def _unbatch_hidden(self, hidden_state_batch: Optional[Dict], indices: torch.Tensor) -> Dict:
        # This helper function for unbatching the hidden state dictionary remains the same.
        if hidden_state_batch is None: return self.init_hidden(indices.shape[0])
        h_gru = hidden_state_batch.get('gru_hidden'); sub_h_gru = h_gru[:, indices, :] if h_gru is not None else None
        episodic_cache = hidden_state_batch.get('episodic_cache'); sub_cache = None
        if episodic_cache is not None and episodic_cache[0] is not None:
            cache_k, cache_v = episodic_cache; sub_k = cache_k[indices]; sub_v = cache_v[indices]
            sub_cache = (sub_k, sub_v)
        return {'gru_hidden': sub_h_gru, 'episodic_cache': sub_cache}

    def evaluate_actions(self,
                         obs_batch: Dict[str, Union[torch.Tensor, Batch, List[Data]]],
                         hidden_state: Optional[Dict],
                         all_agent_types_in_team: List[str],
                         **kwargs
                         ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict, Dict, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        (V6 - Corrected for Empty Batches)
        Evaluates actions, robustly handling cases where no agents of a specific role are present.
        """
        B = obs_batch['self'].shape[0]; device = obs_batch['self'].device
        outputs_by_role = {}
        role_indices_map = {role: [] for role in self.agent_types}
        for i, role in enumerate(all_agent_types_in_team):
             if role in role_indices_map:
                role_indices_map[role].append(i)

        env_graph_full_list = get_data_list_from_graph(obs_batch.get('graph'))
        mem_graph_full_list = get_data_list_from_graph(obs_batch.get('memory_graph'))

        for role, indices in role_indices_map.items():
            if not indices: continue
            
            actor = self.actors[role]
            role_idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            role_hidden_state = self._unbatch_hidden(hidden_state, role_idx_tensor)
            
            # --- Check if the graph lists for this role are empty ---
            graphs_for_role_env = [env_graph_full_list[i] for i in indices if i < len(env_graph_full_list)]
            graphs_for_role_mem = [mem_graph_full_list[i] for i in indices if i < len(mem_graph_full_list)]

            # Only create a Batch object if the list is not empty
            env_graphs_for_role = Batch.from_data_list(graphs_for_role_env) if graphs_for_role_env else None
            mem_graphs_for_role = Batch.from_data_list(graphs_for_role_mem) if graphs_for_role_mem else None

            (dist_params, next_h, role_out, reward_weights, 
             _, _, joint_action, _, _) = actor(
                self_obs=obs_batch['self'][role_idx_tensor],
                raw_map=obs_batch['map'][role_idx_tensor],
                env_graph=env_graphs_for_role,
                memory_map=obs_batch['memory_map'][role_idx_tensor],
                memory_graph=mem_graphs_for_role,
                hidden_state=role_hidden_state
            )
            outputs_by_role[role] = (dist_params, next_h, role_out, reward_weights, joint_action)

        # --- Collate results
        final_movement=torch.zeros(B,self.movement_dim,device=device); final_pickup_logits=torch.zeros(B,self.pickup_dim,device=device)
        final_log_pi=torch.zeros(B,1,device=device); final_role_continuous=torch.zeros(B,self.actors[self.agent_types[0]].role_embed_dim,device=device)
        final_reward_weights = torch.zeros(B, self.num_reward_components, device=device)
        final_joint_action = torch.zeros(B, self.movement_dim + 1, device=device)
        
        final_h_gru, final_episodic_k, final_episodic_v = None, None, None
        first_valid_h = next((o[1] for o in outputs_by_role.values() if o and o[1]), None)
        if first_valid_h and first_valid_h.get('gru_hidden') is not None:
            h_gru_template = first_valid_h['gru_hidden']; final_h_gru = torch.zeros(h_gru_template.shape[0], B, h_gru_template.shape[2], device=device)
        if first_valid_h and first_valid_h.get('episodic_cache') is not None and first_valid_h['episodic_cache'][0] is not None:
            k_template, _ = first_valid_h['episodic_cache']; final_episodic_k = torch.zeros(B, k_template.shape[1], k_template.shape[2], device=device); final_episodic_v = torch.zeros(B, k_template.shape[1], k_template.shape[2], device=device)

        for role, indices in role_indices_map.items():
            if not indices or role not in outputs_by_role: continue
            dist_params, next_h, role_out, reward_weights, joint_action_t = outputs_by_role[role]
            role_idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            
            movement_dist = D.Normal(dist_params['movement_mean'], dist_params['movement_std']); pickup_dist = D.Categorical(logits=dist_params['pickup_logits'])
            movement_sampled_pretanh = movement_dist.rsample(); movement_sampled = torch.tanh(movement_sampled_pretanh); pickup_sampled = pickup_dist.sample()
                        
            log_pi_mov = (movement_dist.log_prob(movement_sampled_pretanh) - torch.log((1-movement_sampled.pow(2)).clamp(min=epsilon))).sum(-1,True) 
            log_pi_pick = pickup_dist.log_prob(pickup_sampled).unsqueeze(-1)            
            
            final_movement.index_copy_(0,role_idx_tensor,movement_sampled); final_pickup_logits.index_copy_(0,role_idx_tensor,dist_params['pickup_logits'])
            final_log_pi.index_copy_(0,role_idx_tensor,log_pi_mov+log_pi_pick); final_role_continuous.index_copy_(0,role_idx_tensor,role_out['continuous'])
            final_reward_weights.index_copy_(0, role_idx_tensor, reward_weights)
            final_joint_action.index_copy_(0, role_idx_tensor, joint_action_t)
            
            if final_h_gru is not None and next_h.get('gru_hidden') is not None: final_h_gru[:, role_idx_tensor, :] = next_h['gru_hidden']
            if final_episodic_k is not None and next_h.get('episodic_cache') is not None:
                k,v=next_h['episodic_cache']; final_episodic_k[role_idx_tensor]=k; final_episodic_v[role_idx_tensor]=v

        final_pickup_sampled = torch.argmax(final_pickup_logits, dim=-1)
        expanded_log_pi = final_log_pi.expand(-1, self.num_reward_components)
        
        action_dict = {"movement": final_movement, "pickup": final_pickup_sampled, "joint_action": final_joint_action}
        next_hidden = {'gru_hidden': final_h_gru, 'episodic_cache':(final_episodic_k, final_episodic_v) if final_episodic_k is not None else None}
        role_info = {'continuous': final_role_continuous, 'discrete_idx': torch.tensor([self.role_name_to_idx.get(t,0) for t in all_agent_types_in_team], device=device), 'agent_type': all_agent_types_in_team}

        # Return the standardized 9-item tuple
        return (action_dict,           # 1. Action dictionary
                expanded_log_pi,       # 2. Log probability of the action
                next_hidden,           # 3. Next hidden state
                role_info,             # 4. Role information
                final_reward_weights,  # 5. Predicted reward weights
                None,                  # 6. Predicted next obs (None for MAAC)
                final_joint_action,    # 7. Raw action tensor (same as in dict for MAAC)
                None,                  # 8. Contrastive embedding (None for MAAC)
                None)                  # 9. Belief state (None for MAAC)
    
    def init_hidden(self, batch_size=1):
        if not self.actors: return {'gru_hidden': None, 'episodic_cache': None}
        template_actor = self.actors[self.agent_types[0]]
        return template_actor.init_hidden(batch_size)

    def clip_gradients(self):
        # This is now handled at the training script level, but the method can be kept as a utility
        if self.max_grad_norm > 0:
            for optim in self.optimizers.values():
                torch.nn.utils.clip_grad_norm_(optim.param_groups[0]['params'], self.max_grad_norm)

    def get_actor_and_optimizers(self, agent_type):
        if agent_type not in self.actors:
             # print(f"Warning: Agent type '{agent_type}' not found. Using default '{self.agent_types[0]}'.")
             agent_type = self.agent_types[0]
        return self.actors[agent_type], self.optimizers[agent_type]

    def get_reward_weights(self, agent_type=None):
        if agent_type is None or agent_type not in self.agent_types: agent_type = self.agent_types[0]
        actor = self.actors[agent_type]
        if hasattr(actor, 'get_role_reward_weights'): return actor.get_role_reward_weights()
        else:
            # print(f"Warning: Actor for type {agent_type} does not have get_role_reward_weights method.")
            current_device = next(self.parameters()).device
            return torch.ones(self.num_reward_components, device=current_device) / self.num_reward_components

    def get_default_role_info(self, batch_size=1):
        default_actor = self.actors[self.agent_types[0]]
        if hasattr(default_actor, 'get_default_role_info'): return default_actor.get_default_role_info(batch_size=batch_size)
        else:
            # print("Warning: Default actor missing get_default_role_info method.")
            current_device = next(self.parameters()).device
            # Ensure discrete_idx is added
            return { "continuous": torch.zeros(batch_size, default_actor.role_embed_dim, device=current_device),
                     "agent_type": default_actor.agent_type,
                     "discrete_idx": torch.zeros(batch_size, dtype=torch.long, device=current_device) }

    @torch.no_grad()
    def act(self,
            obs: Dict[str, Union[np.ndarray, torch.Tensor, Data]],
            hidden_state: Optional[Dict] = None,
            agent_type: Optional[str] = None,
            noise_scale: float = 0.0
            ) -> Tuple[Dict[str, Union[np.ndarray, int]], Dict, Dict, Dict]:
        """
        (Standardized) Selects an action for a single agent.
        Returns a standardized 4-tuple: (action_dict, next_hidden_state, role_info, aux_outputs)
        """
        self.eval()
        current_device = next(self.parameters()).device
        agent_type = agent_type if agent_type in self.agent_types else self.agent_types[0]

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

        # Use the standardized act_batch for the core logic
        action_batch_dict, next_hidden_batch, role_batch_dict, aux_outputs_batch = self.act_batch(
            obs_batch=obs_batch,
            all_agent_types_in_team=[agent_type],
            hidden_state=hidden_state,
            noise_scale=noise_scale
        )

        # Un-batch results for a single agent
        movement_np = action_batch_dict['movement'][0].cpu().numpy()
        pickup_val = action_batch_dict['pickup'][0].item()
        action_env_dict = {"movement": movement_np, "pickup": int(np.clip(round(pickup_val), 0, self.pickup_dim - 1))}
        
        # Detach and convert to numpy where applicable
        next_hidden_single = {k: v.cpu().numpy() if hasattr(v,'cpu') and v is not None else v for k, v in next_hidden_batch.items()}
        role_info_single = {k: v[0].cpu().numpy() if hasattr(v,'cpu') and v is not None else v for k, v in role_batch_dict.items()}
        aux_outputs_single = {k: v[0].cpu().numpy() if hasattr(v,'cpu') and v is not None else v for k,v in aux_outputs_batch.items()}

        return action_env_dict, next_hidden_single, role_info_single, aux_outputs_single

    def act_batch(self,
                  obs_batch: Dict[str, Union[torch.Tensor, Batch]],
                  hidden_state: Optional[Dict] = None,
                  all_agent_types_in_team: Optional[List[str]] = None,
                  noise_scale: float = 0.0,
                  **kwargs # <<< ADDED to absorb unused kwargs
                  ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict, Dict]:
        """
        (V7 Standardized) Runs a forward pass for a batch of agents.
        Returns a standardized 4-tuple: (action_dict, next_hidden, role_info, aux_outputs)
        """
        # Get the full 9-tuple from the core evaluation method
        eval_output = self.evaluate_actions(
            obs_batch=obs_batch,
            hidden_state=hidden_state,
            all_agent_types_in_team=all_agent_types_in_team,
        )
        sampled_action_dict, _, next_hidden_state, role_info, _, _, _, _, _ = eval_output

        # Add noise for exploration during rollouts
        if noise_scale > 0:
            noise = torch.randn_like(sampled_action_dict['movement']) * noise_scale
            sampled_action_dict['movement'] = torch.clamp(sampled_action_dict['movement'] + noise, -1.0, 1.0)
            sampled_action_dict['joint_action'] = torch.cat([
                sampled_action_dict['movement'],
                sampled_action_dict['pickup'].unsqueeze(-1).float()
            ], dim=1)
        
        # Detach all tensor outputs before returning
        action_output_dict_final = {k: v.detach() for k, v in sampled_action_dict.items()}
        role_info_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in role_info.items()}
        
        next_hidden_detached = {}
        if next_hidden_state and next_hidden_state.get('gru_hidden') is not None:
            h_gru = next_hidden_state['gru_hidden'].detach()
            cache_k, cache_v = next_hidden_state.get('episodic_cache', (None, None))
            episodic_cache_detached = (cache_k.detach() if cache_k is not None else None, 
                                       cache_v.detach() if cache_v is not None else None)
            next_hidden_detached = {'gru_hidden': h_gru, 'episodic_cache': episodic_cache_detached}
        
        # MAAC does not produce auxiliary outputs in its standard inference pass
        aux_outputs = {}

        return action_output_dict_final, next_hidden_detached, role_info_detached, aux_outputs
    
    def init_hidden_for_batch(self, agent_types_in_batch: List[str]) -> Dict:
        """
        (Corrected) Initializes a correctly structured batch of hidden states for agents with different roles.
        """
        batch_size = len(agent_types_in_batch)
        if batch_size == 0 or not self.actors:
            return {'gru_hidden': None, 'episodic_cache': None}
        
        # Get template from the first actor type to determine dimensions
        template_actor = self.actors[self.agent_types[0]]
        h_gru_template = template_actor.init_hidden(1)['gru_hidden']
        k_template, _ = template_actor.init_hidden(1)['episodic_cache']
        
        # Create final tensors with the full batch size
        final_h_gru = torch.zeros(h_gru_template.shape[0], batch_size, h_gru_template.shape[2], device=device)
        # Episodic cache is initialized empty for all at the start of an episode
        final_episodic_k = torch.empty(batch_size, 0, k_template.shape[2], device=device)
        final_episodic_v = torch.empty(batch_size, 0, k_template.shape[2], device=device)

        return {'gru_hidden': final_h_gru, 'episodic_cache': (final_episodic_k, final_episodic_v)}

    def assign_roles_for_episode(self, agent_indices_for_team: List[int]) -> Dict[int, str]:
        """
        Assigns roles to the agents designated for this policy for an episode.
        This is the encapsulated version of the logic previously in the training script.
        Args:
            agent_indices_for_team (List[int]): A list of the global agent indices
                                                that have been assigned to this policy's team.
        Returns:
            Dict[int, str]: A map from global_agent_idx -> role_string.
        """
        agent_role_assignments = {}
        
        # self.agent_types is ['scout', 'collector', 'defender', 'attacker']
        available_roles = self.agent_types
        if not available_roles:
            return {} # Should not happen if policy is initialized correctly

        num_roles = len(available_roles)
        if num_roles == 0:
            return {}
            
        # Simple round-robin assignment for now. This can be made more sophisticated later
        # (e.g., sorting agents by strength, etc.) without changing the training loop.
        for i, agent_global_idx in enumerate(agent_indices_for_team):
            assigned_role = available_roles[i % num_roles]
            agent_role_assignments[agent_global_idx] = assigned_role
            
        return agent_role_assignments

    def eval(self):
        super().eval()
        for actor in self.actors.values(): actor.eval()
        if hasattr(self, 'shared_actor_edge_feature_mlp') and self.shared_actor_edge_feature_mlp: self.shared_actor_edge_feature_mlp.eval()
        if hasattr(self, 'shared_comm_gnn_layers') and self.shared_comm_gnn_layers: self.shared_comm_gnn_layers.eval()
        if hasattr(self, 'shared_comm_gnn_output_norm') and self.shared_comm_gnn_output_norm: self.shared_comm_gnn_output_norm.eval()
        if hasattr(self, 'shared_comm_fusion_mlp') and self.shared_comm_fusion_mlp: self.shared_comm_fusion_mlp.eval()
        if hasattr(self, 'role_type_embedding_layer') and self.role_type_embedding_layer: self.role_type_embedding_layer.eval()


    def train(self, mode=True):
        super().train(mode)
        for actor in self.actors.values(): actor.train(mode)
        if hasattr(self, 'shared_actor_edge_feature_mlp') and self.shared_actor_edge_feature_mlp: self.shared_actor_edge_feature_mlp.train(mode)
        if hasattr(self, 'shared_comm_gnn_layers') and self.shared_comm_gnn_layers: self.shared_comm_gnn_layers.train(mode)
        if hasattr(self, 'shared_comm_gnn_output_norm') and self.shared_comm_gnn_output_norm: self.shared_comm_gnn_output_norm.train(mode)
        if hasattr(self, 'shared_comm_fusion_mlp') and self.shared_comm_fusion_mlp: self.shared_comm_fusion_mlp.train(mode)
        if hasattr(self, 'role_type_embedding_layer') and self.role_type_embedding_layer: self.role_type_embedding_layer.train(mode)












# In maac_attentionGNN.py
# REPLACE the entire if __name__ == "__main__" block at the end of the file.

if __name__ == "__main__":
    import traceback
    import torch
    import random
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    from constants import NODE_FEATURE_DIM, RAW_CH_COUNT, OCC_CH_COUNT, MEM_NODE_FEATURE_DIM, NODE_FEATURE_MAP
    from env.env import Swarm2DEnv

    print("\n" + "="*60)
    print("  Testing MAACPolicy with V5 Actors (Stateless Architecture)")
    print("="*60)

    # --- Test Parameters ---
    B = 4; SELF_DIM = 24; RAW_MAP_CH = RAW_CH_COUNT; MEM_MAP_CH = OCC_CH_COUNT
    MAP_GRID = 32; ACTION_DIM = 2; PICKUP_DIM = 3; REWARD_DIM = 18
    N_DISCRETE_ROLES = 4; ROLE_EMB_DIM = 16; WORLD_WIDTH = 1000.0; WORLD_HEIGHT = 1000.0
    MAP_EMBED_DIM = 32; CNN_LAYERS = 2; H_DIM = 64; NCA_ITER = 2; MSG_DIM = 32
    MOVEMENT_DIM = 2; NUM_REWARD_COMPONENTS = 18; AGENT_TYPES = MAAC_ROLES_GLOBAL
    D_MODEL = 64; CANDIDATE_IN_DIM = 16; TEMPORAL_HIDDEN_DIM = 128; TITAN_NHEAD = 4; MEMORY_LENGTH = 10
    GNN_HIDDEN_DIM = 64; GNN_LAYERS = 2; GNN_HEADS = 4;
    MEM_CONNECTION_RADIUS_TEST = OBS_RADIUS * 2.5

    print("\n--- 1. Instantiating MAACPolicy ---")
    try:
        policy = MAACPolicy(
            self_feature_dim=SELF_DIM, raw_map_channels=RAW_MAP_CH, map_grid_size=MAP_GRID,
            movement_dim=MOVEMENT_DIM, num_reward_components=NUM_REWARD_COMPONENTS, agent_types=AGENT_TYPES,
            d_model=D_MODEL, candidate_in_dim=CANDIDATE_IN_DIM,
            temporal_hidden_dim=TEMPORAL_HIDDEN_DIM, role_embed_dim=ROLE_EMB_DIM,
            gnn_hidden_dim=GNN_HIDDEN_DIM, gnn_layers=GNN_LAYERS, gnn_heads=GNN_HEADS,
            titan_nhead=TITAN_NHEAD, memory_length=MEMORY_LENGTH,
            obs_radius=OBS_RADIUS,
            mem_connection_radius=MEM_CONNECTION_RADIUS_TEST,
            actor_lr=1e-4, max_grad_norm=1.0, dropout_rate=0.1
        ).to(device)
        print("  Policy instantiated successfully.")
    except Exception as e:
        print(f"!!!!!! Policy Init Failed: {e}"); traceback.print_exc(); sys.exit(1)

    print("\n--- 2. Generating Dummy Batched Observation (5-part context) ---")
    obs_self = torch.randn(B, SELF_DIM, device=device)
    obs_raw_map = torch.randn(B, RAW_MAP_CH, MAP_GRID, MAP_GRID, device=device)
    obs_mem_map = torch.randn(B, MEM_MAP_CH, MAP_GRID, MAP_GRID, device=device)
    
    # Create lists for individual graph Data objects
    env_graphs = []
    mem_graphs = []

    for _ in range(B):
        # Create a dummy live graph with edges
        num_live_nodes = random.randint(5, 10)
        live_pos = torch.rand(num_live_nodes, 2) * 1000
        live_graph = Data(
            x=torch.randn(num_live_nodes, NODE_FEATURE_DIM),
            pos=live_pos,
            radii=torch.rand(num_live_nodes) * 5
        )
        live_graph.edge_index = pyg_nn.radius_graph(live_pos, r=OBS_RADIUS, max_num_neighbors=16)
        env_graphs.append(live_graph)

        # Create a dummy memory graph with edges
        num_mem_nodes = random.randint(15, 25)
        mem_pos = torch.rand(num_mem_nodes, 2) * 1000
        mem_graph = Data(
            x=torch.randn(num_mem_nodes, MEM_NODE_FEATURE_DIM + 1),
            pos=mem_pos,
            radii=torch.rand(num_mem_nodes) * 5
        )
        mem_graph.edge_index = pyg_nn.radius_graph(mem_pos, r=MEM_CONNECTION_RADIUS_TEST, max_num_neighbors=32)
        mem_graphs.append(mem_graph)

    # Assemble the final observation batch
    batched_env_graph = Batch.from_data_list(env_graphs)
    batched_mem_graph = Batch.from_data_list(mem_graphs)

    obs_batch = {
        'self': obs_self,
        'map': obs_raw_map,
        'graph': batched_env_graph,
        'memory_map': obs_mem_map,
        'memory_graph': batched_mem_graph
    }
    print("  Dummy 5-part observation batch created successfully.")


    print("\n--- 3. Executing `evaluate_actions` to test the full pipeline ---")
    try:
        policy.train()
        all_agent_types = [random.choice(AGENT_TYPES) for _ in range(B)]
        initial_hidden_state = policy.init_hidden(B)

        # The policy expects a keyword argument 'all_agent_types_in_team'
        output_tuple = policy.evaluate_actions(
            obs_batch=obs_batch,
            hidden_state=initial_hidden_state,
            all_agent_types_in_team=all_agent_types
        )
        print("  `evaluate_actions` executed successfully.")
        
        # ### Unpack the new 9-item tuple ###
        action_dict, expanded_log_pi, next_hidden, role_info, reward_weights, _, joint_action, _, _ = output_tuple

        # --- Verification checks ---
        assert joint_action.shape == (B, MOVEMENT_DIM + 1), "Joint Action shape incorrect."
        print(f"  - Joint Action shape OK: {joint_action.shape}")
        assert expanded_log_pi.shape == (B, policy.num_reward_components), f"Log Pi shape incorrect."    
        print(f"  - Log Pi shape OK: {expanded_log_pi.shape}")
        assert isinstance(next_hidden, dict) and 'gru_hidden' in next_hidden
        print(f"  - Next Hidden State structure and shapes OK.")
        assert role_info['continuous'].shape == (B, ROLE_EMB_DIM)
        print(f"  - Role Info shape OK: {role_info['continuous'].shape}")
        assert reward_weights.shape == (B, policy.num_reward_components)
        print(f"  - Reward Weights shape OK: {reward_weights.shape}")

    except Exception as e:
        print(f"!!!!!! Forward Pass or Verification Failed: {e}"); traceback.print_exc()

    print("\n" + "="*60)
    print("  MAAC V5 Test Completed.")
    print("="*60 + "\n")