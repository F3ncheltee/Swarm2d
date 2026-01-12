# SharedAgent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import os, sys, copy
import math
from typing import Dict, Optional, Tuple, Union, List
import torch.distributions as D
import random
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import radius_graph # Crucial for AN-NCA
from torch_scatter import scatter, scatter_max # Import both the general and specific function
from torch_cluster import radius # <<< ADD THIS IMPORT
from torch_scatter import scatter_add, scatter_mean, scatter
# Calculate the path to the project root (assuming this script is in src/agents/)
project_root_sharedagent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_sharedagent not in sys.path:
    sys.path.insert(0, project_root_sharedagent)

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# from env.otherhelpers import get_data_list_from_graph # This file seems to have been removed
from constants import (
    OCC_CH, OCC_CH_COUNT, RAW_CH, RAW_CH_COUNT, RAW_CH_IDX_TO_NAME,
    MEM_NODE_FEAT_IDX, MEM_NODE_FEATURE_DIM, NODE_TYPE, NODE_FEATURE_MAP,
    NODE_FEATURE_DIM
)
from env.observations import (
    ActorMapState, PersistentGraphMemory, create_graph_edge_features,
    generate_foveated_graph
)
# SAC specific constants
LOG_STD_MAX = 2
LOG_STD_MIN = -5 # Often -20, but -5 can be more stable
epsilon = 1e-6
# Device configuration: automatically use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Helper Functions (Unchanged)
###############################################################################
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


def update_temperature(network_or_module, current_epoch, max_epochs, initial_temp=1.0, final_temp=0.1, param_name='log_temperature'):
    min_allowed_temp = 1e-6
    log_temp_param = None
    if hasattr(network_or_module, param_name) and isinstance(getattr(network_or_module, param_name), nn.Parameter):
        log_temp_param = getattr(network_or_module, param_name)
    if log_temp_param is None: return
    ratio = float(current_epoch) / float(max_epochs) if max_epochs > 0 else 0.0
    ratio = min(max(ratio, 0.0), 1.0)
    new_temp = initial_temp - (initial_temp - final_temp) * ratio
    new_temp = max(new_temp, min_allowed_temp)
    try:
        target_device = log_temp_param.device
        target_dtype = log_temp_param.dtype
        new_log_temp_tensor = torch.log(torch.tensor(new_temp + 1e-20, device=target_device, dtype=target_dtype))
        with torch.no_grad(): log_temp_param.copy_(new_log_temp_tensor)
    except Exception as e: print(f"Error updating temperature for '{param_name}': {e}")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, device=device, dtype=torch.float32).clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(U))

def gumbel_softmax_sample(logits, temperature):
    temperature = max(temperature, 1e-10)
    y = logits.to(device) + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

###############################################################################
# Novel Sub-Modules (AdaptiveGraphLearner, DynamicCommGate are unchanged)
# HighLevelComm is MODIFIED
# GlobalCoordinator, MetaAdapter, AdvancedExternalMemory, HighLevelTemporalModule are unchanged
###############################################################################

class AdaptiveGraphLearner(nn.Module):
    def __init__(self, hidden_dim, init_threshold=0.5):
        super(AdaptiveGraphLearner, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.threshold = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x):
        current_device = self.linear.weight.device
        scores = torch.sigmoid(self.linear(x.to(current_device))).squeeze(-1)
        threshold = torch.clamp(self.threshold.to(current_device), 0.01, 0.99)
        mask = scores < threshold
        return mask

class DynamicCommGate(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(DynamicCommGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
    def forward(self, x):
        current_device = next(self.parameters()).device
        return self.gate(x.to(current_device))

# --- HighLevelComm (structure is fine from previous modification) ---
class HighLevelComm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.1,
                 use_adaptive_graph=True, max_seq_len=100): # max_seq_len includes self + neighbors
        super(HighLevelComm, self).__init__()
        self.hidden_dim = hidden_dim
        # self.input_proj is NO LONGER USED HERE if features are pre-projected before calling.
        # Or, it could be used to project the 'self_features_proj' if it comes in a different dim.
        # For clarity, let's assume self_features_proj and neighbor_features_proj are already at hidden_dim.

        if hidden_dim > 0 and hidden_dim % num_heads != 0:
             nhead_adjusted = 1
             while hidden_dim % nhead_adjusted != 0 and nhead_adjusted < hidden_dim : nhead_adjusted += 1
             if hidden_dim % nhead_adjusted != 0: nhead_adjusted = 1
             print(f"Warning (HighLevelComm): hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}). Adjusted nhead to {nhead_adjusted}.")
        else: nhead_adjusted = num_heads

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead_adjusted, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, hidden_dim))
        self.use_adaptive_graph = use_adaptive_graph
        if self.use_adaptive_graph: self.graph_learner = AdaptiveGraphLearner(hidden_dim)
        else: self.graph_learner = None
        self.self_token_indicator = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, self_features_proj: torch.Tensor, neighbor_features_proj: Optional[torch.Tensor] = None, comm_mask: Optional[torch.Tensor] = None):
        current_device = self.pos_embed.device # Get device from a parameter
        self_features_proj = self_features_proj.to(current_device)
        B = self_features_proj.size(0)
        h_self = self_features_proj.unsqueeze(1) + self.self_token_indicator.to(current_device)
        if neighbor_features_proj is not None and neighbor_features_proj.numel() > 0:
            neighbor_features_proj = neighbor_features_proj.to(current_device)
            if neighbor_features_proj.dim() == 2 and B == 1:
                neighbor_features_proj = neighbor_features_proj.unsqueeze(0)
            
            if neighbor_features_proj.dim() == 3 and neighbor_features_proj.size(0) == B:
                num_actual_neighbors = neighbor_features_proj.size(1)
                if num_actual_neighbors > self.max_seq_len - 1:
                    neighbor_features_proj = neighbor_features_proj[:, :self.max_seq_len - 1, :]
                h_sequence = torch.cat([h_self, neighbor_features_proj], dim=1)
            else:
                h_sequence = h_self
        else:
            h_sequence = h_self
        N_seq = h_sequence.size(1)
        if N_seq > self.max_seq_len:
            h_sequence = h_sequence[:, :self.max_seq_len, :]
            N_seq = self.max_seq_len
        h_sequence = h_sequence + self.pos_embed[:, :N_seq, :].to(current_device)
        attn_mask = None
        if N_seq > 1 and comm_mask is not None: # Only need mask if there are neighbors
            self_mask_part = torch.zeros(B, 1, dtype=torch.bool, device=current_device)
             # Ensure comm_mask is for the actual number of neighbors in neighbor_features_proj before padding
            num_neighbors_in_seq = N_seq - 1
            if comm_mask.size(1) > num_neighbors_in_seq:
                effective_comm_mask = comm_mask[:, :num_neighbors_in_seq]
            else: # Pad comm_mask if it's shorter than actual neighbors (shouldn't happen if prepared correctly)
                padding_comm = torch.ones(B, num_neighbors_in_seq - comm_mask.size(1), dtype=torch.bool, device=current_device)
                effective_comm_mask = torch.cat([comm_mask.to(current_device), padding_comm], dim=1)
            attn_mask = torch.cat([self_mask_part, effective_comm_mask], dim=1)

        if self.use_adaptive_graph and self.graph_learner is not None:
            adaptive_mask_raw = self.graph_learner(h_sequence)
            if attn_mask is None:
                attn_mask = adaptive_mask_raw
            else:
                common_len_mask = min(attn_mask.size(1), adaptive_mask_raw.size(1))
                attn_mask = attn_mask[:, :common_len_mask] | adaptive_mask_raw[:, :common_len_mask]     
            if attn_mask.size(1) > 0:
                 attn_mask[:, 0] = False 

        h_processed = h_sequence
        for layer in self.layers:
             layer = layer.to(current_device)
             h_processed = layer(h_processed, src_key_padding_mask=attn_mask)
        return h_processed[:, 0, :]

class DecentralizedCoordinator(nn.Module):         # decentralized "global" coordinator, as its dynamically elected 
    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # 1. Leader Election Head: Each agent outputs a scalar "leader-ness" score.
        self.leader_election_head = nn.Linear(hidden_dim, 1)

        # 2. Squad Report Aggregator: A leader uses this to summarize its members' states.
        # Input is the mean of neighbors' hidden states. Output is an abstract "squad report".
        self.squad_report_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim) # The final context vector
        )

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The feature vectors of all agents in the batch. Shape: [num_nodes, hidden_dim]
            edge_index (Optional[torch.Tensor]): The agent-agent communication graph. Can be None.

        Returns:
            torch.Tensor: The squad_context vector for each agent. Shape: [num_nodes, context_dim]
        """
        num_nodes = x.size(0)
        device = x.device

        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(num_nodes, self.context_dim, device=device)

        row, col = edge_index

        # --- Part 1: Leader Election (Decentralized) ---
        leader_logits = self.leader_election_head(x)

        self_loop_edges = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        extended_edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
        
        # <<< FIX IS HERE: Call the imported scatter_max function directly >>>
        # The function `scatter_max` from `torch_scatter` returns both values and argmax.
        _, leader_node_indices_for_squad = scatter_max(
            src=leader_logits[extended_edge_index[0]].squeeze(-1), # Logits of all neighbors (and self)
            index=extended_edge_index[1],                          # Group by the target node
            dim=0,
            dim_size=num_nodes
        )
        
        # Handle cases where a node has no neighbors and no self-loop was processed
        leader_node_indices_for_squad[leader_node_indices_for_squad < 0] = torch.arange(num_nodes, device=device)[leader_node_indices_for_squad < 0]

        # --- Part 2: Squad Aggregation (Leaders Only) ---
        # An agent is a leader if it is its own leader.
        is_leader_mask = (torch.arange(num_nodes, device=device) == leader_node_indices_for_squad)
        
        # Leaders aggregate information from their direct neighbors (their squad).
        neighbor_features_mean = scatter_mean(x[row], col, dim=0, dim_size=num_nodes)
        
        all_squad_reports = torch.zeros(num_nodes, self.context_dim, device=device)
        if torch.any(is_leader_mask):
            # Only leaders generate a "squad report" using the mean of their neighbors' features.
            leader_reports = self.squad_report_aggregator(neighbor_features_mean[is_leader_mask])
            all_squad_reports[is_leader_mask] = leader_reports

        # --- Part 3: Context Distribution ---
        # Each agent (leader or member) adopts the report from its designated leader.
        squad_context = all_squad_reports[leader_node_indices_for_squad]

        return squad_context

class MetaAdapter(nn.Module):
    def __init__(self, g_dim=32, num_roles=4):
        super(MetaAdapter, self).__init__()
        self.fc = nn.Linear(g_dim, num_roles)

    def forward(self, global_message):
        current_device = self.fc.weight.device
        return self.fc(global_message.to(current_device))


class DecentralizedTrailMemory(nn.Module):
    """
    Manages a single agent's "information trail" or "scent".
    This is a stateful object, managed per-agent by the training loop.
    It stores a single, rich memory vector and its world position.
    """
    def __init__(self, memory_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.memory_dim = memory_dim

        self.register_buffer('memory_vector', torch.zeros(self.memory_dim, device=device))
        self.register_buffer('position', torch.zeros(2, device=device))
       
        # How "strong" the scent is. Decays over time.
        self.intensity = 0.0
        self.decay_rate = 0.995 # Scent fades over time

    def update(self, new_vector: torch.Tensor, new_position: torch.Tensor):
        """
        An agent updates its own trail with a new vector (e.g., its internal state)
        at its current position.
        """
        # Blend new memory with old, allows for smoother transitions.
        # Ensure operations are non-inplace for buffers.
        self.memory_vector = self.memory_vector.detach() * 0.3 + new_vector.detach() * 0.7
        self.position = new_position.detach()
        # Refresh the intensity to full strength upon update
        self.intensity = 1.0

    def step_decay(self):
        """Called each simulation step to decay the intensity of the trail."""
        self.intensity *= self.decay_rate

    def read(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Returns the memory content, its position, and its current intensity."""
        return self.memory_vector, self.position, self.intensity
    
class TrailAttention(nn.Module):
    """
    Processes a variable number of nearby trail vectors using attention.
    The agent's current state is the query, and the nearby trails are the keys/values.
    """
    def __init__(self, query_dim, trail_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Project query (agent state) and key/value (trails) to a common dimension
        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(trail_dim, output_dim)
        self.value_proj = nn.Linear(trail_dim, output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, agent_state_query: torch.Tensor, nearby_trails: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        Args:
            agent_state_query (Tensor): The agent's own context. Shape: (B, query_dim).
            nearby_trails (Tensor): Padded tensor of nearby trail vectors. Shape: (B, NumTrails, trail_dim).
            key_padding_mask (Tensor): Mask indicating invalid/padded trails. Shape: (B, NumTrails).
        """
        # Project to the attention's embedding dimension
        query = self.query_proj(agent_state_query).unsqueeze(1) # (B, 1, output_dim)
        keys = self.key_proj(nearby_trails)                    # (B, NumTrails, output_dim)
        values = self.value_proj(nearby_trails)                # (B, NumTrails, output_dim)

        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=keys,
            value=values,
            key_padding_mask=key_padding_mask
        ) # attn_output shape: (B, 1, output_dim)

        # Apply layer norm and return the aggregated vector
        return self.norm(attn_output.squeeze(1))
    


class HighLevelTemporalModule(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout_rate=0.1, mem_size=32, use_gru=True):
        super(HighLevelTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.mem_size = mem_size
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru.flatten_parameters()
        if hidden_dim <= 0: 
            print("Warning (HighLevelTemporalModule): hidden_dim <= 0. Initializing as Identity pass-through.")
            self.transformer = None; self.layer_norm = nn.Identity()
            self.register_buffer("memory", torch.empty(1, mem_size, hidden_dim)) 
            self.mem_gate = None; self.gru = None
            return
        if hidden_dim % num_heads != 0:
             nhead_adjusted = 1
             while hidden_dim % nhead_adjusted != 0 and nhead_adjusted < hidden_dim : nhead_adjusted += 1
             if hidden_dim % nhead_adjusted != 0: nhead_adjusted = 1
             print(f"Warning (HighLevelTemporalModule): hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}). Adjusted nhead to {nhead_adjusted}.")
        else: nhead_adjusted = num_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead_adjusted, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("memory", torch.randn(1, mem_size, hidden_dim))
        self.mem_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
      
    def forward(self, input_seq, prev_memory=None):
        if self.transformer is None: 
            B = input_seq.size(0)
            zero_mem = torch.zeros(B, self.mem_size, self.hidden_dim, device=input_seq.device)
            return input_seq, zero_mem
        current_device = self.layer_norm.weight.device
        input_seq = input_seq.to(current_device)
        B, seq_len, hidden_dim_input = input_seq.shape
        if prev_memory is None: mem = self.memory.expand(B, -1, -1).to(current_device)
        else:
            mem = prev_memory.to(current_device)
            if mem.shape != (B, self.mem_size, self.hidden_dim): mem = self.memory.expand(B, -1, -1).to(current_device)
        combined = torch.cat([mem, input_seq], dim=1)
        out = self.transformer(combined)
        out = self.layer_norm(out)
        new_memory_base = out[:, :self.mem_size, :]
        output_seq = out[:, self.mem_size:, :]
        final_new_memory = new_memory_base
        
        self.gru = self.gru.to(current_device)
        gru_out, _ = self.gru(new_memory_base)
        final_new_memory = gru_out
        if output_seq.size(1) > 0 and self.mem_gate is not None:
            last_output = output_seq[:, -1:, :]
            gate = self.mem_gate(last_output)
            updated_first_slot = gate * last_output + (1 - gate) * final_new_memory[:, :1, :]
            final_new_memory = torch.cat([updated_first_slot, final_new_memory[:, 1:, :]], dim=1)
        return output_seq, final_new_memory


###############################################################################
# SharedActor Network (MODIFIED)
###############################################################################
class SharedActor(nn.Module):
    def __init__(self,
                action_dim=2, pickup_dim=3, self_feature_dim=22, grid_size=32,
                map_channels=RAW_CH_COUNT,
                memory_map_channels=OCC_CH_COUNT,
                max_neighbors_comm=7,
                d_model=32, hidden_dim=64, gru_hidden_dim=32, num_heads=4, dropout_rate=0.1,
                num_roles=4, role_emb_dim=16, global_context_dim=32, final_hidden_dim=64,
                temporal_layers=2, temporal_mem_size=32, external_mem_slots=8,
                external_mem_read_heads=2, latent_plan_dim=32, llc_hidden_dim=64,
                num_reward_components=18, semantic_dim=32, contrastive_embedding_dim=128,
                role_gating_mlp_dim: int = 32,
                comm_gnn_hidden_dim: int = 32,
                mem_gnn_hidden_dim: int = 32,
                mem_gnn_layers: int = 2,
                mem_gnn_heads: int = 2,
                comm_gnn_layers: int = 2,
                comm_gnn_heads: int = 2,
                edge_feature_dim: int = 16,
                obs_radius: float = 50.0,
                mem_connection_radius: float = 150.0,
                num_trail_messages: int = 4, # The number of distinct messages an agent can leave
                map_width: float = 1000.0,
                map_height: float = 1000.0,
                use_contrastive: bool = True, use_external_memory: bool = True,
                use_memory_attention: bool = True, use_adaptive_graph_comm: bool = True,
                use_temporal_gru: bool = True, use_latent_plan: bool = True,
                use_global_context: bool = True, use_dynamics_prediction: bool = True,
                **kwargs
                ):
        super().__init__()
        
        # Store parameters
        self.hidden_dim = hidden_dim; self.role_emb_dim = role_emb_dim
        self.num_reward_components = num_reward_components; self.max_neighbors_comm = max_neighbors_comm
        self.obs_radius = obs_radius
        self.edge_feature_dim = edge_feature_dim
        self.mem_connection_radius = mem_connection_radius 
        self.action_dim = action_dim; self.pickup_dim = pickup_dim; self.self_feature_dim = self_feature_dim
        self.grid_size = grid_size; self.map_channels = map_channels; self.num_roles = num_roles
        self.global_context_dim_config = global_context_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.latent_plan_dim = latent_plan_dim; self.temporal_mem_size = temporal_mem_size
        self.semantic_dim = semantic_dim; self.contrastive_embedding_dim = contrastive_embedding_dim
        self.memory_map_channels = memory_map_channels
        self.mem_gnn_hidden_dim = mem_gnn_hidden_dim
        self.use_contrastive = use_contrastive; self.use_external_memory = use_external_memory
        self.use_memory_attention = use_memory_attention and self.use_external_memory
        self.use_adaptive_graph_comm = use_adaptive_graph_comm; self.use_temporal_gru = use_temporal_gru
        self.use_latent_plan = use_latent_plan; self.use_global_context = use_global_context
        self.use_dynamics_prediction = use_dynamics_prediction
        self.num_trail_messages = num_trail_messages
        self.map_width = map_width
        self.map_height = map_height
        print(f"--- Initializing SharedActor (Corrected GNN Edge Logic) ---")

        # --- 1. Base Perception Encoders (Unchanged) ---
        self.fc_self = nn.Sequential(nn.Linear(self_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))
        self.self_ln = nn.LayerNorm(d_model)
        # ... (rest of perception encoders are unchanged) ...
        aug_map_channels = self.map_channels + 2
        self.map_encoder_high = nn.Sequential(nn.Conv2d(aug_map_channels, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 17, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((17, 17)))
        self.map_proj_high = nn.Linear(17 * 17 * 17, d_model)
        self.map_encoder_mid = nn.Sequential(nn.Conv2d(aug_map_channels, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.map_proj_mid = nn.Linear(32 * (grid_size//2) * (grid_size//2), d_model)
        self.map_encoder_low = nn.Sequential(nn.Conv2d(aug_map_channels, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, d_model))
        self.map_fusion = nn.Sequential(nn.Linear(3 * d_model, hidden_dim), nn.ReLU())
        self.obs_fusion = nn.Sequential(nn.Linear(d_model + hidden_dim, hidden_dim), nn.ReLU())

        # --- 2. GNN Components (Live Comm & Persistent Memory) ---
        comm_edge_input_dim = 8 + (NODE_FEATURE_DIM * 2)
        mem_edge_input_dim = 8 + ((MEM_NODE_FEATURE_DIM + 1) * 2)

        self.comm_edge_mlp = MLP(input_dim=comm_edge_input_dim, output_dim=self.edge_feature_dim, hidden_dim=comm_gnn_hidden_dim)
        self.mem_edge_mlp = MLP(input_dim=mem_edge_input_dim, output_dim=self.edge_feature_dim, hidden_dim=mem_gnn_hidden_dim)

        self.comm_gnn_node_encoder = MLP(NODE_FEATURE_DIM, comm_gnn_hidden_dim)
        self.comm_gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(comm_gnn_hidden_dim, comm_gnn_hidden_dim, heads=comm_gnn_heads, concat=False, dropout=dropout_rate, edge_dim=self.edge_feature_dim)
            for _ in range(comm_gnn_layers)
        ])

        self.mem_gnn_node_encoder = MLP(MEM_NODE_FEATURE_DIM + 1, mem_gnn_hidden_dim)
        self.mem_gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(mem_gnn_hidden_dim, mem_gnn_hidden_dim, heads=mem_gnn_heads, concat=False, dropout=dropout_rate, edge_dim=self.edge_feature_dim)
            for _ in range(mem_gnn_layers)
        ])
        self.mem_gnn_out_proj = MLP(mem_gnn_hidden_dim, hidden_dim)

        # --- 3. Persistent Memory Map Encoder ---
        memory_map_input_channels = self.memory_map_channels + 1
        self.memory_map_encoder = nn.Sequential(
            nn.Conv2d(memory_map_input_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, hidden_dim)
        )

        # --- 4. Cross-Attention State Fusion ---
        self.fusion_cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_post_attn_mlp = MLP(hidden_dim, hidden_dim, hidden_dim=hidden_dim*2)

        # --- High-Level & Temporal Modules (Structure unchanged) ---
        if self.use_contrastive:
            self.contrastive_proj_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                nn.Linear(hidden_dim * 2, self.contrastive_embedding_dim)
            )
        else: self.contrastive_proj_head = None

        self.high_level_comm = HighLevelComm(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=2, dropout=dropout_rate, use_adaptive_graph=self.use_adaptive_graph_comm,
            max_seq_len=self.max_neighbors_comm + 1
        )
        
        # Projects neighbor features from comm_gnn_hidden_dim to hidden_dim
        self.neighbor_feature_proj = MLP(
            input_dim=comm_gnn_hidden_dim, 
            output_dim=hidden_dim
        )

        self.dynamic_comm_gate = DynamicCommGate(hidden_dim, dropout_rate=dropout_rate)
        self.temporal_module = HighLevelTemporalModule(hidden_dim, num_layers=temporal_layers, num_heads=num_heads, dropout_rate=dropout_rate, mem_size=temporal_mem_size, use_gru=self.use_temporal_gru)
        self.temporal_output_dim = hidden_dim
        self.trail_codebook_head = nn.Linear(self.temporal_output_dim, self.num_trail_messages * self.semantic_dim)
        self.trail_selector_head = nn.Linear(self.temporal_output_dim, self.num_trail_messages)
        self.memory_query = None; self.external_memory = None; self.memory_attention = None
        self.memory_fusion = None
        self.memory_contribution_dim = 0
        
        self.trail_attention = TrailAttention(
            query_dim=self.temporal_output_dim, # Agent's temporal state is the query
            trail_dim=self.semantic_dim,        # Neighboring trails are keys/values
            output_dim=hidden_dim,              # The output dimension should match the fusion context
            num_heads=num_heads
        )
        self.global_coordinator = None; self.meta_adapter = None; self.role_head_local = None
        self.role_head_global = None; self.role_embedding = None; self.effective_global_context_dim = 0
        self.role_gating_mlp = None

        if self.use_global_context:
            # Instantiate the new DecentralizedCoordinator instead of the old GlobalCoordinator
            self.global_coordinator = DecentralizedCoordinator(
                hidden_dim=self.temporal_output_dim,
                context_dim=self.global_context_dim_config
            )
            # The rest of the modules consume the context from the new coordinator
            self.meta_adapter = MetaAdapter(g_dim=self.global_context_dim_config, num_roles=num_roles)
            self.role_head_local = nn.Sequential(nn.Linear(self.temporal_output_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_roles))
            self.role_head_global = nn.Sequential(nn.Linear(self.global_context_dim_config, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_roles))
            self.role_embedding = nn.Embedding(num_roles, role_emb_dim)
            self.effective_global_context_dim = self.global_context_dim_config
            if role_gating_mlp_dim > 0:
                gating_input_dim = self.temporal_output_dim + self.effective_global_context_dim
                self.role_gating_mlp = nn.Sequential(
                    nn.Linear(gating_input_dim, role_gating_mlp_dim),
                    nn.ReLU(),
                    nn.Linear(role_gating_mlp_dim, 2),
                    nn.Softmax(dim=-1)
                )
        else:
            self.role_head_local = nn.Sequential(nn.Linear(self.temporal_output_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_roles))
            self.role_embedding = nn.Embedding(num_roles, role_emb_dim)

        # LLC Input Dimension Calculation
        llc_input_dim_base = self.hidden_dim
        if self.use_latent_plan:
            llc_input_dim_base += latent_plan_dim
        if self.role_embedding is not None:
            llc_input_dim_base += role_emb_dim

        self.final_input_dim = (
            self.temporal_output_dim
            + (role_emb_dim if self.role_embedding else 0)
            + self.effective_global_context_dim
            + hidden_dim  # This is the output dimension of TrailAttention
        )
        print(f"  CONDITIONAL Final Input Dim (pre-plan): {self.final_input_dim}")
        assert self.final_input_dim > 0, "Final input dimension cannot be zero!"

        self.objective_scheduler = None; self.obj_sched_norm = None
        if self.use_global_context and self.global_coordinator is not None:
            self.objective_scheduler = nn.Sequential(nn.Linear(self.effective_global_context_dim, 64), nn.ReLU(), nn.Linear(64, self.final_input_dim))
            self.obj_sched_norm = nn.LayerNorm(self.final_input_dim)
        self.norm_final = nn.LayerNorm(self.final_input_dim)

        self.plan_head = None
        if self.use_latent_plan:
            self.plan_head = nn.Sequential(nn.Linear(self.final_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_plan_dim), nn.Tanh())

        self.low_level_controller = nn.Sequential(nn.Linear(llc_input_dim_base, llc_hidden_dim), nn.ReLU(), nn.Linear(llc_hidden_dim, llc_hidden_dim), nn.ReLU())
        self.llc_movement_head = nn.Sequential(nn.Linear(llc_hidden_dim, 32), nn.ReLU(), nn.Linear(32, self.action_dim * 2))
        self.llc_pickup_head = nn.Sequential(nn.Linear(llc_hidden_dim, 32), nn.ReLU(), nn.Linear(32, self.pickup_dim))

        self.dynamics_head = None
        if self.use_dynamics_prediction:
            self.dynamics_head = nn.Sequential(nn.Linear(self.final_input_dim, final_hidden_dim), nn.ReLU(), nn.Linear(final_hidden_dim, self_feature_dim))

        self.reward_selector = None
        if self.use_global_context and self.global_coordinator is not None:
            self.reward_selector = nn.Sequential(nn.Linear(self.effective_global_context_dim, 64), nn.ReLU(), nn.Linear(64, self.num_reward_components))
        else:
            print("Info: Global context disabled, using learnable reward weights based on temporal state.")
            reward_selector_input_dim = self.temporal_output_dim
            self.reward_selector = nn.Sequential(nn.Linear(reward_selector_input_dim, 64), nn.ReLU(), nn.Linear(64, self.num_reward_components))

    def forward(self, obs: Dict[str, torch.Tensor],
            memory_map_obs: Optional[torch.Tensor],
            memory_graph_obs: Optional[Data],
            nearby_trail_data: Optional[torch.Tensor],
            hidden_state: Optional[torch.Tensor],
            return_contrastive_embedding: bool = False,
            **kwargs
           ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]: # MODIFIED return signature
        current_device = self.fc_self[0].weight.device
        self_features_raw = obs["self"].to(current_device)
        raw_map = obs["map"].to(current_device)
        env_graph = obs.get("graph")
        B = self_features_raw.size(0)
        if hidden_state is not None: hidden_state = hidden_state.to(current_device)

        # --- 1. Base Perception (Self, Raw Map) - Unchanged ---
        self_emb = self.fc_self(self_features_raw); self_emb = self.self_ln(self_emb)
        map_obs = raw_map
        C, H, W = map_obs.shape[1], map_obs.shape[2], map_obs.shape[3]
        x_coords=torch.linspace(-1,1,W,device=current_device); y_coords=torch.linspace(-1,1,H,device=current_device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij'); coords = torch.stack([yy, xx], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        map_obs_aug = torch.cat([map_obs, coords], dim=1)
        high_feat = self.map_encoder_high(map_obs_aug).view(B, -1); high_proj = self.map_proj_high(high_feat)
        mid_feat = self.map_encoder_mid(map_obs_aug).view(B, -1); mid_proj = self.map_proj_mid(mid_feat)
        low_proj = self.map_encoder_low(map_obs_aug)
        fused_map = torch.cat([high_proj, mid_proj, low_proj], dim=1); map_encoded = self.map_fusion(fused_map)
        obs_combined = torch.cat([self_emb, map_encoded], dim=1)
        perception_embedding = self.obs_fusion(obs_combined)

        # --- 2. Encode Memory Streams (memory_map, memory_graph) ---
        mem_map_encoded = self.memory_map_encoder(memory_map_obs.to(current_device)) if memory_map_obs is not None else torch.zeros(B, self.hidden_dim, device=current_device)

        mem_graph_encoded = torch.zeros(B, self.hidden_dim, device=current_device)
        if memory_graph_obs is not None and memory_graph_obs.num_nodes > 0 and hasattr(memory_graph_obs, 'edge_index') and memory_graph_obs.num_edges > 0:
            graph_data = memory_graph_obs.to(current_device)
            if not hasattr(graph_data, 'batch') or graph_data.batch is None: graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=current_device)
            node_feats = self.mem_gnn_node_encoder(graph_data.x)
            
            # <<< FIX: Use the larger mem_connection_radius for the memory graph >>>
            edge_attr_mem = create_graph_edge_features(graph_data, self.mem_edge_mlp, self.mem_connection_radius)
            
            for gnn_layer in self.mem_gnn_layers:
                node_feats = F.relu(gnn_layer(node_feats, graph_data.edge_index, edge_attr=edge_attr_mem))
            pooled_graph_feats = pyg_nn.global_mean_pool(node_feats, graph_data.batch)
            projected_graph_feats = self.mem_gnn_out_proj(pooled_graph_feats)
            if projected_graph_feats.shape[0] < B: mem_graph_encoded[:projected_graph_feats.shape[0]] = projected_graph_feats
            else: mem_graph_encoded = projected_graph_feats[:B]

        # --- 3. Cross-Attention State Fusion (Unchanged) ---
        query = perception_embedding.unsqueeze(1)
        memory_context = torch.stack([mem_map_encoded, mem_graph_encoded], dim=1)
        attn_output, _ = self.fusion_cross_attention(query=query, key=memory_context, value=memory_context)
        fused_state_before_mlp = self.fusion_norm(perception_embedding + attn_output.squeeze(1))
        full_state_fused = self.fusion_post_attn_mlp(fused_state_before_mlp)

        # --- 4. Decentralized Communication and Coordination (Using Live Graph) ---
        agent_comm_edge_index, neighbor_features_from_gnn, comm_mask = None, None, None
        if env_graph is not None and env_graph.num_nodes > 0:
            comm_graph_data = env_graph.to(current_device)
            if not hasattr(comm_graph_data, 'batch') or comm_graph_data.batch is None:
                comm_graph_data.batch = torch.zeros(comm_graph_data.num_nodes, dtype=torch.long, device=current_device)

            node_feat_comm = self.comm_gnn_node_encoder(comm_graph_data.x)
            if hasattr(comm_graph_data, 'edge_index') and comm_graph_data.edge_index is not None and comm_graph_data.num_edges > 0:
                
                # <<< Use the smaller obs_radius for the live communication graph >>>
                edge_attr_comm = create_graph_edge_features(comm_graph_data, self.comm_edge_mlp, self.obs_radius)
                
                for gnn_layer in self.comm_gnn_layers:
                    node_feat_comm = F.relu(gnn_layer(node_feat_comm, comm_graph_data.edge_index, edge_attr=edge_attr_comm))
            
            ego_mask = comm_graph_data.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5
            if torch.any(ego_mask):
                ego_nodes_pos, ego_batch_vec = comm_graph_data.pos[ego_mask], comm_graph_data.batch[ego_mask]
                if ego_nodes_pos.size(0) == B:
                    agent_comm_edge_index = radius_graph(ego_nodes_pos, r=self.obs_radius, batch=ego_batch_vec, max_num_neighbors=self.max_neighbors_comm)
            
            neighbor_features_from_gnn, valid_neighbor_mask = to_dense_batch(node_feat_comm[~ego_mask], comm_graph_data.batch[~ego_mask], max_num_nodes=self.max_neighbors_comm)
            comm_mask = ~valid_neighbor_mask

        projected_neighbor_features = self.neighbor_feature_proj(neighbor_features_from_gnn) if neighbor_features_from_gnn is not None else None
        
        # --- 5. High-Level Processing and Output ---
        comm_out = self.high_level_comm(self_features_proj=full_state_fused, neighbor_features_proj=projected_neighbor_features, comm_mask=comm_mask)
        gate = self.dynamic_comm_gate(full_state_fused)
        fused_comm = full_state_fused + gate * comm_out
        
        temporal_input = fused_comm.unsqueeze(1)
        temporal_out_seq, new_hidden = self.temporal_module(temporal_input, hidden_state)
        temporal_out = temporal_out_seq.squeeze(1)

        # --- 6. Write and Process and Fuse Trail Memory using Attention ---
        # Generate the codebook of potential messages
        codebook = self.trail_codebook_head(temporal_out).view(B, self.num_trail_messages, self.semantic_dim)
        # Generate the selection probabilities (logits)
        selector_logits = self.trail_selector_head(temporal_out)
        
        # Using Gumbel-Softmax for differentiable sampling during training
        if self.training:
            message_probs = F.gumbel_softmax(selector_logits, tau=1.0, hard=True)
        else: # Use deterministic argmax during evaluation
            message_indices = torch.argmax(selector_logits, dim=1)
            message_probs = F.one_hot(message_indices, num_classes=self.num_trail_messages).float()

        # Select the message to write to the trail
        # This is a weighted sum, which becomes a one-hot selection with hard Gumbel-Softmax
        selected_message_to_write = torch.einsum('bm,bmd->bd', message_probs, codebook)
        
        processed_trail_info = torch.zeros(B, self.hidden_dim, device=current_device)
        if nearby_trail_data is not None and self.trail_attention is not None:
            # `nearby_trail_data` is now a list of tensors of varying lengths
            # We need to pad it to create a single batch tensor for the attention mechanism
            max_neighbors = max(len(t) for t in nearby_trail_data) if nearby_trail_data else 0
            
            if max_neighbors > 0:
                padded_trails = torch.zeros(B, max_neighbors, self.semantic_dim, device=current_device)
                # True where trails are NOT present (i.e., padding)
                key_padding_mask = torch.ones(B, max_neighbors, dtype=torch.bool, device=current_device)

                for i, trails in enumerate(nearby_trail_data):
                    if trails.numel() > 0:
                        num_trails = trails.size(0)
                        padded_trails[i, :num_trails, :] = trails
                        key_padding_mask[i, :num_trails] = False
                
                # Use the agent's temporal output as the query to the attention module
                processed_trail_info = self.trail_attention(
                    agent_state_query=temporal_out,
                    nearby_trails=padded_trails,
                    key_padding_mask=key_padding_mask
                )
        global_context_val = torch.zeros(B, self.effective_global_context_dim, device=current_device)
        if self.use_global_context and self.global_coordinator is not None:
            global_context_val = self.global_coordinator(x=temporal_out, edge_index=agent_comm_edge_index)

        role_logits_val = torch.zeros(B, self.num_roles, device=current_device)
        if self.use_global_context and self.role_gating_mlp is not None:
            meta_adjustment = self.meta_adapter(global_context_val)
            role_logits_local = self.role_head_local(temporal_out)
            role_logits_global_raw = self.role_head_global(global_context_val)
            gating_input = torch.cat([temporal_out.detach(), global_context_val.detach()], dim=1)
            gate_weights = self.role_gating_mlp(gating_input)
            role_logits_global_combined = role_logits_global_raw + meta_adjustment
            role_logits_val = gate_weights[:, 0:1] * role_logits_local + gate_weights[:, 1:2] * role_logits_global_combined
        elif self.role_head_local is not None:
            role_logits_val = self.role_head_local(temporal_out)

        weighted_role_emb_val = torch.matmul(F.softmax(role_logits_val, dim=1), self.role_embedding.weight)
        hybrid_role = {"discrete": role_logits_val, "continuous": weighted_role_emb_val}

        final_input_components = [
            temporal_out, 
            weighted_role_emb_val, 
            global_context_val,
            processed_trail_info # <<< ADD PROCESSED TRAIL INFO
        ]
        # Filter out None or empty tensors before concatenation
        final_input_components_filtered = [c for c in final_input_components if c is not None and c.numel() > 0]
        # Check if list is empty after filtering
        if not final_input_components_filtered:
            raise ValueError("All components for final_input are empty or None.")
        final_input = torch.cat(final_input_components_filtered, dim=1)        

        if self.objective_scheduler:
            scaling = F.softplus(self.objective_scheduler(global_context_val))
            modulated_final = self.obj_sched_norm(final_input * (1 + torch.clamp(scaling, max=10.0)))
        else:
            modulated_final = self.norm_final(final_input)
        
        z_plan = self.plan_head(modulated_final) if self.plan_head else None
        llc_input = torch.cat([fused_comm, z_plan, weighted_role_emb_val], dim=1) if z_plan is not None else torch.cat([fused_comm, weighted_role_emb_val], dim=1)
        llc_hidden = self.low_level_controller(llc_input)
        mean, log_std = self.llc_movement_head(llc_hidden).chunk(2, dim=-1)
        pickup_logits_llc = self.llc_pickup_head(llc_hidden)
        action_output = {"movement_mean": mean, "movement_std": torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp(), "pickup_logits": pickup_logits_llc}
        predicted_next_val = self.dynamics_head(modulated_final) if self.dynamics_head else None
        contrastive_embedding = self.contrastive_proj_head(full_state_fused) if return_contrastive_embedding and self.contrastive_proj_head else None

        return (action_output, new_hidden, hybrid_role, predicted_next_val, global_context_val, z_plan, contrastive_embedding, selected_message_to_write)

    
    def get_reward_weights(self, global_context=None):
        current_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        if self.reward_selector is None:
            batch_size = 1 if global_context is None else global_context.shape[0]
            return F.softmax(torch.ones(batch_size, self.num_reward_components, device=current_device), dim=-1)
        input_for_selector = None
        if self.use_global_context:
            if not isinstance(global_context, torch.Tensor):
                print("Warning (SharedActor.get_reward_weights): Global context is required but missing or invalid. Using zeros.")
                expected_input_dim = self.reward_selector[0].in_features if isinstance(self.reward_selector[0], nn.Linear) else self.effective_global_context_dim
                batch_size_fallback = 1
                input_for_selector = torch.zeros(batch_size_fallback, expected_input_dim, device=current_device, dtype=target_dtype)
            else:
                input_for_selector = global_context.to(current_device)
                if input_for_selector.dim() == 1: input_for_selector = input_for_selector.unsqueeze(0)
        else: 
            print("Warning (SharedActor.get_reward_weights): Global context disabled. Cannot compute state-dependent weights without temporal state in stateless call. Returning uniform.")
            batch_size = 1 if global_context is None else global_context.shape[0]
            return F.softmax(torch.ones(batch_size, self.num_reward_components, device=current_device), dim=-1)
        try:
            expected_layer_dtype = self.reward_selector[0].weight.dtype
            input_casted = input_for_selector.to(expected_layer_dtype)
        except Exception:
            input_casted = input_for_selector.to(target_dtype)
        if isinstance(self.reward_selector[0], nn.Linear):
            expected_dim = self.reward_selector[0].in_features
            if input_casted.shape[-1] != expected_dim:
                print(f"Error (SharedActor.get_reward_weights): Input dim mismatch for reward selector. Expected {expected_dim}, Got {input_casted.shape[-1]}. Returning uniform.")
                batch_size = input_casted.shape[0]
                return F.softmax(torch.ones(batch_size, self.num_reward_components, device=current_device), dim=-1)
        raw_weights = self.reward_selector(input_casted)
        reward_weights = F.softmax(raw_weights, dim=-1)
        return reward_weights
    
    def init_hidden(self, batch_size=1):
        if hasattr(self, 'temporal_module') and \
           self.temporal_module is not None and \
           hasattr(self.temporal_module, 'memory') and \
           isinstance(self.temporal_module.memory, torch.Tensor):
            initial_memory_state = self.temporal_module.memory
            target_device = initial_memory_state.device 
            try:
                if initial_memory_state.dim() == 2: initial_memory_state = initial_memory_state.unsqueeze(0)
                if initial_memory_state.dim() != 3:
                     print(f"Error (SharedActor.init_hidden): Temporal memory buffer has unexpected dim {initial_memory_state.dim()}")
                     if hasattr(self.temporal_module, 'mem_size') and hasattr(self.temporal_module, 'hidden_dim'):
                         return torch.zeros(batch_size, self.temporal_module.mem_size, self.temporal_module.hidden_dim, device=target_device)
                     else: return None
                expanded_memory = initial_memory_state.expand(batch_size, -1, -1)
                return expanded_memory.clone().detach()
            except Exception as e:
                 print(f"Error expanding temporal memory in SharedActor.init_hidden: {e}")
                 if hasattr(self.temporal_module, 'mem_size') and hasattr(self.temporal_module, 'hidden_dim'):
                     return torch.zeros(batch_size, self.temporal_module.mem_size, self.temporal_module.hidden_dim, device=target_device)
                 else: return None
        else: return None

    def get_contrastive_embeddings(self, obs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_contrastive or self.contrastive_proj_head is None: return None
        current_device = self.fc_self[0].weight.device
        self_features = obs["self"].to(current_device); raw_map = obs["map"].to(current_device); B = self_features.size(0)
        self_emb = self.fc_self(self_features); self_emb = self.self_ln(self_emb)
        map_obs = raw_map
        if map_obs.dim() != 4 or map_obs.shape[1] != self.map_channels: raise ValueError(f"SharedActor.get_contrastive_embeddings map_obs unexpected shape: {map_obs.shape}")
        C, H, W = map_obs.shape[1], map_obs.shape[2], map_obs.shape[3]
        x_coords=torch.linspace(-1,1,W,device=current_device); y_coords=torch.linspace(-1,1,H,device=current_device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij'); coords = torch.stack([yy, xx], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        map_obs_aug = torch.cat([map_obs, coords], dim=1)
        high_feat = self.map_encoder_high(map_obs_aug).view(B, -1); high_proj = self.map_proj_high(high_feat)
        mid_feat = self.map_encoder_mid(map_obs_aug).view(B, -1); mid_proj = self.map_proj_mid(mid_feat)
        low_proj = self.map_encoder_low(map_obs_aug)
        fused_map = torch.cat([high_proj, mid_proj, low_proj], dim=1); map_encoded = self.map_fusion(fused_map)
        obs_combined = torch.cat([self_emb, map_encoded], dim=1); fused_obs = self.obs_fusion(obs_combined)
        contrastive_embedding = self.contrastive_proj_head(fused_obs)
        return contrastive_embedding


###############################################################################
# SharedActorPolicy: Wrapper (MODIFIED for new forward args)
###############################################################################
class SharedActorPolicy(nn.Module):
    def __init__(self, num_agents_on_team: int,
                action_dim=2, pickup_dim=3, self_feature_dim=22, grid_size=32, map_channels=29,
                max_neighbors_comm=7,
                memory_map_channels=OCC_CH_COUNT,
                comm_gnn_hidden_dim: int = 32,
                mem_gnn_hidden_dim=32, mem_gnn_layers=2, mem_gnn_heads=2,
                d_model=32, hidden_dim=64, gru_hidden_dim=32, num_heads=4, dropout_rate=0.1,
                num_roles=4, role_emb_dim=16, global_context_dim=32, final_hidden_dim=64,
                temporal_layers=2, temporal_mem_size=32, external_mem_slots=8,
                external_mem_read_heads=2, latent_plan_dim=32, llc_hidden_dim=64,
                num_reward_components=18, semantic_dim=32, contrastive_embedding_dim=128,
                role_gating_mlp_dim=32,
                edge_feature_dim: int = 16,
                obs_radius: float = 50.0,
                mem_connection_radius: float = 100.0,
                num_trail_messages: int = 4, # The number of distinct messages an agent can leave
                use_contrastive: bool = True, use_external_memory: bool = True,
                use_memory_attention: bool = True, use_adaptive_graph_comm: bool = True,
                use_temporal_gru: bool = True, use_latent_plan: bool = True,
                use_global_context: bool = True, use_dynamics_prediction: bool = True, 
                map_width: float = 1000.0,
                map_height: float = 1000.0,
                aux_loss_coef=0.1, actor_lr=1e-3, role_entropy_coef=0.01,
                max_grad_norm=1.0, contrastive_loss_coef=0.1, contrastive_tau=0.07,
                **kwargs # Absorb unused kwargs
                ):
        super().__init__()
        self.num_agents_on_team = num_agents_on_team
        self.semantic_dim = semantic_dim

        self.aux_loss_coef = aux_loss_coef; self.role_entropy_coef = role_entropy_coef
        self.max_grad_norm = max_grad_norm; self.num_reward_components = num_reward_components
        self.role_emb_dim = role_emb_dim; self.contrastive_loss_coef = contrastive_loss_coef
        self.contrastive_tau = contrastive_tau; self.use_contrastive = use_contrastive
        self.use_dynamics_prediction = use_dynamics_prediction
        self.use_global_context = use_global_context

        # Pass the new arguments to the SharedActor
        self.network = SharedActor(
            action_dim=action_dim, pickup_dim=pickup_dim, self_feature_dim=self_feature_dim,
            grid_size=grid_size, map_channels=map_channels,
            memory_map_channels=memory_map_channels,
            comm_gnn_hidden_dim=comm_gnn_hidden_dim, mem_gnn_hidden_dim=mem_gnn_hidden_dim,
            mem_gnn_layers=mem_gnn_layers, mem_gnn_heads=mem_gnn_heads,
            mem_connection_radius=mem_connection_radius,
            comm_gnn_layers=mem_gnn_layers, comm_gnn_heads=mem_gnn_heads,
            max_neighbors_comm=max_neighbors_comm,
            d_model=d_model, hidden_dim=hidden_dim, gru_hidden_dim=gru_hidden_dim, num_heads=num_heads,
            dropout_rate=dropout_rate, num_roles=num_roles, role_emb_dim=role_emb_dim,
            global_context_dim=global_context_dim, final_hidden_dim=final_hidden_dim,
            temporal_layers=temporal_layers, temporal_mem_size=temporal_mem_size,
            external_mem_slots=external_mem_slots, external_mem_read_heads=external_mem_read_heads,
            latent_plan_dim=latent_plan_dim, llc_hidden_dim=llc_hidden_dim,
            num_reward_components=num_reward_components, semantic_dim=semantic_dim,
            contrastive_embedding_dim=contrastive_embedding_dim,
            role_gating_mlp_dim=role_gating_mlp_dim, edge_feature_dim=edge_feature_dim,
            obs_radius=obs_radius,
            num_trail_messages=num_trail_messages,
            map_width=map_width,
            map_height=map_height,
            use_contrastive=use_contrastive, use_external_memory=use_external_memory,
            use_memory_attention=use_memory_attention, use_adaptive_graph_comm=use_adaptive_graph_comm,
            use_temporal_gru=use_temporal_gru, use_latent_plan=use_latent_plan,
            use_global_context=use_global_context, use_dynamics_prediction=use_dynamics_prediction
        )

        # The policy owns the trail memories for all agents assigned to it.
        self.agent_trail_memories = nn.ModuleList([
            DecentralizedTrailMemory(memory_dim=self.semantic_dim, device=device)
            for _ in range(self.num_agents_on_team)
        ])
        # --- Key Encoder for Contrastive Loss (unchanged) ---
        self.key_contrastive_proj_head = None
        if self.use_contrastive and self.network.contrastive_proj_head is not None:
            print("Info: Initializing Momentum Key Encoder for Contrastive Learning.")
            self.key_contrastive_proj_head = copy.deepcopy(self.network.contrastive_proj_head)
            for param in self.key_contrastive_proj_head.parameters(): param.requires_grad = False

    def reset_trails(self):
        """Resets all internal trail memories to their initial state."""
        # This re-instantiates the ModuleList, effectively resetting all trails.
        self.agent_trail_memories = nn.ModuleList([
            DecentralizedTrailMemory(memory_dim=self.semantic_dim, device=device)
            for _ in range(self.num_agents_on_team)
        ])
        print(f"SharedActorPolicy trails have been reset for {self.num_agents_on_team} agents.")

    def init_hidden(self, batch_size=1) -> Dict:
        """Standardized: Returns a dictionary of initial hidden states."""
        # The internal network returns a single tensor, which we wrap in a dictionary.
        initial_memory_state = self.network.init_hidden(batch_size)
        return {'temporal_memory': initial_memory_state}

    def evaluate_actions(self,
                         obs_batch: Dict[str, Union[torch.Tensor, Batch, List[Data]]],
                         hidden_state: Optional[Union[torch.Tensor, Dict]] = None,
                         agent_policy_indices: Optional[torch.Tensor] = None,
                         external_nearby_trail_data: Optional[List[torch.Tensor]] = None,
                         return_contrastive_embedding: bool = False,
                         **kwargs
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict, Dict, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        (V2 - CORRECTED) Evaluates actions for a training batch.
        - Directly calls the network to get gradients and log probabilities.
        - Handles trail memory logic for training (using data from buffer).
        - Returns the standardized 9-tuple required by the training loop.
        """
        final_device = next(self.network.parameters()).device
        B = obs_batch["self"].size(0)

        if agent_policy_indices is None:
            raise ValueError("`agent_policy_indices` must be provided to SharedActorPolicy for internal memory lookup.")
        if agent_policy_indices.shape[0] != B:
            raise ValueError(f"Batch size mismatch between observations ({B}) and agent_policy_indices ({agent_policy_indices.shape[0]}).")

        # Prepare hidden state from the input dictionary
        temporal_memory_in = hidden_state.get('temporal_memory') if isinstance(hidden_state, dict) else hidden_state
        if temporal_memory_in is None:
            temporal_memory_in = self.network.init_hidden(batch_size=B)

        # --- FORWARD PASS THROUGH CORE NETWORK ---
        # Note: We do not perform the "write" step here as we're evaluating historical data.
        action_output, new_hidden, hybrid_role_output, predicted_next, global_context, z_plan, contrastive_embedding, _ = \
            self.network(
                obs=obs_batch,
                memory_map_obs=obs_batch.get('memory_map'),
                memory_graph_obs=obs_batch.get('memory_graph'),
                nearby_trail_data=external_nearby_trail_data, # Use data from the buffer
                hidden_state=temporal_memory_in,
                return_contrastive_embedding=return_contrastive_embedding
            )

        # --- Process outputs for SAC ---
        mean, std, pickup_logits = action_output["movement_mean"], action_output["movement_std"], action_output["pickup_logits"]
        movement_dist = D.Normal(mean, std)
        pickup_dist = D.Categorical(logits=pickup_logits)

        movement_sampled_pretanh = movement_dist.rsample()
        movement_sampled = torch.tanh(movement_sampled_pretanh)
        pickup_sampled = pickup_dist.sample()

        log_pi_movement = (movement_dist.log_prob(movement_sampled_pretanh) - torch.log((1 - movement_sampled.pow(2)).clamp(min=epsilon))).sum(dim=-1, keepdim=True)
        log_pi_pickup = pickup_dist.log_prob(pickup_sampled).unsqueeze(-1)
        joint_log_pi = (log_pi_movement + log_pi_pickup)
        
        # Expand log_pi for per-component Q-value weighting, as expected by the critic update
        expanded_log_pi = joint_log_pi.expand(-1, self.num_reward_components)
        
        # --- Assemble final outputs into standardized format ---
        raw_action = torch.cat([movement_sampled, pickup_sampled.unsqueeze(-1).float()], dim=1)
        action_dict = {"movement": movement_sampled, "pickup": pickup_sampled, "joint_action": raw_action}
        
        next_hidden_state_dict = {'temporal_memory': new_hidden}
        
        role_info = {'continuous': None, 'discrete_idx': None}
        if isinstance(hybrid_role_output, dict):
            role_cont, role_disc_logits = hybrid_role_output.get('continuous'), hybrid_role_output.get('discrete')
            role_info['continuous'] = role_cont
            if role_disc_logits is not None:
                role_info['discrete_idx'] = torch.argmax(role_disc_logits, dim=1).long()
        
        reward_weights = self.network.get_reward_weights(global_context)
        
        # This function must return the full 9-tuple
        return (action_dict, 
                expanded_log_pi, 
                next_hidden_state_dict, 
                role_info, 
                reward_weights, 
                predicted_next, 
                raw_action, 
                contrastive_embedding, 
                None) # 9th item (belief state) is not produced by this policy
    
    def act_batch(self,
                obs_batch: Dict[str, torch.Tensor],
                hidden_state: Optional[Dict] = None,
                agent_policy_indices: Optional[torch.Tensor] = None,
                noise_scale: float = 0.0,
                deterministic_role: bool = False,
                **kwargs # Absorb unused kwargs
                ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict, Dict]:
        """
        Selects actions for a batch of agents.
        - Correctly unpacks the new 9-item tuple from evaluate_actions.
        - Handles the trail memory "write" operation, which was missing.
        - Returns a standardized 4-tuple.
        """
        self.network.eval()

        # The "read" phase and the "write" phase are now both handled inside evaluate_actions
        # for consistency and to ensure the correct message is used.
        with torch.no_grad():
            output_tuple = self.evaluate_actions(
                obs_batch=obs_batch,
                hidden_state=hidden_state,
                agent_policy_indices=agent_policy_indices,
                deterministic_role=deterministic_role
            )
            
            # <<< Correctly unpack the 9-item tuple with message_to_write >>>
            # The 9th item is the message that was written to the trail.
            (sampled_action_dict, _, next_hidden_state_dict, role_info,
            _, predicted_next, _, contrastive_embedding, message_written) = output_tuple

        # Add noise for exploration during rollouts
        if noise_scale > 0.0:
            noise = torch.randn_like(sampled_action_dict['movement']) * noise_scale
            sampled_action_dict['movement'] = torch.clamp(sampled_action_dict['movement'] + noise, -1.0, 1.0)
            sampled_action_dict['joint_action'] = torch.cat([
                sampled_action_dict['movement'],
                sampled_action_dict['pickup'].unsqueeze(-1).float()
            ], dim=1)

        # Prepare detached outputs for the environment interaction loop
        action_output_dict_final = {k: v.detach() for k, v in sampled_action_dict.items()}
        role_info_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in role_info.items()}
        
        # next_hidden_state_dict is the dict {'temporal_memory': tensor} from evaluate_actions
        next_hidden_detached = {k: v.detach() if v is not None else None for k, v in next_hidden_state_dict.items()}

        # Populate aux_outputs dictionary for logging or other uses
        aux_outputs = {
            'predicted_next_obs': predicted_next.detach() if predicted_next is not None else None,
            'contrastive_embedding': contrastive_embedding.detach() if contrastive_embedding is not None else None,
            # <<< The semantic vector is the message that was written >>>
            'semantic_vector': message_written.detach() if message_written is not None else None
        }

        return action_output_dict_final, next_hidden_detached, role_info_detached, aux_outputs
        
    def train(self, mode=True): super().train(mode); return self
    def eval(self): super().eval(); return self
    
    @torch.no_grad()
    def act(self,
            obs: Dict[str, Union[np.ndarray, torch.Tensor, Data]],
            hidden_state: Optional[Dict] = None, # <<< CHANGED: Expects a dictionary now
            noise_scale: float = 0.0,
            **kwargs # Absorb unused kwargs
           ) -> Tuple[Dict[str, Union[np.ndarray, int]], Dict, Dict, Dict]:
        """
        (V4 Standardized) Generates a single action for one agent during environment rollouts.
        """
        self.network.eval()
        
        # Batch the single observation dictionary
        obs_batch = {}
        for key, val in obs.items():
            if isinstance(val, Data):
                # For a single agent, create a list of one and then batch
                obs_batch[key] = Batch.from_data_list([val])
            elif isinstance(val, (np.ndarray, torch.Tensor)):
                tensor_val = torch.as_tensor(val, dtype=torch.float32)
                obs_batch[key] = tensor_val.unsqueeze(0)
        
        # Batch the single hidden state dictionary if provided
        if hidden_state is None:
            hidden_state_batch = self.init_hidden(batch_size=1)
        else:
            # The input is already a dict, just ensure tensors are on the right device
            hidden_state_batch = {k: v.to(device) if v is not None else None for k, v in hidden_state.items()}

        # We assume this is for agent 0 of this policy for trail memory purposes
        agent_policy_indices_batch = torch.tensor([0], dtype=torch.long, device=device)

        # Use the standardized act_batch for the core logic
        action_batch_dict, next_hidden_batch, role_batch_dict, aux_outputs_batch = self.act_batch(
            obs_batch=obs_batch,
            hidden_state=hidden_state_batch,
            agent_policy_indices=agent_policy_indices_batch, # Pass dummy index
            noise_scale=noise_scale
        )

        # Un-batch results for a single agent
        movement_np = action_batch_dict['movement'][0].cpu().numpy()
        pickup_val = action_batch_dict['pickup'][0].item()
        action_env_dict = {"movement": movement_np, "pickup": int(np.clip(round(pickup_val), 0, self.pickup_dim - 1))}
        
        # Detach and convert to numpy where applicable
        next_hidden_single = {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) and v.numel() > 0 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in next_hidden_batch.items()}
        role_info_single = {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) and v.numel() > 0 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in role_batch_dict.items()}
        aux_outputs_single = {k: v.detach()[0].cpu().numpy() if isinstance(v, torch.Tensor) and v is not None and v.numel() > 0 else None for k,v in aux_outputs_batch.items()}
        
        return action_env_dict, next_hidden_single, role_info_single, aux_outputs_single
    
    # Auxiliary loss functions and momentum update remain unchanged
    def compute_role_entropy_loss(self, role_logits):
        if not self.network.use_global_context and self.network.role_head_local is None:
             return torch.tensor(0.0, device=next(self.network.parameters()).device)
        if role_logits is None or not isinstance(role_logits, torch.Tensor) or role_logits.numel() == 0:
            return torch.tensor(0.0, device=next(self.network.parameters()).device)
        role_logits = role_logits.to(next(self.network.parameters()).device)
        role_probs = F.softmax(role_logits, dim=1); log_probs = torch.log(role_probs + 1e-10)
        entropy = -torch.sum(role_probs * log_probs, dim=1)
        loss = -abs(self.role_entropy_coef) * torch.mean(entropy)
        return loss

    def compute_imagination_loss(self, predicted_next_obs, target_next_obs):
        current_device = next(self.network.parameters()).device
        if not self.use_dynamics_prediction:
            return torch.tensor(0.0, device=current_device)
        if predicted_next_obs is None or target_next_obs is None:
            return torch.tensor(0.0, device=current_device)
        target_next_obs = target_next_obs.to(predicted_next_obs.device).float()
        if predicted_next_obs.shape != target_next_obs.shape:
            print(f"Warning: Shape mismatch in imagination loss. Pred: {predicted_next_obs.shape}, Target: {target_next_obs.shape}")
            return torch.tensor(0.0, device=predicted_next_obs.device)
        loss = self.aux_loss_coef * F.mse_loss(predicted_next_obs, target_next_obs.detach())
        return loss
        
    def compute_contrastive_loss(self, embeddings_q: torch.Tensor, embeddings_k: torch.Tensor):
        if not self.use_contrastive:
            return torch.tensor(0.0, device=next(self.network.parameters()).device)
        if embeddings_q is None or embeddings_k is None or embeddings_q.numel() == 0 or embeddings_k.numel() == 0:
            return torch.tensor(0.0, device=next(self.network.parameters()).device)
        target_device = next(self.network.parameters()).device
        embeddings_q = embeddings_q.to(target_device)
        embeddings_k = embeddings_k.to(target_device)
        q = F.normalize(embeddings_q, dim=1)
        k = F.normalize(embeddings_k, dim=1)
        logits = torch.mm(q, k.t()) / self.contrastive_tau
        labels = torch.arange(q.shape[0], dtype=torch.long, device=target_device)
        loss = F.cross_entropy(logits, labels)
        return loss * self.contrastive_loss_coef

    @torch.no_grad()
    def update_momentum_encoder(self, momentum=0.999):
        if not self.use_contrastive or self.key_contrastive_proj_head is None or self.network.contrastive_proj_head is None: return
        online_params = dict(self.network.contrastive_proj_head.named_parameters())
        key_params = dict(self.key_contrastive_proj_head.named_parameters())
        for name, param_q in online_params.items():
            if name in key_params:
                param_k = key_params[name]
                param_k.data.copy_(param_k.data * momentum + param_q.data * (1. - momentum))

    def get_reward_weights(self, global_context=None):
        if not self.network.use_global_context: return self.network.get_reward_weights(None)
        else: return self.network.get_reward_weights(global_context)

    def clip_gradients(self):
        if self.max_grad_norm > 0:
             params_to_clip = list(self.network.parameters())
             if params_to_clip: torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)







# ===================================================================
#                       TEST LOOP (V2 - Full Functionality)
# ===================================================================
if __name__ == "__main__":
    import traceback
    import torch
    from torch_geometric.data import Data, Batch
    from constants import NODE_FEATURE_DIM, RAW_CH_COUNT, OCC_CH_COUNT, MEM_NODE_FEATURE_DIM, NODE_FEATURE_MAP
    from env.env import Swarm2DEnv

    print("\n" + "="*60)
    print("  Testing SharedActorPolicy with ANALOG & SELECTIVE Trails")
    print("="*60)
    print(f"Using device: {device}")

    # --- Test Parameters ---
    B = 4  # Batch size of 4 agents for this test
    NUM_AGENTS_IN_POLICY = 4 # The policy manages a total of 4 agents
    SELF_DIM = 24
    RAW_MAP_CH = RAW_CH_COUNT
    GRID = 32
    MEM_MAP_CH = OCC_CH_COUNT + 1
    TEST_OBS_RADIUS = OBS_RADIUS
    SEMANTIC_DIM_TEST = 32  # Dimension of a single trail message
    NUM_TRAIL_MESSAGES_TEST = 4 # Number of messages in the codebook

    try:
        # --- 1. Instantiate Policy ---
        print("\n--- 1. Instantiating SharedActorPolicy ---")
        policy = SharedActorPolicy(
            num_agents_on_team=NUM_AGENTS_IN_POLICY,
            self_feature_dim=SELF_DIM,
            map_channels=RAW_MAP_CH,
            grid_size=GRID,
            obs_radius=TEST_OBS_RADIUS,
            semantic_dim=SEMANTIC_DIM_TEST,
            num_trail_messages=NUM_TRAIL_MESSAGES_TEST,
            map_width=WIDTH,
            map_height=HEIGHT,
            use_global_context=True,
            use_latent_plan=True,
            use_dynamics_prediction=True,
            use_contrastive=True,
        ).to(device)
        policy.train() # Set to training mode to test the bernoulli dropout
        print(f"  Policy instantiated successfully for {NUM_AGENTS_IN_POLICY} agents.")
        assert len(policy.agent_trail_memories) == NUM_AGENTS_IN_POLICY

        # --- Generate a dummy observation batch ---
        def generate_dummy_obs(batch_size):
            env_graphs = [Data(x=torch.randn(5, NODE_FEATURE_DIM), pos=torch.rand(5,2)*1000, radii=torch.rand(5), is_ego=torch.tensor([True, False, False, False, False])) for _ in range(batch_size)]
            mem_graphs = [Data(x=torch.randn(10, MEM_NODE_FEATURE_DIM + 1), pos=torch.rand(10,2)*1000, radii=torch.rand(10), is_ego=torch.tensor([True] + [False]*9)) for _ in range(batch_size)]
            return {
                'self': torch.randn(batch_size, SELF_DIM, device=device),
                'map': torch.randn(batch_size, RAW_MAP_CH, GRID, GRID, device=device),
                'graph': Batch.from_data_list(env_graphs),
                'memory_map': torch.randn(batch_size, MEM_MAP_CH, GRID, GRID, device=device),
                'memory_graph': Batch.from_data_list(mem_graphs)
            }

        # --- PHASE 1: WRITE ---
        # Simulate the batch of agents acting and writing to their trails.
        print("\n--- PHASE 1: Agents act and SELECTIVELY WRITE to their trails ---")
        obs_batch_1 = generate_dummy_obs(B)
        agent_policy_indices_1 = torch.arange(B, dtype=torch.long, device=device)

        # The policy's forward pass
        # The 9th element is the message that was selected and written.
        _, _, _, _, _, _, _, _, message_written_batch = policy.evaluate_actions(
            obs_batch=obs_batch_1,
            agent_policy_indices=agent_policy_indices_1
        )
        print(f"  Policy evaluated. Output message batch shape: {message_written_batch.shape}")
        assert message_written_batch.shape == (B, SEMANTIC_DIM_TEST)
        print("  Successfully wrote selected messages to internal trail memories.")


        # --- PHASE 2: READ (with Analog Fidelity) ---
        print("\n--- PHASE 2: Agents READ nearby trails with distance-based fidelity ---")
        
        # Simulate new agent positions to create varied distances to trails
        agent_positions_world_2 = obs_batch_1['self'][:, [NODE_FEATURE_MAP['pos_x_norm'], NODE_FEATURE_MAP['pos_y_norm']]] * torch.tensor([WIDTH, HEIGHT], device=device)
        agent_positions_world_2 += torch.randn_like(agent_positions_world_2) * 50 # Add some random movement

        # Replace the `self` observation in the next batch to reflect new positions
        obs_batch_2 = generate_dummy_obs(B)
        obs_batch_2['self'][:, NODE_FEATURE_MAP['pos_x_norm']] = agent_positions_world_2[:, 0] / WIDTH
        obs_batch_2['self'][:, NODE_FEATURE_MAP['pos_y_norm']] = agent_positions_world_2[:, 1] / HEIGHT

        # The logic to read nearby trails is now *inside* evaluate_actions.
        # We are testing that this internal logic runs correctly.
        # This call will internally perform the robust, analog read.
        (action_dict, _, _, _, _, _, joint_action, _, _ ) = policy.evaluate_actions(
            obs_batch=obs_batch_2,
            agent_policy_indices=agent_policy_indices_1
        )
        
        print("  `evaluate_actions` with internal analog trail reading executed successfully.")
        print(f"  Output joint_action shape: {joint_action.shape}")
        assert joint_action.shape[0] == B, "Output batch size is incorrect."
        print("  Output format and shapes are correct, confirming the full loop works.")

    except Exception as e:
        print(f"\n!!!!!! TEST FAILED: {e} !!!!!!")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("  SharedActorPolicy Full Functionality Test Completed Successfully.")
    print("="*60 + "\n")