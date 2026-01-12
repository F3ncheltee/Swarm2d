import sys
# advanced_critic.py

import os
import math
import traceback
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, List
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.distributions as D # For sampling roles
# --- Project-Specific Imports (for standalone testing) ---
# To make this file runnable, we assume the project root is in the path
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# Import constants
from constants import OCC_CH, OCC_CH_COUNT, RAW_CH_COUNT, REWARD_COMPONENT_KEYS, ROLE_EMBED_DIM, JOINT_ACTION_DIM

# --- PyTorch Geometric Imports ---
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    # Use scatter directly from PyG utils if possible, else fallback
    try:
        from torch_geometric.utils import scatter
    except ImportError:
        try:
            from torch_scatter import scatter
            print("Using torch_scatter.scatter")
        except ImportError:
            print("ERROR: Neither torch_geometric.utils.scatter nor torch_scatter found.")
            sys.exit(1)
    from torch_geometric.utils import to_dense_batch
except ImportError:
    print("Error: PyTorch Geometric not found.")
    print("Please install it: pip install torch-geometric torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Constants (Copied from train_multi_team.py for standalone testing)
###############################################################################
from constants import (
    NODE_TYPE, NUM_NODE_TYPES, NODE_FEATURE_MAP, NODE_FEATURE_DIM,
    MEM_NODE_FEAT_IDX, MEM_NODE_FEATURE_DIM
)
from env.observations import (
    PersistentGraphMemory, create_graph_edge_features,
    OccMapObservationManager
)
from policies.critics.criticobservation import generate_occ_map_for_critic
###############################################################################
# Constants for Standalone Testing
################################################################################
# It's good practice to define this here for clarity, even if also defined in train script
CRITIC_EDGE_FEATURE_DIM = 2 + (NODE_FEATURE_DIM * 2) + 2 # RelPos(2) + Features(N*2) + Team(2)
NUM_CERTAINTY_CHANNELS = 4 # obstacle, ally, enemy, resource
GLOBAL_CUE_DIM = 20
JOINT_ACTION_DIM = 3
MAAC_ROLES_GLOBAL = ["scout", "collector", "defender", "attacker"]
ROLE_NAME_TO_IDX_MAP = {name: idx for idx, name in enumerate(MAAC_ROLES_GLOBAL)}
NUM_REWARD_COMPONENTS = 18

###############################################################################
# Helper Modules (Keep as they were)
###############################################################################
#### MLP 
class MLP(nn.Module):
    """ Simple Multi-Layer Perceptron with LayerNorm and GELU """
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=2, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim, 32) # Ensure hidden_dim is reasonable
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


##################################################
########## TITAN architecture helper #############
##################################################
class PerceiverFusion(nn.Module):
    """
    (V2 - Multi-Modal) Fuses a variable-sized set of multi-modal tokens (from GNNs, CNNs, Cues)
    into a fixed-size latent representation using cross-attention.
    """
    def __init__(self, latent_dim: int, num_latents: int, perceiver_common_dim: int, num_heads: int, dropout_rate: float = 0.1): # <<< RENAMED input_dim
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            kdim=perceiver_common_dim,
            vdim=perceiver_common_dim,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attn_norm = nn.LayerNorm(latent_dim)
        self.cross_attn_ffn = MLP(latent_dim, latent_dim, hidden_dim=latent_dim * 2)
        
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(2)
        ])

    def forward(self, input_tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_tokens (Tensor): Concatenated tokens from all modalities. Shape: [B, num_tokens, input_dim]
            key_padding_mask (Tensor): Mask for the input tokens. Shape: [B, num_tokens]
        """
        B = input_tokens.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)

        # Cross-attend from latents (query) to the multi-modal input tokens (key, value)
        cross_attn_out, _ = self.cross_attention(
            query=latents,
            key=input_tokens,
            value=input_tokens,
            key_padding_mask=key_padding_mask
        )
        latents = self.cross_attn_norm(latents + cross_attn_out)
        latents = self.cross_attn_ffn(latents) # Added FFN for more capacity
        
        # Let the latent variables process and refine the information they've gathered
        for layer in self.self_attention_layers:
            latents = layer(latents)
            
        # Return the final processed latents, which can be pooled or used directly
        return latents

# ===================================================================
#         FINAL SpatioTemporalFeatureExtractor (Integrates Multi-Modal Perceiver)
# ===================================================================
class SpatioTemporalFeatureExtractor(nn.Module):
    """
    (V6 - Unified Graph) Encapsulates single-timestep feature extraction.
    - Processes a single, unified graph input which can be either a live FoV graph
      or a foveated memory graph with clustered nodes.
    - Dynamically selects the correct GNN node encoder based on feature dimensions.
    - Fuses GNN, CNN (from occ_map and optional raw_map), and Cue modalities
      using either a Perceiver or simple concatenation.
    """
    def __init__(self,
                 use_gnn: bool, use_raw_map_input: bool, raw_map_channels: int, occ_map_channels: int,
                 critic_grid_size: int, gnn_hidden_dim: int, gnn_layers: int, gnn_heads: int, edge_feature_dim: int,
                 map_cnn_output_dim: int, map_cnn_res_factor: int, certainty_proc_output_dim: int,
                 cue_dim: int, output_embedding_dim: int, dropout_rate: float,
                 # --- Perceiver Params ---
                 use_perceiver: bool,
                 perceiver_latent_dim: int,
                 perceiver_num_latents: int,
                 perceiver_common_dim: int
                ):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_raw_map_input = use_raw_map_input
        self.use_perceiver = use_perceiver

        # --- 1. GNN Pathway (Now Unified) ---
        self.gnn_node_encoder_live = MLP(NODE_FEATURE_DIM, gnn_hidden_dim)
        self.gnn_node_encoder_mem = MLP(MEM_NODE_FEATURE_DIM + 1, gnn_hidden_dim)
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=gnn_heads, concat=False, dropout=dropout_rate, edge_dim=edge_feature_dim)
            for _ in range(gnn_layers)
        ])
        live_edge_raw_dim = 8 + (2 * NODE_FEATURE_DIM)
        mem_edge_raw_dim = 8 + (2 * (MEM_NODE_FEATURE_DIM + 1))
        self.live_edge_mlp = MLP(live_edge_raw_dim, edge_feature_dim, hidden_dim=gnn_hidden_dim // 2)
        self.mem_edge_mlp = MLP(mem_edge_raw_dim, edge_feature_dim, hidden_dim=gnn_hidden_dim // 2)

        # --- 2. Map Pathway (Corrected) ---
        # Certainty channels are now the primary input, not an auxiliary one
        self.cnn_occ = CNNMapEncoder(
            input_channels=occ_map_channels, 
            output_dim=map_cnn_output_dim, 
            final_res_factor=map_cnn_res_factor
        )
        self.cnn_raw = CNNMapEncoder(
            input_channels=raw_map_channels, 
            output_dim=map_cnn_output_dim, 
            final_res_factor=map_cnn_res_factor
        ) if self.use_raw_map_input else None

        # --- 3. Cue Pathway (Unchanged) ---
        self.cue_processor = MLP(GLOBAL_CUE_DIM, cue_dim, hidden_dim=64)

        # --- 4. Fusion Mechanism ---
        if self.use_perceiver:
            self.gnn_token_proj = MLP(gnn_hidden_dim, perceiver_common_dim)
            # This MLP's input_dim MUST match the output of the map pathway.
            # The map pathway adds the cnn_occ and cnn_raw outputs, but their channel counts are the same,
            # so the final channel count is still map_cnn_output_dim.
            map_pathway_output_dim = self.cnn_occ.output_dim
            self.map_token_proj = MLP(map_pathway_output_dim, perceiver_common_dim)
            self.cue_token_proj = MLP(cue_dim, perceiver_common_dim)
            self.perceiver_fusion = PerceiverFusion(
                latent_dim=perceiver_latent_dim, num_latents=perceiver_num_latents,
                perceiver_common_dim=perceiver_common_dim, num_heads=4, dropout_rate=dropout_rate
            )
            self.latent_self_attention = nn.MultiheadAttention(
                embed_dim=perceiver_latent_dim, num_heads=4, batch_first=True, dropout=dropout_rate
            )
            self.latent_pool_norm = nn.LayerNorm(perceiver_latent_dim)
            self.fusion_mlp = MLP(perceiver_latent_dim, output_embedding_dim)
        else: # Concatenation-based fusion
            # Correcting this path as well for robustness
            map_pooled_dim = self.cnn_occ.output_dim
            fusion_dim = gnn_hidden_dim + map_pooled_dim + cue_dim
            self.fusion_mlp = MLP(fusion_dim, output_embedding_dim)

    def forward(self, obs: Dict, obs_radius: float) -> torch.Tensor:
        batch_size = obs['cues'].shape[0]
        device = obs['cues'].device

        # --- 1. GNN Path ---
        gnn_nodes = torch.empty(0, self.gnn_node_encoder_live.mlp[-1].out_features, device=device)
        gnn_batch_idx = torch.empty(0, dtype=torch.long, device=device)
        if self.use_gnn and 'env_graph' in obs and obs['env_graph'] is not None and obs['env_graph'].num_nodes > 0:
            graph_data = obs['env_graph'].to(device)
            is_memory_graph = graph_data.x.shape[1] > NODE_FEATURE_DIM
            node_encoder = self.gnn_node_encoder_mem if is_memory_graph else self.gnn_node_encoder_live
            edge_mlp = self.mem_edge_mlp if is_memory_graph else self.live_edge_mlp
            edge_feature_radius = obs_radius * 2.5 if is_memory_graph else obs_radius
            gnn_nodes = F.gelu(node_encoder(graph_data.x))
            edge_features = create_graph_edge_features(graph_data, edge_mlp, edge_feature_radius)
            for gnn_layer in self.gnn_layers:
                gnn_nodes = F.gelu(gnn_layer(gnn_nodes, graph_data.edge_index, edge_attr=edge_features))
            gnn_batch_idx = getattr(graph_data, 'batch', torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device))

        # --- 2. Map Path ---
        map_spatial_features = self.cnn_occ(obs['occ_map'])
        if self.use_raw_map_input and 'raw_map' in obs and obs['raw_map'] is not None:
             map_spatial_features = map_spatial_features + self.cnn_raw(obs['raw_map'])

        # --- 3. Cue Path ---
        cues_embedding = self.cue_processor(obs['cues'])

        # --- 4. Fusion ---
        if self.use_perceiver:
            gnn_tokens = self.gnn_token_proj(gnn_nodes)
            B, C, H, W = map_spatial_features.shape
            map_tokens_flat = map_spatial_features.flatten(2).permute(0, 2, 1)
            map_tokens = self.map_token_proj(map_tokens_flat)
            cue_tokens = self.cue_token_proj(cues_embedding).unsqueeze(1)
            dense_gnn, gnn_mask = to_dense_batch(gnn_tokens, gnn_batch_idx, batch_size=batch_size)
            all_tokens = torch.cat([dense_gnn, map_tokens, cue_tokens], dim=1)
            all_masks = torch.cat([
                gnn_mask, torch.ones_like(map_tokens[:, :, 0], dtype=torch.bool),
                torch.ones_like(cue_tokens[:, :, 0], dtype=torch.bool)
            ], dim=1)
            
            processed_latents = self.perceiver_fusion(all_tokens, key_padding_mask=~all_masks)

            # <<< IMPROVEMENT 1: REPLACE MEAN POOLING WITH ATTENTION POOLING >>>
            # The latents attend to themselves to determine relative importance.
            attn_pooled_latents, _ = self.latent_self_attention(
                query=processed_latents, key=processed_latents, value=processed_latents
            )
            # Add residual connection and layer norm
            pooled_latents = self.latent_pool_norm(processed_latents + attn_pooled_latents)
            
            # We still need a single vector, so we take the mean of the *attention-weighted* latents.
            final_embedding = self.fusion_mlp(pooled_latents.mean(dim=1))
            
        else: # Concatenation fusion (unchanged)
            gnn_pooled = pyg_nn.global_mean_pool(gnn_nodes, gnn_batch_idx)
            map_pooled = map_spatial_features.mean(dim=[2,3])
            final_gnn = torch.zeros(batch_size, gnn_pooled.shape[-1], device=device)
            unique_batch_indices = torch.unique(gnn_batch_idx)
            if unique_batch_indices.numel() > 0:
                final_gnn[unique_batch_indices] = gnn_pooled
            final_fusion_input = torch.cat([final_gnn, map_pooled, cues_embedding], dim=1)
            final_embedding = self.fusion_mlp(final_fusion_input)

        return final_embedding

### --- Titan Transformer Layer --- ###
class TitanTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, episodic_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        query = src
        if episodic_memory is not None and episodic_memory[0].numel() > 0:
            mem_k, mem_v = episodic_memory
            key = torch.cat([mem_k, src], dim=1)
            value = torch.cat([mem_v, src], dim=1)
        else:
            key, value = src, src

        attn_output, _ = self.self_attn(query, key, value)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class ResidualBlock(nn.Module):
    """Simple Residual Block for CNNs."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Dynamic group calculation with clamping
        num_groups1 = max(1, min(channels // 4, 32))
        self.norm1 = nn.GroupNorm(num_groups=num_groups1, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        num_groups2 = max(1, min(channels // 4, 32))
        self.norm2 = nn.GroupNorm(num_groups=num_groups2, num_channels=channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return F.gelu(out)

class CNNMapEncoder(nn.Module):
    """Encodes grid maps using strided convolutions and residual blocks."""
    def __init__(self, input_channels, base_filters=32, num_blocks=2, output_dim=64, final_res_factor=4):
        super().__init__()
        if input_channels <= 0:
             print(f"Warning (CNNMapEncoder): input_channels={input_channels} is invalid. Initializing as Identity.")
             self.encoder = nn.Identity()
             self.output_dim = output_dim # Still store output dim for identity case consistency?
             self.final_res_factor = 1 # No change in resolution
             return

        if not (final_res_factor > 0 and (final_res_factor & (final_res_factor - 1) == 0)):
            raise ValueError(f"final_res_factor must be a power of 2, got {final_res_factor}")
        self.final_res_factor = final_res_factor
        num_downsample = int(math.log2(final_res_factor))

        self.init_conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.init_norm = nn.GroupNorm(num_groups=max(1, min(base_filters // 4, 32)), num_channels=base_filters)

        self.layers = nn.ModuleList()
        current_filters = base_filters
        for i in range(num_downsample):
            self.layers.append(ResidualBlock(current_filters))
            for _ in range(num_blocks - 1):
                self.layers.append(ResidualBlock(current_filters))
            out_filters = current_filters * 2
            self.layers.append(nn.Conv2d(current_filters, out_filters, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.GroupNorm(num_groups=max(1, min(out_filters // 4, 32)), num_channels=out_filters))
            self.layers.append(nn.GELU())
            current_filters = out_filters

        for _ in range(num_blocks):
            self.layers.append(ResidualBlock(current_filters))

        self.final_conv = nn.Conv2d(current_filters, output_dim, kernel_size=1)
        self.final_norm = nn.GroupNorm(num_groups=max(1, min(output_dim // 4, 32)), num_channels=output_dim)
        self.encoder = nn.Sequential(
             self.init_conv, self.init_norm, nn.GELU(),
             *self.layers,
             self.final_conv, self.final_norm, nn.GELU() # <<< Moved GELU after norm
        )
        self.output_dim = output_dim

    def forward(self, x):
        # Pass through identity if initialized that way
        if isinstance(self.encoder, nn.Identity):
            # Need to return something of expected output shape, maybe zeros?
            B, _, H, W = x.shape
            out_H = H // self.final_res_factor
            out_W = W // self.final_res_factor
            return torch.zeros(B, self.output_dim, out_H, out_W, device=x.device, dtype=x.dtype)
        return self.encoder(x)

class CertaintyProcessor(nn.Module):
    # Keep as before
    """ Processes certainty channels to match spatial dimensions of CNN features. """
    def __init__(self, input_channels=NUM_CERTAINTY_CHANNELS, output_channels=16, base_filters=8, final_res_factor=4):
        super().__init__()
        if input_channels <= 0:
            print(f"Warning (CertaintyProcessor): input_channels={input_channels} is invalid. Initializing as Identity.")
            self.processor = nn.Identity()
            self.output_channels = output_channels # Store for identity case consistency?
            return

        if not (final_res_factor > 0 and (final_res_factor & (final_res_factor - 1) == 0)):
            raise ValueError(f"final_res_factor must be a power of 2, got {final_res_factor}")
        num_downsample = int(math.log2(final_res_factor))

        self.init_conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.init_norm = nn.GroupNorm(num_groups=max(1, min(base_filters // 2, 8)), num_channels=base_filters)

        self.layers = nn.ModuleList()
        current_filters = base_filters
        for _ in range(num_downsample):
            out_filters = current_filters * 2
            self.layers.append(nn.Conv2d(current_filters, out_filters, kernel_size=3, padding=1))
            self.layers.append(nn.GroupNorm(num_groups=max(1, min(out_filters // 2, 8)), num_channels=out_filters))
            self.layers.append(nn.GELU())
            self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2)) # Downsample
            current_filters = out_filters

        self.final_conv = nn.Conv2d(current_filters, output_channels, kernel_size=1)
        self.processor = nn.Sequential(
             self.init_conv, self.init_norm, nn.GELU(),
             *self.layers,
             self.final_conv
        )
        self.output_channels = output_channels

    def forward(self, x):
        if isinstance(self.processor, nn.Identity):
            B, _, H, W = x.shape
            # Calculate expected output size based on num_downsample (which is 0 for identity)
            out_H, out_W = H, W # No downsampling for identity
            return torch.zeros(B, self.output_channels, out_H, out_W, device=x.device, dtype=x.dtype)
        return self.processor(x)


###############################################################################
#                           UNIFIED CRITIC CORE
###############################################################################
class UnifiedCriticCore(nn.Module):
    def __init__(self,
                 # Feature extractor params
                 use_gnn: bool, use_raw_map_input: bool, raw_map_channels: int, occ_map_channels: int,
                 critic_grid_size: int, gnn_hidden_dim: int, gnn_layers: int, gnn_heads: int,
                 edge_feature_dim: int, map_cnn_output_dim: int, map_cnn_res_factor: int,
                 certainty_proc_output_dim: int,
                 use_perceiver: bool, perceiver_latent_dim: int, perceiver_num_latents: int,
                 perceiver_common_dim: int,
                 # Titan params
                 temporal_embedding_dim: int, titan_nhead: int, titan_ff_dim: int, titan_layers: int,
                 titan_memory_gate_threshold: float,
                 # Final fusion params
                 role_embedding_dim: int, num_discrete_roles: int, internal_discrete_role_embed_out_dim: int,
                 joint_action_dim: int, action_embedding_dim: int, num_reward_components: int,
                 cue_dim: int, fusion_mlp_dim: int, num_atoms: int, v_min: float, v_max: float, dropout_rate: float,
                 verbose: bool = True):
        super().__init__()
        self.num_reward_components = num_reward_components
        self.num_atoms = num_atoms
        self.temporal_embedding_dim = temporal_embedding_dim
        self.titan_mem_gate_thresh = titan_memory_gate_threshold

        # --- 1. Instantiate the Spatio-Temporal Feature Extractor (This part is correct) ---
        self.feature_extractor = SpatioTemporalFeatureExtractor(
            use_gnn=use_gnn, use_raw_map_input=use_raw_map_input, raw_map_channels=raw_map_channels,
            occ_map_channels=occ_map_channels, critic_grid_size=critic_grid_size,
            gnn_hidden_dim=gnn_hidden_dim, gnn_layers=gnn_layers, gnn_heads=gnn_heads, edge_feature_dim=edge_feature_dim,
            map_cnn_output_dim=map_cnn_output_dim, map_cnn_res_factor=map_cnn_res_factor,
            certainty_proc_output_dim=certainty_proc_output_dim,
            cue_dim=cue_dim, output_embedding_dim=temporal_embedding_dim, dropout_rate=dropout_rate,
            use_perceiver=use_perceiver,
            perceiver_latent_dim=perceiver_latent_dim,
            perceiver_num_latents=perceiver_num_latents,
            perceiver_common_dim=perceiver_common_dim
        )
        if verbose:
            print("--- Instantiated Spatio-Temporal Feature Extractor for Critic (Unified Graph V6) ---")

        # --- 2. Instantiate the Titan Transformer Encoder 
        self.titan_layers = nn.ModuleList([
            TitanTransformerEncoderLayer(d_model=temporal_embedding_dim, nhead=titan_nhead, dim_feedforward=titan_ff_dim, dropout=dropout_rate)
            for _ in range(titan_layers)
        ])
        if verbose:
            print(f"--- Instantiated Titan Transformer ({titan_layers} layers) for Critic ---")

        # --- 3. Role and Action Embeddings 
        self.role_mlp = nn.Sequential(nn.Linear(role_embedding_dim, 32), nn.GELU(), nn.LayerNorm(32))
        self.internal_role_embedding = nn.Embedding(num_discrete_roles, internal_discrete_role_embed_out_dim)
        self.action_mlp = nn.Sequential(nn.Linear(joint_action_dim, action_embedding_dim), nn.GELU(), nn.LayerNorm(action_embedding_dim))
        
        # --- 4. Final Fusion MLP 
        # Calculate the input dimension for the final MLP that processes the concatenated context.
        final_fusion_input_dim = temporal_embedding_dim + 32 + internal_discrete_role_embed_out_dim + action_embedding_dim
        self.fusion_mlp = MLP(final_fusion_input_dim, fusion_mlp_dim)
        
        # --- 5. Output Heads
        self.q_head = nn.Linear(fusion_mlp_dim, num_reward_components * num_atoms)
        self.confidence_head = nn.Sequential(nn.Linear(fusion_mlp_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.register_buffer('atoms', torch.linspace(v_min, v_max, num_atoms))

    def forward(self, *,
                obs_sequence: List[Dict],
                role_embedding: torch.Tensor,
                internal_discrete_role_idx: torch.Tensor,
                joint_action: torch.Tensor,
                episodic_memory_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # --- 1. Process sequence of observations to get state embeddings ---
        state_embeddings = []
        for obs_t in obs_sequence:
            obs_radius_t = obs_t.get('obs_radius', 50.0)
            state_embedding_t = self.feature_extractor(obs_t, obs_radius_t)
            state_embeddings.append(state_embedding_t)
        seq_embeddings = torch.stack(state_embeddings, dim=1)
        
        # --- 2. Pass the entire sequence through the Titan Transformer layers ---
        temporal_context = seq_embeddings
        for layer in self.titan_layers:
            temporal_context = layer(temporal_context, episodic_memory=episodic_memory_cache)
        final_context_vector = temporal_context[:, -1, :]
        
        # --- 3. IMPROVEMENT 2: GATED MEMORY UPDATE ---
        with torch.no_grad():
            new_episodic_keys, new_episodic_values = episodic_memory_cache if episodic_memory_cache is not None else (None, None)
            
            # We only gate the memory for the *last* embedding of the sequence
            last_state_embedding = seq_embeddings[:, -1, :].detach()

            # Check if it's significant enough to add
            should_add_to_memory = torch.ones(last_state_embedding.shape[0], dtype=torch.bool, device=last_state_embedding.device)
            if episodic_memory_cache is not None and episodic_memory_cache[0].numel() > 0:
                last_stored_key = episodic_memory_cache[0][:, -1, :]
                # Calculate cosine similarity
                similarity = F.cosine_similarity(last_state_embedding, last_stored_key)
                # Only add if similarity is below the threshold
                should_add_to_memory = similarity < self.titan_mem_gate_thresh

            # Prepare the new keys/values to be added
            keys_to_add = last_state_embedding[should_add_to_memory].unsqueeze(1)
            values_to_add = last_state_embedding[should_add_to_memory].unsqueeze(1)

            if new_episodic_keys is None or new_episodic_keys.numel() == 0:
                # If cache is empty, initialize it with the significant new states
                new_episodic_keys = keys_to_add
                new_episodic_values = values_to_add
            elif keys_to_add.numel() > 0:
                # Append only the significant new states to the existing cache
                # This requires careful handling of batch items where we don't add
                temp_keys = new_episodic_keys
                temp_values = new_episodic_values
                
                # A simple approach: just concatenate. A more complex one would handle different lengths per batch item.
                # For simplicity, we'll concatenate for all items if ANY item needs an update.
                if should_add_to_memory.any():
                    new_episodic_keys = torch.cat([temp_keys, last_state_embedding.unsqueeze(1)], dim=1)
                    new_episodic_values = torch.cat([temp_values, last_state_embedding.unsqueeze(1)], dim=1)

        # --- 4. Fuse the time-aware context with action/role info for the final Q-value ---
        role_cont_vec = self.role_mlp(role_embedding)
        role_disc_vec = self.internal_role_embedding(internal_discrete_role_idx)
        action_vec = self.action_mlp(joint_action)
        
        fusion_input = torch.cat([final_context_vector, role_cont_vec, role_disc_vec, action_vec], dim=1)
        fused_representation = self.fusion_mlp(fusion_input)
        
        # --- 5. Calculate final outputs ---
        q_logits = self.q_head(fused_representation).view(-1, self.num_reward_components, self.num_atoms)
        q_dist = F.softmax(q_logits, dim=-1)
        expected_q_vector = torch.sum(q_dist * self.atoms.view(1, 1, -1), dim=-1)
        confidence = self.confidence_head(fused_representation)
        
        return expected_q_vector, q_dist, confidence, (new_episodic_keys, new_episodic_values)
    
###############################################################################
#                           CRITIC WRAPPER
###############################################################################
class BaseCriticWrapper(nn.Module):
    """
    Wrapper for the UnifiedCriticCore. Handles optimization and provides a
    clean interface for training and evaluation.
    """
    def __init__(self, critic_core: UnifiedCriticCore, critic_lr: float = 1e-4, weight_decay: float = 1e-4, max_grad_norm: float = 1.0):
        super().__init__()
        if not isinstance(critic_core, UnifiedCriticCore):
             raise TypeError(f"Expected critic_core to be an instance of UnifiedCriticCore, got {type(critic_core)}")
        self.critic_core = critic_core
        self.episodic_memory_cache = None
        self.optimizer = torch.optim.AdamW(self.critic_core.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm

            
    def reset_cache(self):
        """Resets the episodic memory cache, e.g., at the start of a new episode."""
        self.episodic_memory_cache = None


    def _prepare_inputs(self,
                        obs: Dict[str, Union[torch.Tensor, np.ndarray, List[Data], int]],
                        role_info: Union[Dict, List, torch.Tensor],
                        joint_action: Union[torch.Tensor, np.ndarray]
                       ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares all inputs, standardizing and moving them to the correct device."""
        device = next(self.parameters()).device
        
        batch_size = joint_action.shape[0] if isinstance(joint_action, (torch.Tensor, np.ndarray)) and joint_action.ndim > 1 else (joint_action.shape[0] if isinstance(joint_action, (torch.Tensor, np.ndarray)) else 1)
        if batch_size == 0: raise ValueError("Cannot determine batch size from critic inputs.")
            
        # Standardize role info
        role_cont_t, role_disc_t = self._standardize_role_info(role_info, batch_size)
        if role_cont_t is None or role_disc_t is None:
            raise ValueError("Role info standardization failed.")

        # Process action
        if not isinstance(joint_action, torch.Tensor):
            joint_action = torch.tensor(joint_action, dtype=torch.float32)
        joint_action_t = joint_action.to(device)

        # Prepare observation dictionary for the core
        prepared_obs = {}
        for key, val in obs.items():
            if isinstance(val, (np.ndarray, torch.Tensor)):
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                prepared_obs[key] = val.to(device)
            elif isinstance(val, list) and val and isinstance(val[0], Data):
                prepared_obs[key] = Batch.from_data_list(val).to(device)
            else: # Pass other types (like team_id int) as is
                 prepared_obs[key] = val

        return prepared_obs, role_cont_t, role_disc_t, joint_action_t

    def _standardize_role_info(self, role_info: Dict, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (V2-Simplified) Standardizes role info for the critic.
        This version assumes the input `role_info` is already a dictionary
        with 'continuous' and 'discrete_idx' keys, prepared by the training loop.
        """
        final_device = next(self.critic_core.parameters()).device
        
        # --- Get Continuous Embedding ---
        role_cont_t = role_info.get('continuous')
        if not isinstance(role_cont_t, torch.Tensor):
            raise TypeError(f"Critic expects 'continuous' role info to be a tensor, got {type(role_cont_t)}")
        
        # --- Get Discrete Index ---
        role_disc_t = role_info.get('discrete_idx')
        if not isinstance(role_disc_t, torch.Tensor):
            raise TypeError(f"Critic expects 'discrete_idx' role info to be a tensor, got {type(role_disc_t)}")

        # --- Final Validation and Device Transfer ---
        if role_cont_t.shape[0] != batch_size:
            raise ValueError("Batch size mismatch for continuous role embedding.")
        if role_disc_t.shape[0] != batch_size:
            raise ValueError("Batch size mismatch for discrete role index.")

        return role_cont_t.to(final_device), role_disc_t.to(final_device)

    def forward(self,
                obs_sequence: List[Dict],
                role_info: Dict,
                joint_action: torch.Tensor,
                is_train: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        if not obs_sequence:
            raise ValueError("obs_sequence cannot be empty.")
        
        device = next(self.parameters()).device
        if is_train:
            self.critic_core.train()
            episodic_memory_input = None
        else:
            self.critic_core.eval()
            episodic_memory_input = self.episodic_memory_cache

        with torch.set_grad_enabled(is_train):
            # --- 1. Prepare Inputs ---
            # The last observation in the sequence is used for role/action, but the whole sequence is passed to the core
            _, role_cont_t, role_disc_t, joint_action_t = self._prepare_inputs(
                obs_sequence[-1], role_info, joint_action
            )
            prepared_obs_sequence = []
            for obs_t in obs_sequence:
                prepared_obs = {}
                for key, val in obs_t.items():
                    if isinstance(val, (torch.Tensor, Data)):
                        prepared_obs[key] = val.to(device)
                    else:
                        prepared_obs[key] = val
                # Ensure obs_radius is present for the feature extractor
                if 'obs_radius' not in prepared_obs:
                     prepared_obs['obs_radius'] = 50.0 # Default fallback
                prepared_obs_sequence.append(prepared_obs)

            # --- 2. Core Forward Pass ---
            expected_q_vector, q_dist, conf, new_cache = self.critic_core(
                obs_sequence=prepared_obs_sequence,
                role_embedding=role_cont_t,
                internal_discrete_role_idx=role_disc_t,
                joint_action=joint_action_t,
                episodic_memory_cache=episodic_memory_input
            )
            
            # --- 3. Update Cache (Inference only) ---
            if not is_train:
                self.episodic_memory_cache = new_cache
                
        return expected_q_vector, q_dist, conf, new_cache

    def state_dict(self):
        return {'network': self.critic_core.state_dict(), 'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict, **kwargs):
        self.critic_core.load_state_dict(state_dict['network'], strict=False)
        if 'optimizer' in state_dict and hasattr(self, 'optimizer'):
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            except Exception as e:
                print(f"Warn: Failed loading critic optim state: {e}")

    def train(self, mode=True): self.critic_core.train(mode)
    def eval(self): self.critic_core.eval()
    
    def clip_gradients(self):
        if self.max_grad_norm > 0:
             params = [p for p in self.critic_core.parameters() if p.grad is not None]
             if params: torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)











#################################################################################
#################################################################################
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Testing Unified Critic Full Pipeline (Global & Limited)")
    print("="*60)

    # --- Test Parameters ---
    TEST_B = 4
    TEST_SEQ_LEN = 8 # Use the constant from your training script
    TEST_GRID = 32
    TEST_ROLE_CONT_DIM = 16
    TEST_DEVICE = device
    TEST_OBS_RADIUS = 50.0

    # --- 1. Instantiation Test (Using the same config as the main script) ---
    print("\n--- 1. Instantiating Unified Critic Core ---")
    try:
        # This configuration must match the one in your main training script
        critic_config = {
            'use_perceiver': True, 'use_gnn': True, 'use_raw_map_input': True,
            'raw_map_channels': RAW_CH_COUNT, 'occ_map_channels': OCC_CH_COUNT,
            'critic_grid_size': TEST_GRID, 'gnn_hidden_dim': 64, 'gnn_layers': 2,
            'gnn_heads': 2, 'edge_feature_dim': 16, 'map_cnn_output_dim': 48,
            'map_cnn_res_factor': 4, 'certainty_proc_output_dim': 16, 'cue_dim': 16,
            'perceiver_latent_dim': 128, 'perceiver_num_latents': 32,
            'perceiver_common_dim': 64,
            'temporal_embedding_dim': 128, 'titan_nhead': 4, 'titan_ff_dim': 256,
            'titan_layers': 2, 'titan_memory_gate_threshold': 0.75,
            'role_embedding_dim': TEST_ROLE_CONT_DIM, 'num_discrete_roles': len(MAAC_ROLES_GLOBAL),
            'internal_discrete_role_embed_out_dim': 8, 'joint_action_dim': JOINT_ACTION_DIM,
            'action_embedding_dim': 16, 'num_reward_components': NUM_REWARD_COMPONENTS,
            'fusion_mlp_dim': 128, 'num_atoms': 51, 'v_min': -50.0, 'v_max': 150.0,
            'dropout_rate': 0.1,
        }

        critic_core = UnifiedCriticCore(**critic_config).to(TEST_DEVICE)
        critic_core.eval() # Set to evaluation mode for the test
        print("  Successfully instantiated UnifiedCriticCore.")

    except Exception as e:
        print(f"!!!!!! ERROR Instantiating Critic: {e} !!!!!!"); traceback.print_exc(); sys.exit(1)

    # --- 2. Helper function to generate dummy critic data ---
    def generate_dummy_critic_sequence(batch_size, seq_len, is_global: bool):
        """Generates a sequence of observation dicts mimicking the data pipeline."""
        obs_sequence = []
        for _ in range(seq_len):
            # --- Maps ---
            occ_map = torch.rand(batch_size, OCC_CH_COUNT, TEST_GRID, TEST_GRID, device=TEST_DEVICE)
            raw_map = torch.rand(batch_size, RAW_CH_COUNT, TEST_GRID, TEST_GRID, device=TEST_DEVICE)
            if is_global:
                # Global critic has perfect certainty
                occ_map[:, OCC_CH['certainty_obstacle']:] = 1.0
            else:
                # Limited critic has variable certainty
                occ_map[:, OCC_CH['certainty_obstacle']:] *= torch.rand_like(occ_map[:, OCC_CH['certainty_obstacle']:])

            # --- Graph (The Unified Foveated Input) ---
            graph_list = []
            for _ in range(batch_size):
                # We simulate a foveated graph by creating a graph with memory-style features
                num_nodes = random.randint(20, 40) if is_global else random.randint(10, 25)
                pos = torch.rand(num_nodes, 2) * 1000
                
                # Foveated graphs have the expanded feature set, including the 'count' feature
                features = torch.randn(num_nodes, MEM_NODE_FEATURE_DIM + 1)
                
                graph = Data(x=features, pos=pos, radii=torch.rand(num_nodes) * 10)
                graph.edge_index = pyg_nn.radius_graph(pos, r=TEST_OBS_RADIUS * 2.0, max_num_neighbors=32)
                graph_list.append(graph)
            
            batched_graph = Batch.from_data_list(graph_list)

            obs_sequence.append({
                "occ_map": occ_map,
                "raw_map": raw_map,
                "cues": torch.randn(batch_size, GLOBAL_CUE_DIM, device=TEST_DEVICE),
                "env_graph": batched_graph, # This is the key unified graph input
                "obs_radius": TEST_OBS_RADIUS
            })
        return obs_sequence

    # --- 3. Test GLOBAL Critic Pipeline ---
    print("\n--- 3. Testing GLOBAL Critic Pipeline ---")
    try:
        global_critic_sequence = generate_dummy_critic_sequence(TEST_B, TEST_SEQ_LEN, is_global=True)
        joint_action = torch.randn(TEST_B, JOINT_ACTION_DIM, device=TEST_DEVICE)
        role_embedding = torch.randn(TEST_B, TEST_ROLE_CONT_DIM, device=TEST_DEVICE)
        role_idx = torch.randint(0, len(MAAC_ROLES_GLOBAL), (TEST_B,), device=TEST_DEVICE)

        with torch.no_grad():
            q_vec, q_dist, conf, _ = critic_core(
                obs_sequence=global_critic_sequence,
                role_embedding=role_embedding,
                internal_discrete_role_idx=role_idx,
                joint_action=joint_action
            )

        print("  Global Critic forward pass successful.")
        assert q_vec.shape == (TEST_B, NUM_REWARD_COMPONENTS), f"Expected Q-vector shape mismatch. Got {q_vec.shape}"
        assert q_dist.shape == (TEST_B, NUM_REWARD_COMPONENTS, 51), f"Q-distribution shape mismatch. Got {q_dist.shape}"
        assert conf.shape == (TEST_B, 1), f"Confidence shape mismatch. Got {conf.shape}"
        print("  Output shapes verified for Global Critic.")

    except Exception as e:
        print(f"!!!!!! ERROR in Global Critic Pipeline Test: {e} !!!!!!"); traceback.print_exc()

    # --- 4. Test LIMITED Critic Pipeline ---
    print("\n--- 4. Testing LIMITED Critic Pipeline ---")
    try:
        limited_critic_sequence = generate_dummy_critic_sequence(TEST_B, TEST_SEQ_LEN, is_global=False)
        # Use the same action/role info for consistency
        joint_action = torch.randn(TEST_B, JOINT_ACTION_DIM, device=TEST_DEVICE)
        role_embedding = torch.randn(TEST_B, TEST_ROLE_CONT_DIM, device=TEST_DEVICE)
        role_idx = torch.randint(0, len(MAAC_ROLES_GLOBAL), (TEST_B,), device=TEST_DEVICE)

        with torch.no_grad():
            q_vec, q_dist, conf, _ = critic_core(
                obs_sequence=limited_critic_sequence,
                role_embedding=role_embedding,
                internal_discrete_role_idx=role_idx,
                joint_action=joint_action
            )

        print("  Limited Critic forward pass successful.")
        assert q_vec.shape == (TEST_B, NUM_REWARD_COMPONENTS), f"Expected Q-vector shape mismatch. Got {q_vec.shape}"
        assert q_dist.shape == (TEST_B, NUM_REWARD_COMPONENTS, 51), f"Q-distribution shape mismatch. Got {q_dist.shape}"
        assert conf.shape == (TEST_B, 1), f"Confidence shape mismatch. Got {conf.shape}"
        print("  Output shapes verified for Limited Critic.")

    except Exception as e:
        print(f"!!!!!! ERROR in Limited Critic Pipeline Test: {e} !!!!!!"); traceback.print_exc()

    print("\n" + "="*60)
    print("  Unified Critic Full Pipeline Test Completed.")
    print("="*60 + "\n")