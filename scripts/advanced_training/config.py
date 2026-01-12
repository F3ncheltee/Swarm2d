import torch
from Swarm2d.policies.actors.MAAC.maac_attentionGNN import MAAC_ROLES_GLOBAL
from Swarm2d.constants import RAW_CH_COUNT, OCC_CH_COUNT, ROLE_EMBED_DIM, JOINT_ACTION_DIM, NUM_REWARD_COMPONENTS, MOVEMENT_DIM, PICKUP_DIM

critic_config = {
    'use_perceiver': True,
    'use_gnn': True,
    'use_raw_map_input': True,
    'raw_map_channels': RAW_CH_COUNT,
    'occ_map_channels': OCC_CH_COUNT,
    'critic_grid_size': 32,
    'gnn_hidden_dim': 64,
    'gnn_layers': 2,
    'gnn_heads': 2,
    'edge_feature_dim': 16,
    'map_cnn_output_dim': 48,
    'map_cnn_res_factor': 4,
    'certainty_proc_output_dim': 16,
    'cue_dim': 16,
    
    # --- Perceiver Pathway Params ---
    'perceiver_latent_dim': 128,
    'perceiver_num_latents': 32,
    'perceiver_common_dim': 64,

    # --- Titan Transformer Params ---
    'temporal_embedding_dim': 128,
    'titan_nhead': 4,
    'titan_ff_dim': 256,
    'titan_layers': 2,
    'titan_memory_gate_threshold': 0.75,

    # --- Final Fusion Params ---
    'role_embedding_dim': ROLE_EMBED_DIM,
    'num_discrete_roles': len(MAAC_ROLES_GLOBAL),
    'internal_discrete_role_embed_out_dim': 8,
    'joint_action_dim': JOINT_ACTION_DIM,
    'action_embedding_dim': 16,
    'num_reward_components': NUM_REWARD_COMPONENTS,
    'fusion_mlp_dim': 128,
    'num_atoms': 51,
    'v_min': -50.0,
    'v_max': 150.0,
    'dropout_rate': 0.1,
}

maac_hyperparameters = {
    # Core Architecture & Learning
    "actor_lr": 3e-4,
    "d_model": 64,
    "candidate_in_dim": 16,
    "temporal_hidden_dim": 128,
    "role_embed_dim": 16,
    "dropout_rate": 0.1,
    "max_grad_norm": 1.0,

    # GNN & Communication
    "gnn_hidden_dim": 64,
    "gnn_layers": 2,
    "gnn_heads": 4,
    
    # Temporal Module (Titan)
    "titan_nhead": 4,
    "memory_length": 10,
}

nca_hyperparameters = {
    # Core Architecture
    "policy_lr": 3e-4,
    "h_dim": 64,
    "nca_iterations": 2,
    "belief_dim": 16,
    "readout_hidden_dim": 128,
    "dropout_rate": 0.0,
    
    # GNN & Communication
    "gnn_hidden_dim": 48,
    "gnn_layers": 2,
    "gnn_heads": 2,
    "edge_feature_dim": 16,
    "msg_dim": 32,
    
    # Specialized Modules
    "map_embed_dim": 32,
    "cnn_layers": 2,
    "role_emb_dim": 16,
    "sym_emb_dim": 8,
    "physics_feat_dim": 16,
    "logic_net_hidden_dim": 24,
    "modulation_dim": 64,
    "modulation_type": 'bias', # 'gate' or 'bias'
    "map_context_for_belief_dim": 16,
    
    # Decentralized Memory
    "decentralized_memory_slots": 16,
    "decentralized_memory_dim": 32,
    
    # Loss & Optimization
    "role_entropy_coef": 0.01,
    "aux_loss_coef": 0.1,
    "belief_loss_coef": 0.05,
    "max_grad_norm": 1.0,
    
    # Ablation Flags
    "use_decentralized_memory": True,
    "use_neighbor_attention": True,
    "use_neural_logic": True,
    "use_symbolic_layer": True,
    "use_physics_features": True,
    "use_dynamics_pred": True # This is a top-level flag for NCA_PINSANPolicy
}

shared_hyperparameters = {
    # Core Architecture & Learning
    "actor_lr": 3e-4,
    "d_model": 32,
    "hidden_dim": 64,
    "gru_hidden_dim": 32,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "role_emb_dim": 16,
    "final_hidden_dim": 64,

    # GNNs
    "comm_gnn_hidden_dim": 32,
    "mem_gnn_hidden_dim": 32,
    "mem_gnn_layers": 2,
    "mem_gnn_heads": 2,
    
    # Temporal & Memory Modules
    "temporal_layers": 2,
    "temporal_mem_size": 32,

    # Planning & Control
    "latent_plan_dim": 24,
    "llc_hidden_dim": 48,
    "semantic_dim": 32,
    "global_context_dim": 32,
    "role_gating_mlp_dim": 32,

    # Loss & Optimization
    "role_entropy_coef": 0.01,
    "aux_loss_coef": 0.1,
    "contrastive_loss_coef": 0.1,
    "contrastive_tau": 0.07,
    "max_grad_norm": 1.0,

    # Ablation Flags
    "use_contrastive": True,
    "use_adaptive_graph_comm": True,
    "use_temporal_gru": True,
    "use_latent_plan": True,
    "use_global_context": True,
    "use_dynamics_prediction": True,
}

BYPASS_HYPERSEARCH = True

if BYPASS_HYPERSEARCH:
    print("Bypassing hyperparameter search. Using default parameters.")
    # Load the updated DEFAULT_CRITIC_PARAMS
    default_critic_config = critic_config.copy()
    # --- Load Policy Defaults (keep as before) ---
    default_maac_config = maac_hyperparameters.copy() # Assuming 'default_params' holds combined defaults
    default_nca_config = nca_hyperparameters.copy()
    default_shared_config = shared_hyperparameters.copy()
    # --- Set best_params based on defaults ---
    best_maac_params = default_maac_config
    best_nca_params = default_nca_config
    best_sharedactor_params = shared_hyperparameters.copy()
else:
    # Placeholder for hyperparameter search logic
    print("Hyperparameter search not implemented, using defaults.")
    default_critic_config = critic_config.copy()
    best_maac_params = maac_hyperparameters.copy()
    best_nca_params = nca_hyperparameters.copy()
    best_sharedactor_params = shared_hyperparameters.copy()

# --- Configuration ---
print("CUDA available:", torch.cuda.is_available())