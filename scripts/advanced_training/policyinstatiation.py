import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.policies.actors.MAAC.maac_attentionGNN import MAACPolicy
from Swarm2d.policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
from Swarm2d.policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy
from Swarm2d.policies.critics.advanced_criticGNN import UnifiedCriticCore, BaseCriticWrapper

# Default hyperparameters
best_maac_params = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'num_layers': 3
}

best_nca_params = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'num_layers': 3
}

best_sharedactor_params = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'num_layers': 3
}

critic_config = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'num_layers': 3
}

default_critic_config = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'num_layers': 3
}

def instantiate_maac_policy(hyperparams: Dict, env: Swarm2DEnv, device: torch.device, verbose: bool = True) -> Tuple[MAACPolicy, Dict]:
    """Instantiates the MAACPolicy and its optimizers."""
    from Swarm2d.constants import RAW_CH_COUNT, OCC_CH_COUNT, REWARD_COMPONENT_KEYS, ROLE_EMBED_DIM
    
    # Extract action dimensions from the dictionary action space
    movement_dim = env.action_space['movement'].shape[0]  # 2
    pickup_dim = env.action_space['pickup'].n  # 3
    
    policy = MAACPolicy(
        self_feature_dim=env.self_obs_dim,
        raw_map_channels=RAW_CH_COUNT,
        map_grid_size=32,
        movement_dim=movement_dim,
        num_reward_components=len(REWARD_COMPONENT_KEYS),
        agent_types=["scout", "collector", "defender", "attacker"],
        d_model=hyperparams.get('hidden_dim', 256),
        candidate_in_dim=hyperparams.get('hidden_dim', 256),
        temporal_hidden_dim=hyperparams.get('hidden_dim', 256),
        role_embed_dim=ROLE_EMBED_DIM,
        gnn_hidden_dim=hyperparams.get('hidden_dim', 256),
        gnn_layers=hyperparams.get('num_layers', 3),
        gnn_heads=4,
        titan_nhead=4,
        memory_length=10,
        obs_radius=env.metadata.get('obs_radius', 50.0),
        mem_connection_radius=env.metadata.get('obs_radius', 50.0) * 2.0,
        actor_lr=hyperparams.get('learning_rate', 3e-4),
        max_grad_norm=1.0,
        dropout_rate=0.1,
        verbose=verbose
    ).to(device)
    
    # Manually flatten GRU parameters after moving the entire policy to the device.
    # With the refactor, the GRU is in the shared_actor_gnn.
    if hasattr(policy, 'shared_actor_gnn') and hasattr(policy.shared_actor_gnn, 'gru'):
        policy.shared_actor_gnn.gru.flatten_parameters()
            
    # The optimizers are now a dictionary within the policy
    optimizers = {
        'shared_gnn_optimizer': policy.shared_gnn_optimizer,
        'role_optimizers': policy.role_optimizers
    }
    return policy, optimizers

def instantiate_nca_policy(hyperparams: Dict, env: Swarm2DEnv, device: torch.device, num_agents_on_team: int, verbose: bool = True) -> Tuple[NCA_PINSANPolicy, Dict]:
    """Instantiates the NCA_PINSANPolicy and its optimizers."""
    from Swarm2d.constants import RAW_CH_COUNT, OCC_CH_COUNT, REWARD_COMPONENT_KEYS, ROLE_EMBED_DIM
    
    # Extract action dimensions from the dictionary action space
    movement_dim = env.action_space['movement'].shape[0]  # 2
    pickup_dim = env.action_space['pickup'].n  # 3
    
    policy = NCA_PINSANPolicy(
        num_agents_on_team=num_agents_on_team,
        action_dim=movement_dim,
        pickup_dim=pickup_dim,
        self_feature_dim=env.self_obs_dim,
        grid_size=32,
        map_channels=RAW_CH_COUNT,
        memory_map_channels=OCC_CH_COUNT,
        h_dim=hyperparams.get('hidden_dim', 256),
        nca_iterations=2,
        msg_dim=32,
        map_embed_dim=32,
        cnn_layers=2,
        sym_dim=4,
        sym_emb_dim=8,
        physics_feat_dim=16,
        gnn_hidden_dim=48,
        gnn_layers=hyperparams.get('num_layers', 3),
        gnn_heads=2,
        edge_feature_dim=16,
        logic_net_hidden_dim=24,
        modulation_dim=hyperparams.get('hidden_dim', 256),
        modulation_type='bias',
        belief_dim=10,
        map_context_for_belief_dim=16,
        readout_hidden_dim=128,
        num_discrete_roles=4,
        role_emb_dim=ROLE_EMBED_DIM,
        num_reward_components=len(REWARD_COMPONENT_KEYS),
        use_decentralized_memory=True,
        decentralized_memory_slots=16,
        decentralized_memory_dim=hyperparams.get('hidden_dim', 256),
        query_input_dim=hyperparams.get('hidden_dim', 256),
        use_dynamics_prediction=True,
        use_contrastive=True,
        obs_radius=env.metadata.get('obs_radius', 50.0),
        comm_radius=env.metadata.get('obs_radius', 150.0),
        mem_connection_radius=env.metadata.get('obs_radius', 50.0) * 2.0,
        policy_lr=hyperparams.get('learning_rate', 3e-4),
        max_grad_norm=1.0,
        aux_loss_coef=0.1,
        role_entropy_coef=0.01,
        contrastive_loss_coef=0.1,
        contrastive_tau=0.07,
        verbose=verbose
    ).to(device)

    # Manually flatten GRU parameters after moving the policy to the device
    # GRUCell objects do not have a `flatten_parameters` method, so we don't call it.
    # The warning only applies to multi-layer nn.GRU modules.

    optimizer = torch.optim.AdamW(policy.parameters(), lr=hyperparams.get('actor_lr', 1e-4))
    return policy, {'policy': optimizer}

def instantiate_shared_policy(hyperparams: Dict, env: Swarm2DEnv, device: torch.device, num_agents_on_team: int, verbose: bool = True) -> Tuple[SharedActorPolicy, Dict]:
    """Instantiates the SharedActorPolicy and its optimizers."""
    from Swarm2d.constants import RAW_CH_COUNT, OCC_CH_COUNT, REWARD_COMPONENT_KEYS, ROLE_EMBED_DIM
    
    # Extract action dimensions from the dictionary action space
    movement_dim = env.action_space['movement'].shape[0]  # 2
    pickup_dim = env.action_space['pickup'].n  # 3
    
    policy = SharedActorPolicy(
        num_agents_on_team=num_agents_on_team,
        action_dim=movement_dim,
        pickup_dim=pickup_dim,
        self_feature_dim=env.self_obs_dim,
        grid_size=32,
        map_channels=RAW_CH_COUNT,
        max_neighbors_comm=7,
        memory_map_channels=OCC_CH_COUNT,
        comm_gnn_hidden_dim=32,
        mem_gnn_hidden_dim=32,
        mem_gnn_layers=2,
        mem_gnn_heads=2,
        d_model=32,
        hidden_dim=hyperparams.get('hidden_dim', 256),
        gru_hidden_dim=32,
        num_heads=4,
        dropout_rate=0.1,
        num_roles=4,
        role_emb_dim=ROLE_EMBED_DIM,
        global_context_dim=32,
        final_hidden_dim=64,
        temporal_layers=2,
        temporal_mem_size=32,
        external_mem_slots=8,
        external_mem_read_heads=2,
        latent_plan_dim=32,
        llc_hidden_dim=64,
        num_reward_components=len(REWARD_COMPONENT_KEYS),
        semantic_dim=32,
        contrastive_embedding_dim=128,
        role_gating_mlp_dim=32,
        edge_feature_dim=16,
        obs_radius=env.metadata.get('obs_radius', 50.0),
        mem_connection_radius=env.metadata.get('obs_radius', 50.0) * 2.0,
        num_trail_messages=4,
        use_contrastive=True,
        use_external_memory=True,
        use_memory_attention=True,
        use_adaptive_graph_comm=True,
        use_temporal_gru=True,
        use_latent_plan=True,
        use_global_context=True,
        use_dynamics_prediction=True,
        map_width=1000.0,
        map_height=1000.0,
        aux_loss_coef=0.1,
        actor_lr=hyperparams.get('learning_rate', 3e-4),
        role_entropy_coef=0.01,
        max_grad_norm=1.0,
        contrastive_loss_coef=0.1,
        contrastive_tau=0.07,
        verbose=verbose
    ).to(device)

    # Manually flatten GRU parameters after moving the policy to the device
    if hasattr(policy.network, 'temporal_module') and hasattr(policy.network.temporal_module, 'gru'):
        policy.network.temporal_module.gru.flatten_parameters()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=hyperparams.get('actor_lr', 1e-4))
    return policy, {'policy': optimizer}