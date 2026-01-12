import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from policies.actors.MAAC.maac_attentionGNN import MAACPolicy
from policies.critics.advanced_criticGNN import BaseCriticWrapper
from trainingCustom.batchhelpers import batch_obs_dicts, batch_actor_hidden_states

DEBUG_ACTOR_UPDATE = True
DEBUG_UPDATE_COUNT_ACTOR = 0
MAX_DEBUG_UPDATES_ACTOR = 5

def update_actor_maac(
    actor_policy: MAACPolicy,
    critic1: BaseCriticWrapper,
    log_alpha: torch.Tensor,
    processed_batch: Dict,
    team_id: int,
    team_optimizers: Dict, # This is the top-level dict of optimizers for all teams
    env_metadata: Dict,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler]
) -> Optional[Tuple[Optional[float], Dict, Optional[float], Optional[float], Optional[torch.Tensor]]]:

    global DEBUG_UPDATE_COUNT_ACTOR
    # --- Data Preparation ---
    obs_batch_actor_S = batch_obs_dicts(processed_batch['actor_obs_S'], device)
    h_actor_t_batched = batch_actor_hidden_states(processed_batch['h_actor_t'], len(processed_batch['actor_obs_S']), actor_policy, device)
    agent_types_in_batch = [r.get('agent_type', actor_policy.agent_types[0]) for r in processed_batch['role_info_S']]

    # --- Forward Pass ---
    actor_policy.train(); critic1.train() # Set to train mode
    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
        policy_outputs = actor_policy.evaluate_actions(
            obs_batch=obs_batch_actor_S,
            hidden_state=h_actor_t_batched,
            all_agent_types_in_team=agent_types_in_batch
        )
        action_dict, log_pi, _, hybrid_role_output, reward_weights, _, _, _, _ = policy_outputs
        actor_action_tensor = action_dict.get("joint_action")

        if DEBUG_ACTOR_UPDATE and DEBUG_UPDATE_COUNT_ACTOR < MAX_DEBUG_UPDATES_ACTOR:
            print(f"\n--- Actor Update Debug (MAACPolicy, Team {team_id}, Update #{DEBUG_UPDATE_COUNT_ACTOR}) ---")
            print(f"  - Batch Size: {len(agent_types_in_batch)}")
            print(f"  - Evaluate Actions Output Shapes:")
            print(f"    - action_tensor: {actor_action_tensor.shape}")
            print(f"    - log_pi: {log_pi.shape}")
            print(f"    - reward_weights: {reward_weights.shape if reward_weights is not None else 'None'}")

        q_pred_vector_grad, _, _, _ = critic1.forward(
            obs_sequence=processed_batch['critic_obs_S_sequence'],
            role_info=hybrid_role_output,
            joint_action=actor_action_tensor, is_train=True
        )

        alpha = log_alpha.exp().detach()
        actor_loss = (-(q_pred_vector_grad * reward_weights).sum(dim=1) + alpha * log_pi.sum(dim=1)).mean()
        detailed_losses = {'MAAC_actor_loss_SAC_avg': actor_loss.item()}

        if DEBUG_ACTOR_UPDATE and DEBUG_UPDATE_COUNT_ACTOR < MAX_DEBUG_UPDATES_ACTOR:
            print(f"  - Q-value from Critic shape: {q_pred_vector_grad.shape}")
            print(f"  - MAAC Actor Loss: {actor_loss.item():.4f}")
            DEBUG_UPDATE_COUNT_ACTOR += 1

    # --- Corrected Optimization ---
    # Retrieve the dictionary of optimizers for this specific MAAC team
    maac_team_optimizers = team_optimizers.get(team_id)
    if not isinstance(maac_team_optimizers, dict):
        print(f"FATAL ERROR (update_actor_maac): Optimizers for MAAC team {team_id} not found or not a dict.")
        return None, {}, None, None, None

    # Zero gradients for all optimizers associated with roles present in this batch
    unique_roles_in_batch = set(agent_types_in_batch)
    for role in unique_roles_in_batch:
        if role in maac_team_optimizers:
            maac_team_optimizers[role].zero_grad(set_to_none=True)

    # Backward pass computes gradients for all involved actor networks
    if scaler: scaler.scale(actor_loss).backward()
    else: actor_loss.backward()

    # Clip gradients for all actors that were part of this forward pass
    params_to_clip = [p for role in unique_roles_in_batch if role in actor_policy.actors for p in actor_policy.actors[role].parameters() if p.grad is not None]
    pre_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in params_to_clip if p.grad is not None]), 2.0).item() if params_to_clip else 0.0

    if scaler:
        for role in unique_roles_in_batch:
            if role in maac_team_optimizers:
                scaler.unscale_(maac_team_optimizers[role])

    if params_to_clip: torch.nn.utils.clip_grad_norm_(params_to_clip, actor_policy.max_grad_norm)
    post_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in params_to_clip if p.grad is not None]), 2.0).item() if params_to_clip else 0.0

    # Step only the optimizers for the roles that were in the batch
    for role in unique_roles_in_batch:
        if role in maac_team_optimizers:
            if scaler: scaler.step(maac_team_optimizers[role])
            else: maac_team_optimizers[role].step()

    return actor_loss.item(), detailed_losses, pre_clip_norm, post_clip_norm, log_pi.detach()