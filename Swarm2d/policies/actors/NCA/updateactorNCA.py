import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
from policies.critics.advanced_criticGNN import BaseCriticWrapper

DEBUG_ACTOR_UPDATE = True
DEBUG_UPDATE_COUNT_ACTOR = 0
MAX_DEBUG_UPDATES_ACTOR = 5

def update_actor_nca(
    actor_policy: NCA_PINSANPolicy,
    critic1: BaseCriticWrapper,
    log_alpha: torch.Tensor,
    processed_batch: Dict,
    batch_data: Dict[str, list],
    team_id: int,
    team_optimizers: Dict,
    env_metadata: Dict,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    **aux_kwargs
) -> Optional[Tuple[Optional[float], Dict, Optional[float], Optional[float], Optional[torch.Tensor]]]:
    """
    Standalone version of the NCA actor update function.
    """
    global DEBUG_UPDATE_COUNT_ACTOR
    try:
        current_batch_size = len(processed_batch['actions'])
        if current_batch_size == 0: return None, {}, None, None, None
        actor_obs_S_list = processed_batch['actor_obs_S']
        for obs_dict in actor_obs_S_list:
            if obs_dict and 'memory_graph' in obs_dict:
                obs_dict['graph'] = obs_dict['memory_graph']

        obs_batch_actor_S = batch_obs_dicts(processed_batch['actor_obs_S'], device)
        actor_obs_Sn_live = processed_batch['actor_obs_Sn_live']
        batched_h_t_input_actor = batch_actor_hidden_states(processed_batch['h_actor_t'], current_batch_size, actor_policy, device)
        critic_obs_sequence_S = processed_batch['critic_obs_S_sequence']
        agent_policy_indices_t = processed_batch['agent_policy_indices']
        
        # Get step number from the last transition in the sequence
        steps_in_batch = torch.tensor([seq[-1][9] for seq in batch_data['raw_transitions']], device=device)

    except Exception as e:
        print(f"ERROR (NCA Actor T{team_id}): Data Prep Failed: {e}"); traceback.print_exc()
        return None, {}, None, None, None

    actor_policy.train(); critic1.train()
    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
        policy_outputs = actor_policy.evaluate_actions(
            obs_batch=obs_batch_actor_S,
            hidden_state=batched_h_t_input_actor,
            agent_policy_indices=agent_policy_indices_t
        )
        action_dict, log_pi, _, hybrid_role_output, reward_weights, predicted_next, _, contrastive_embedding, belief_state_pred = policy_outputs
        actor_action_tensor = action_dict.get("joint_action")
        
        if DEBUG_ACTOR_UPDATE and DEBUG_UPDATE_COUNT_ACTOR < MAX_DEBUG_UPDATES_ACTOR:
            print(f"\n--- Actor Update Debug (NCAPolicy, Team {team_id}, Update #{DEBUG_UPDATE_COUNT_ACTOR}) ---")
            print(f"  - Batch Size: {current_batch_size}")
            print(f"  - Evaluate Actions Output Shapes:")
            print(f"    - action_tensor: {actor_action_tensor.shape}")
            print(f"    - log_pi: {log_pi.shape}")
            print(f"    - reward_weights: {reward_weights.shape if reward_weights is not None else 'None'}")
            
        role_cont_critic, role_disc_critic = batch_standardize_roles([hybrid_role_output] * current_batch_size, critic1)
        
        q_pred_vector_grad, _, _, _ = critic1.forward(
            obs_sequence=critic_obs_sequence_S,
            role_info={'continuous': role_cont_critic, 'discrete_idx': role_disc_critic},
            joint_action=actor_action_tensor, is_train=True
        )

        alpha = log_alpha.exp().detach()
        actor_loss = (-(q_pred_vector_grad * reward_weights).sum(dim=1) + alpha * log_pi.sum(dim=1)).mean()
        
        entropy_loss = actor_policy.compute_role_entropy_loss(hybrid_role_output.get('discrete'))
        imagination_loss = actor_policy.compute_imagination_loss(predicted_next, actor_obs_Sn_live['self'])
        
        with torch.no_grad():
            target_belief_t = generate_cues_from_graph(actor_obs_Sn_live['graph'], team_id, env_metadata, steps_in_batch + 1)[:, :actor_policy.network.belief_dim]
        
        belief_loss = F.mse_loss(belief_state_pred, target_belief_t) if belief_state_pred is not None else torch.tensor(0.0, device=device)
        
        total_loss = actor_loss + entropy_loss + imagination_loss + (belief_loss * aux_kwargs.get('belief_loss_coef', 0.05))
        detailed_losses = {'nca_total_loss': total_loss.item(), 'nca_sac_loss': actor_loss.item(), 'nca_entropy_loss': entropy_loss.item(), 'nca_imagine_loss': imagination_loss.item(), 'nca_belief_loss': belief_loss.item()}

        if DEBUG_ACTOR_UPDATE and DEBUG_UPDATE_COUNT_ACTOR < MAX_DEBUG_UPDATES_ACTOR:
            print(f"  - Q-value from Critic shape: {q_pred_vector_grad.shape}")
            print(f"  - Losses:")
            for name, value in detailed_losses.items():
                print(f"    - {name}: {value:.4f}")
            DEBUG_UPDATE_COUNT_ACTOR += 1

    optim = team_optimizers[team_id]['policy']
    optim.zero_grad(set_to_none=True)
    if scaler: scaler.scale(total_loss).backward()
    else: total_loss.backward()
    
    params_to_clip = [p for p in actor_policy.parameters() if p.grad is not None]
    pre_clip_norm = np.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in params_to_clip)) if params_to_clip else 0.0
    if scaler: scaler.unscale_(optim)
    actor_policy.clip_gradients()
    post_clip_norm = np.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in params_to_clip)) if params_to_clip else 0.0
    if scaler: scaler.step(optim)
    else: optim.step()
    
    return total_loss.item(), detailed_losses, pre_clip_norm, post_clip_norm, log_pi.detach()
