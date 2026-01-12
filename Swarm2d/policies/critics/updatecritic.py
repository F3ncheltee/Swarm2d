import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
import traceback

from policies.critics.advanced_criticGNN import BaseCriticWrapper
from policies.actors.MAAC.maac_attentionGNN import MAACPolicy
from policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
from policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy
from training.log_utils import log_input_details


DEBUG_LOG_INPUTS = True
MAX_DEBUG_UPDATES = 5
max_grad_norm = 1.0
DEBUG_UPDATE_COUNT = 0


def update_critic_fn_sac(
    critic1: BaseCriticWrapper,
    critic2: BaseCriticWrapper,
    target_critic1: BaseCriticWrapper,
    target_critic2: BaseCriticWrapper,
    target_policy: Union[MAACPolicy, NCA_PINSANPolicy, SharedActorPolicy],
    log_alpha: torch.Tensor,
    processed_batch: Dict,
    gamma: float,
    is_weights: np.ndarray, # <<< ADDED: Importance sampling weights from the buffer
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """ Updates twin critics using a pre-processed batch and IS weights. Returns new TD errors and value tensors for logging. """
    
    global DEBUG_UPDATE_COUNT # Use the global counter
    if DEBUG_LOG_INPUTS and DEBUG_UPDATE_COUNT < MAX_DEBUG_UPDATES:
        # We need to know which team this update is for. This info isn't directly passed,
        # but we can infer it or just give it a generic name.
        team_id = "Unknown (Inferred from role)" # Placeholder
        is_global = "Unknown"
        if processed_batch['critic_obs_S_sequence'][0]['cues'].sum() > 0: # A simple heuristic
            is_global = "Probably Global"
        
        log_name = f"CRITIC INPUT BATCH[0] - Update #{DEBUG_UPDATE_COUNT} - {is_global}"
        
        # The observation is a sequence of dicts, so we pass the whole sequence
        log_input_details(log_name, processed_batch['critic_obs_S_sequence'], is_critic_obs=True)
        DEBUG_UPDATE_COUNT += 1

    try:
        # --- 1. Unpack the pre-processed batch dictionary ---
        critic_obs_S_sequence = processed_batch['critic_obs_S_sequence']
        critic_obs_Sn_sequence = processed_batch['critic_obs_Sn_sequence']
        actor_obs_Sn_live_batch = processed_batch['actor_obs_Sn_live']
        R_vector_t = processed_batch['rewards']
        D_t = processed_batch['dones']
        action_S_t = processed_batch['actions']
        role_cont_S_t = processed_batch['role_cont']
        role_disc_S_t = processed_batch['role_disc']
        batched_h_tp1_target = processed_batch['h_actor_tp1']
        
        if DEBUG_LOG_INPUTS and DEBUG_UPDATE_COUNT < MAX_DEBUG_UPDATES:
            print("--- Critic Update Batch Shapes ---")
            print(f"  Rewards (R_vector_t): {R_vector_t.shape}")
            print(f"  Dones (D_t): {D_t.shape}")
            print(f"  Actions (action_S_t): {action_S_t.shape}")
            print(f"  Role Continuous (role_cont_S_t): {role_cont_S_t.shape}")
            if isinstance(critic_obs_S_sequence, list) and len(critic_obs_S_sequence) > 0:
                print(f"  First Obs in Sequence (type): {type(critic_obs_S_sequence[0])}")

        # <<< ADDED: Convert IS weights to a tensor >>>
        is_weights_t = torch.tensor(is_weights, device=device, dtype=torch.float32)

        # Kwargs for policies (no change here)
        policy_kwargs = {'obs_batch': actor_obs_Sn_live_batch, 'hidden_state': batched_h_tp1_target}

    except KeyError as e:
        print(f"ERROR (Critic Update): Missing key in pre-processed batch: {e}"); traceback.print_exc()
        # Return a dummy error array with the correct shape
        return None, None, None, None, None, np.zeros(len(is_weights)), None, None, None

    # --- 2. Target Q Calculation (no change here) ---
    target_policy.eval(); target_critic1.eval(); target_critic2.eval()
    with torch.no_grad():
        policy_outputs = target_policy.evaluate_actions(**policy_kwargs)
        _, log_pi_next, _, hybrid_role_output_next, _, _, A_next_tensor, _, _ = policy_outputs
        
        q_target1_next, _, _, _ = target_critic1.forward(obs_sequence=critic_obs_Sn_sequence, role_info=hybrid_role_output_next, joint_action=A_next_tensor)
        q_target2_next, _, _, _ = target_critic2.forward(obs_sequence=critic_obs_Sn_sequence, role_info=hybrid_role_output_next, joint_action=A_next_tensor)
        
        q_target_next_min = torch.min(q_target1_next, q_target2_next)
        alpha = log_alpha.exp()
        log_pi_next_expanded = log_pi_next.expand_as(q_target_next_min)
        target_q_value = q_target_next_min - alpha * log_pi_next_expanded
        td_target = R_vector_t + gamma * (1.0 - D_t) * target_q_value

    # --- 3. Online Critic Evaluation & Loss (MODIFIED) ---
    critic1.train(); critic2.train()
    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
        role_info_S_t_dict = {'continuous': role_cont_S_t, 'discrete_idx': role_disc_S_t}
        q1_pred, _, _, _ = critic1.forward(obs_sequence=critic_obs_S_sequence, role_info=role_info_S_t_dict, joint_action=action_S_t, is_train=True)
        q2_pred, _, _, _ = critic2.forward(obs_sequence=critic_obs_S_sequence, role_info=role_info_S_t_dict, joint_action=action_S_t, is_train=True)
        
        # <<< MODIFIED: Calculate per-item error for PER update >>>
        td_error1 = (td_target.detach() - q1_pred).abs().mean(dim=1)
        td_error2 = (td_target.detach() - q2_pred).abs().mean(dim=1)
        new_priorities = ((td_error1 + td_error2) / 2.0).cpu().numpy()

        # <<< MODIFIED: Weight the losses by IS weights >>>
        loss1 = (is_weights_t * nn.functional.mse_loss(q1_pred, td_target.detach(), reduction='none').mean(dim=1)).mean()
        loss2 = (is_weights_t * nn.functional.mse_loss(q2_pred, td_target.detach(), reduction='none').mean(dim=1)).mean()
        total_critic_loss = loss1 + loss2

        if DEBUG_LOG_INPUTS and DEBUG_UPDATE_COUNT < MAX_DEBUG_UPDATES:
            print("--- Critic Update Losses ---")
            print(f"  Loss 1: {loss1.item():.4f}")
            print(f"  Loss 2: {loss2.item():.4f}")
            print(f"  Total Critic Loss: {total_critic_loss.item():.4f}")

    # --- 4. Optimization (no change here) ---
    critic1.optimizer.zero_grad(set_to_none=True); critic2.optimizer.zero_grad(set_to_none=True)
    if scaler: scaler.scale(total_critic_loss).backward()
    else: total_critic_loss.backward()

    params1 = [p for p in critic1.parameters() if p.grad is not None and p.requires_grad]
    params2 = [p for p in critic2.parameters() if p.grad is not None and p.requires_grad]
    all_params = params1 + params2
    pre_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in all_params if p.grad is not None]), 2.0).item() if all_params else 0.0
    if scaler:
        if params1: scaler.unscale_(critic1.optimizer)
        if params2: scaler.unscale_(critic2.optimizer)
    if params1: torch.nn.utils.clip_grad_norm_(params1, max_grad_norm)
    if params2: torch.nn.utils.clip_grad_norm_(params2, max_grad_norm)
    post_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in all_params if p.grad is not None]), 2.0).item() if all_params else 0.0
    if scaler:
        if params1: scaler.step(critic1.optimizer)
        if params2: scaler.step(critic2.optimizer)
    else:
        if params1: critic1.optimizer.step()
        if params2: critic2.optimizer.step()

    return loss1.item(), loss2.item(), ((q1_pred + q2_pred) / 2.0).mean().item(), pre_clip_norm, post_clip_norm, new_priorities, q1_pred.detach(), q2_pred.detach(), td_target.detach()
