import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from torch_geometric.data import Data, Batch

# Import policies for proper isinstance checks
from policies.actors.MAAC.maac_attentionGNN import MAACPolicy
from policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
from policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy
from policies.critics.advanced_criticGNN import BaseCriticWrapper

MAAC_ROLES_GLOBAL = ["scout", "collector", "defender", "attacker"]
ROLE_NAME_TO_IDX_MAP = {name: idx for idx, name in enumerate(MAAC_ROLES_GLOBAL)}

from policies.critics.criticobservation import generate_critic_observation_from_tensors, merge_and_deduplicate_graphs, vectorize_reward

def get_data_list_from_graph(graph_batch: Optional[Batch]) -> List[Data]:
    """Convert a Batch object to a list of Data objects."""
    if graph_batch is None:
        return []
    if isinstance(graph_batch, Data):
        return [graph_batch]
    return graph_batch.to_data_list()


def unbatch_hidden_states_helper(
    hidden_state_batch: Optional[Union[Dict, Tuple, torch.Tensor]],
    indices: List[int]
) -> List[Optional[Union[Dict, Tuple, torch.Tensor]]]:
    """
    (V2 - Corrected) Unpacks a batched hidden state into a list of individual
    hidden states, keeping them as detached PyTorch Tensors on their original device.
    """
    if not indices:
        return []
    batch_size = len(indices) # Note: The batch size is now determined by the provided indices.

    if hidden_state_batch is None:
        return [None] * batch_size

    # Case 1: Dictionary (e.g., from MAACPolicy or NCA in dict format)
    if isinstance(hidden_state_batch, dict):
        unbatched = [{} for _ in range(batch_size)]
        for key, tensor_val in hidden_state_batch.items():
            # Skip non-tensor values or empty tensors
            if not isinstance(tensor_val, torch.Tensor) or tensor_val.numel() == 0:
                continue

            # Handle episodic cache tuple (keys, values)
            if key == 'episodic_cache' and isinstance(tensor_val, tuple):
                k_batch, v_batch = tensor_val
                if k_batch is not None and v_batch is not None:
                    for i, _ in enumerate(indices):
                        # Ensure we handle cases where cache might be empty for some items
                        unbatched[i][key] = (k_batch[i:i+1].detach(), v_batch[i:i+1].detach())
                continue # Move to the next key

            # Handle GRU-like states (Layers, Batch, Dim)
            if tensor_val.dim() == 3 and tensor_val.shape[1] == len(indices):
                for i, _ in enumerate(indices):
                    unbatched[i][key] = tensor_val[:, i:i+1].detach()
            # Handle other states (Batch, ...)
            elif tensor_val.dim() > 0 and tensor_val.shape[0] == len(indices):
                for i, _ in enumerate(indices):
                    unbatched[i][key] = tensor_val[i:i+1].detach()
        return unbatched

    # Case 2: Tuple (e.g., from NCA_PINSANPolicy)
    if isinstance(hidden_state_batch, tuple):
        unbatched = [[] for _ in range(batch_size)]
        for tensor_in_tuple in hidden_state_batch:
            if tensor_in_tuple is None:
                for i in range(batch_size): unbatched[i].append(None)
                continue

            if tensor_in_tuple.shape[0] == batch_size:
                for i in range(batch_size):
                    unbatched[i].append(tensor_in_tuple[i:i+1].detach())
        return [tuple(h) for h in unbatched]

    # Case 3: Simple Tensor (e.g., from SharedActorPolicy)
    if isinstance(hidden_state_batch, torch.Tensor) and hidden_state_batch.shape[0] == batch_size:
        return [hidden_state_batch[i:i+1].detach() for i in range(batch_size)]

    print(f"Warning (unbatch_hidden_states_helper): Unhandled hidden state type '{type(hidden_state_batch)}'. Returning list of None.")
    return [None] * batch_size


def batch_standardize_roles(
    role_list: List[Optional[Dict]],
    critic_ref: BaseCriticWrapper
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    (V4) Standardizes a BATCH of role info items for the UnifiedCriticCore.
    Handles MAAC dicts, NCA/Shared dicts, and potential None values from buffer.
    """
    if not role_list: return None, None
    batch_size = len(role_list)
    final_device = next(critic_ref.parameters()).device

    try:
        expected_cont_dim = critic_ref.critic_core.role_mlp[0].in_features
        num_discrete_roles = critic_ref.critic_core.internal_role_embedding.num_embeddings
    except Exception as e:
        print(f"FATAL ERROR: Could not get role dimensions from critic_ref: {e}"); return None, None

    processed_cont_list = []; processed_disc_idx_list = []

    # Unpack batched dictionary if needed
    if len(role_list) == 1 and isinstance(role_list[0], dict) and isinstance(role_list[0].get('continuous'), torch.Tensor) and role_list[0]['continuous'].shape[0] == batch_size:
        batched_dict = role_list[0]
        unbatched_cont = list(batched_dict['continuous'].split(1, dim=0))
        unbatched_disc = list(batched_dict['discrete_idx'].split(1, dim=0))
        role_list = [{'continuous': c, 'discrete_idx': d} for c, d in zip(unbatched_cont, unbatched_disc)]

    for item in role_list:
        cont_val = torch.zeros(expected_cont_dim, device=final_device); disc_idx = 0
        if isinstance(item, dict):
            raw_cont = item.get('continuous')
            if isinstance(raw_cont, (torch.Tensor, np.ndarray)):
                cont_val = torch.as_tensor(raw_cont, dtype=torch.float32, device=final_device).squeeze()
            
            raw_disc = item.get('discrete_idx')
            if raw_disc is None: raw_disc = ROLE_NAME_TO_IDX_MAP.get(item.get('agent_type'))
            if isinstance(raw_disc, (torch.Tensor, int, float, np.number)): disc_idx = int(raw_disc)
        
        if cont_val.shape != (expected_cont_dim,): cont_val = torch.zeros(expected_cont_dim, device=final_device)
        processed_cont_list.append(cont_val)
        processed_disc_idx_list.append(max(0, min(disc_idx, num_discrete_roles - 1)))
        
    try:
        role_embedding_cont_t = torch.stack(processed_cont_list); role_idx_disc_t = torch.tensor(processed_disc_idx_list, dtype=torch.long, device=final_device)
        return role_embedding_cont_t, role_idx_disc_t
    except Exception as e:
        print(f"Error stacking standardized role info: {e}"); return None, None

        
def batch_actor_hidden_states(
    hidden_state_list: List[Optional[Union[Dict, Tuple, torch.Tensor]]],
    batch_size: int,
    policy_ref: nn.Module,
    device: torch.device
) -> Optional[Union[Dict, Tuple, torch.Tensor]]:
    """
    Batches hidden states from a list, correctly handling the unique
    dictionary formats from each policy type (MAAC, NCA, Shared). Now uses a passed device.
    """
    # This is complex because hidden states can be dicts (MAAC) or tensors (others)
    # And they might be None for the first step.
    if all(h is None for h in hidden_state_list):
        return policy_ref.init_hidden(batch_size) # Let the policy handle it

    # Determine if we are dealing with a dictionary-based hidden state (like MAAC's)
    is_dict_hidden = isinstance(hidden_state_list[0], dict)

    if is_dict_hidden:
        # Batching for dictionary hidden states (GRU + Episodic Cache)
        default_h_state = policy_ref.init_hidden(1)
        
        # --- FIX: Access temporal_hidden_dim from the correct location for each policy ---
        if isinstance(policy_ref, MAACPolicy):
            hidden_dim = policy_ref.shared_actor_gnn.temporal_hidden_dim
        else: # Fallback for other potential dict-based policies
            # This part is tricky; we have to infer it from the initialized hidden state
            # This logic assumes the 'gru_hidden' part exists and its last dim is the hidden size
            temp_h = policy_ref.init_hidden(1)
            hidden_dim = temp_h.get('gru_hidden').shape[-1] if temp_h.get('gru_hidden') is not None else 256 # Fallback size

        default_gru = default_h_state.get('gru_hidden', torch.zeros(1, 1, hidden_dim, device=device))
        
        # Determine the shape of the episodic cache from a valid state
        k_shape, v_shape = None, None
        for h in hidden_state_list:
            if h is not None and 'episodic_cache' in h and isinstance(h['episodic_cache'], tuple):
                k_shape = h['episodic_cache'][0].shape
                v_shape = h['episodic_cache'][1].shape
                break

        if k_shape is None or v_shape is None:
            # Fallback if cache shape cannot be determined (e.g., if all are None)
            d_model = hidden_dim # Use the determined hidden_dim
            default_k = torch.empty(1, 0, d_model, device=device)
            default_v = torch.empty(1, 0, d_model, device=device)
        else:
            d_model = k_shape[-1] # Use the dimension of the cache tensors
            default_k = torch.empty(1, 0, d_model, device=device)
            default_v = torch.empty(1, 0, d_model, device=device)

        h_gru_list, cache_k_list, cache_v_list = [], [], []

        for h_state in hidden_state_list:
            if not isinstance(h_state, dict): h_state = default_h_state
            
            h_gru = h_state.get('gru_hidden', default_gru)
            if h_gru is not None and h_gru.dim() == 3:
                # Standardize to (1, D) from (1, 1, D) for easier processing
                h_gru = h_gru.squeeze(0)
            
            k, v = h_state.get('episodic_cache', (default_k, default_v))
            
            h_gru_list.append(h_gru)
            cache_k_list.append(k)
            cache_v_list.append(v)
        
        # Stack GRU states and move to correct device
        batched_h_gru = torch.stack(h_gru_list, dim=0).permute(1, 0, 2).to(device)
        
        # Pad and batch episodic memory cache
        max_len_k = max(k.size(1) for k in cache_k_list if k is not None and k.numel() > 0) if any(k is not None and k.numel() > 0 for k in cache_k_list) else 0

        if max_len_k > 0:
            padded_k_list = [F.pad(k, (0, 0, 0, max_len_k - k.size(1))) for k in cache_k_list]
            padded_v_list = [F.pad(v, (0, 0, 0, max_len_k - v.size(1))) for v in cache_v_list]
            batched_cache = (torch.cat(padded_k_list, dim=0).to(device), torch.cat(padded_v_list, dim=0).to(device))
        else:
            d_model = hidden_dim
            batched_cache = (torch.empty(batch_size, 0, d_model, device=device), torch.empty(batch_size, 0, d_model, device=device))
        
        return {'gru_hidden': batched_h_gru, 'episodic_cache': batched_cache}

    else:
        # --- Standard Tensor Batching (for NCA and SharedActor) ---
        # All hidden states are simple tensors. We just stack them.
        h_state_tensors = [h for h in hidden_state_list if h is not None]
        if not h_state_tensors:
             # If all were None, initialize a new batch
             return policy_ref.init_hidden(batch_size)
        
        # Assuming the hidden state tensor has shape [num_layers, batch_size, hidden_dim]
        # and for a single agent it's [num_layers, 1, hidden_dim]. We want to cat on the batch_dim (dim=1).
        # Squeeze and unsqueeze might be needed if shapes are inconsistent.
        # Let's assume the policy's init_hidden(1) gives the right shape.
        try:
            # Most robust: concatenate along the batch dimension (dim=1)
            return torch.cat(h_state_tensors, dim=1)
        except RuntimeError:
            # Fallback if shapes are weird (e.g., missing batch dim)
            return torch.stack(h_state_tensors, dim=1)

                
def batch_obs_dicts(
    list_of_dicts: List[Dict],
    device: torch.device
) -> Dict[str, Union[torch.Tensor, Batch]]:
    """
    Batches a list of observation dictionaries.
    Handles the case where some Data objects are on CPU (from buffer) and fallbacks
    must be created on the target device, preventing device mismatch errors.
    """
    if not list_of_dicts:
        return {}
    
    first_item = next((d for d in list_of_dicts if d), None)
    if first_item is None:
        return {}
    
    batch_size = len(list_of_dicts)
    batched_obs = {}
    
    # Identify which keys are for tensors and which are for graphs
    tensor_keys = [k for k, v in first_item.items() if isinstance(v, torch.Tensor)]
    graph_keys = [k for k, v in first_item.items() if isinstance(v, Data)]

    # --- Process Tensors ---
    for key in tensor_keys:
        # We can use torch.stack for efficiency if shapes are consistent
        try:
            example_tensor = first_item[key]
            # Create a list of tensors, providing a zero tensor as a default for missing ones
            tensor_list = [
                obs_dict.get(key, torch.zeros_like(example_tensor))
                for obs_dict in list_of_dicts
            ]
            # Stack and move to the target device in one operation
            batched_obs[key] = torch.stack(tensor_list).to(device)
        except Exception as e:
            # Fallback for tensors with varying shapes if necessary (less efficient)
            print(f"Warning: Could not stack tensors for key '{key}', possibly due to shape mismatch. Error: {e}")
            # This part would need a more complex padding/handling logic if needed
            pass

    # --- Process Graphs (Corrected Logic) ---
    for key in graph_keys:
        processed_graph_list = []
        for obs_dict in list_of_dicts:
            graph_val = obs_dict.get(key)
            
            if graph_val is not None and isinstance(graph_val, Data):
                processed_graph_list.append(graph_val.to(device))
            else:
                # If the graph is missing, create a new empty Data object directly on the target device.
                feat_dim = MEM_NODE_FEATURE_DIM + 1 if 'memory' in key else NODE_FEATURE_DIM
                fallback_graph = Data(x=torch.empty((0, feat_dim), device=device, dtype=torch.float32),
                                    edge_index=torch.empty((2, 0), device=device, dtype=torch.long),
                                    pos=torch.empty((0, 2), device=device, dtype=torch.float32),
                                    radii=torch.empty((0,), device=device, dtype=torch.float32))
                processed_graph_list.append(fallback_graph)

        # Now, batch the list of graphs, all of which are guaranteed to be on the same device.
        if processed_graph_list:
            batched_obs[key] = Batch.from_data_list(processed_graph_list)
            
    return batched_obs


def prepare_batch_for_update(
    raw_sequences: List[List[Tuple]],
    team_id: int,
    is_global_critic: bool,
    target_policy_ref: nn.Module,
    critic1_ref: BaseCriticWrapper,
    env_metadata_ref: Dict,
    device: torch.device
) -> Optional[Dict]:
    """
    (V10 - Caching & Optimized) Prepares a batch for critic and actor updates.
    - Aggregates team-wide views ONCE per unique timestep in the batch.
    - Uses pre-computed tensors directly from the replay buffer.
    """
    if not raw_sequences: return None
    batch_size = len(raw_sequences)
    sequence_len = len(raw_sequences[0])

    try:
        # --- 1. Flatten all transitions and identify unique timesteps to process ---
        flat_transitions = []
        unique_timesteps_to_process = set()
        for b in range(batch_size):
            for i in range(sequence_len):
                transition = raw_sequences[b][i]
                ep, step = transition[8], transition[9]
                flat_transitions.append({
                    "batch_idx": b, "seq_idx": i, "ep": ep, "step": step,
                    "transition_data": transition
                })
                unique_timesteps_to_process.add((ep, step, team_id))

        # --- 2. Process unique timesteps to create aggregated team views & critic obs ---
        team_view_cache = {}
        for ep, step, t_id in unique_timesteps_to_process:
            # Check if this timestep has already been processed and cached
            if (ep, step, t_id) in team_view_cache:
                continue

            # Find all transitions in the flat list that match this unique key
            transitions_at_this_step = [
                t["transition_data"] for t in flat_transitions
                if t["ep"] == ep and t["step"] == step
            ]
            if not transitions_at_this_step: continue
            
            # Use the first transition to get the packed env state for this timestep
            packed_env_state = transitions_at_this_step[0][12]
            
            # --- AGGREGATE MEMORY & LIVE DATA for this timestep ---
            team_memory_map, team_persistent_graph, team_raw_map, team_live_graph = None, None, None, None
            
            live_graphs_at_step = [t[17] for t in transitions_at_this_step if t[17] is not None]
            if live_graphs_at_step:
                # Assuming merge_and_deduplicate_graphs is defined elsewhere and handles device placement
                team_live_graph_batch = merge_and_deduplicate_graphs(live_graphs_at_step, device, connection_radius=env_metadata_ref['obs_radius'])
                if team_live_graph_batch.num_graphs > 0:
                    team_live_graph = team_live_graph_batch.to_data_list()[0]
                else:
                    team_live_graph = None # Explicitly set to None if empty
            
            if not is_global_critic:
                team_memory_map = transitions_at_this_step[0][14]
                team_raw_map = transitions_at_this_step[0][15]
                team_persistent_graph = transitions_at_this_step[0][16]

            # --- Generate and cache the single critic observation for this timestep ---
            all_pos = packed_env_state["all_pos"].to(device)
            all_feat = packed_env_state["all_feat"].to(device)
            all_types = packed_env_state["all_types"].to(device)
            all_teams = packed_env_state["all_teams"].to(device)
            all_radii = packed_env_state["all_radii"].to(device)

            critic_obs_for_step = generate_critic_observation_from_tensors(
                team_id, is_global_critic, all_pos, all_feat, all_types, all_teams, all_radii,
                packed_env_state["step"],
                team_memory_map.to(device) if team_memory_map is not None else None,
                team_persistent_graph.to(device) if team_persistent_graph is not None else None,
                team_raw_map.to(device) if team_raw_map is not None else None,
                team_live_graph.to(device) if team_live_graph is not None else None,
                env_metadata_ref, device
            )
            team_view_cache[(ep, step, t_id)] = critic_obs_for_step

        # --- 3. Generate critic observations for all transitions using cached views ---
        all_critic_obs = [None] * len(flat_transitions)
        for i, t_info in enumerate(flat_transitions):
            ep, step, t_id = t_info["ep"], t_info["step"], team_id
            all_critic_obs[i] = team_view_cache[(ep, step, t_id)]

        # --- 4. Re-assemble sequences and batch other data ---
        critic_obs_S_sequence = []
        critic_obs_Sn_sequence = []
        for i in range(sequence_len - 1): # For S_t
            obs_at_seq_step = [all_critic_obs[b * sequence_len + i] for b in range(batch_size)]
            critic_obs_S_sequence.append(batch_obs_dicts(obs_at_seq_step, device))
        for i in range(1, sequence_len): # For S_t+1
            obs_at_seq_step = [all_critic_obs[b * sequence_len + i] for b in range(batch_size)]
            critic_obs_Sn_sequence.append(batch_obs_dicts(obs_at_seq_step, device))

        # Batch other necessary data (from the last transition at step t, which is seq_len-2)
        last_transitions_in_S = [raw_sequences[b][-2] for b in range(batch_size)]
        final_transitions_in_Sn = [raw_sequences[b][-1] for b in range(batch_size)]
        
        actor_obs_Sn_live_list = [t[7] for t in final_transitions_in_Sn]
        full_actor_obs_Sn = [
            {**obs_live, "memory_map": t_mem[14], "memory_graph": t_mem[16]}
            for obs_live, t_mem in zip(actor_obs_Sn_live_list, last_transitions_in_S)
        ]
        actor_obs_Sn_live_batch = batch_obs_dicts(full_actor_obs_Sn, device)
        
        action_S_mov = torch.stack([torch.as_tensor(t[1]['movement'], dtype=torch.float32) for t in last_transitions_in_S])
        action_S_pickup = torch.tensor([t[1]['pickup'] for t in last_transitions_in_S], dtype=torch.float32).unsqueeze(1)
        
        # Batching raw actor observations for S_t for actor update
        actor_obs_S_list = [t[0] for t in last_transitions_in_S]

        processed_batch = {
            "critic_obs_S_sequence": critic_obs_S_sequence, "critic_obs_Sn_sequence": critic_obs_Sn_sequence,
            "actor_obs_Sn_live": actor_obs_Sn_live_batch,
            "actor_obs_S": actor_obs_S_list, # For actor update
            "rewards": torch.stack([vectorize_reward(t[2]) for t in last_transitions_in_S]).to(device),
            "dones": torch.tensor(np.array([t[3] for t in last_transitions_in_S]), dtype=torch.float32).view(-1, 1).to(device),
            "actions": torch.cat([action_S_mov, action_S_pickup], dim=1).to(device),
            "role_info_S": [t[4] for t in last_transitions_in_S], # Pass as list for batching later
            "h_actor_t": [t[5] for t in last_transitions_in_S],
            "h_actor_tp1": [t[6] for t in final_transitions_in_Sn],
            "agent_policy_indices": torch.tensor([t[11] for t in last_transitions_in_S], device=device, dtype=torch.long)
        }
        return processed_batch

    except Exception as e:
        print(f"ERROR during batch preparation for team {team_id}: {e}")
        traceback.print_exc()
        return None
        