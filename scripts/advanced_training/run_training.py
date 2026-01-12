import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import pickle
import time
import cProfile
import pstats
import io
import gc
import traceback
import random
import signal
import sys
from collections import deque
from typing import Dict, Union, Optional, List, Tuple
from tqdm import tqdm

from torch_geometric.data import Data, Batch

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.env.observations import ActorMapState

# Global variable to store profiler for signal handling
profiler_cprofile_global = None

def save_profiling_results(profiler, log_dir=".", episode=None):
    """Save profiling results to file and print summary"""
    if not profiler:
        return
    
    try:
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        
        # Print top 20 functions
        print("\n" + "="*40 + f" cProfile Results (Episode {episode}) " + "="*40)
        ps.print_stats(20)
        print(s.getvalue())
        
        # Save to file
        episode_suffix = f"_ep{episode}" if episode is not None else ""
        profile_path = os.path.join(log_dir, f"training_cProfile{episode_suffix}.prof")
        ps.dump_stats(profile_path)
        print(f"cProfile stats saved to {profile_path}")
        
    except Exception as e:
        print(f"Warning: Could not save cProfile stats: {e}")

def signal_handler(signum, frame):
    """Handle interruption signals to save profiling results"""
    print(f"\n\nReceived signal {signum}. Saving profiling results before exit...")
    if profiler_cprofile_global:
        save_profiling_results(profiler_cprofile_global, episode="interrupted")
    print("Profiling results saved. Exiting...")
    sys.exit(0)
from Swarm2d.policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy, TrailManager
from Swarm2d.policies.actors.MAAC.maac_attentionGNN import MAACPolicy
from Swarm2d.policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
from Swarm2d.policies.critics.advanced_criticGNN import UnifiedCriticCore, BaseCriticWrapper
from Swarm2d.policies.critics.ReplayBuffer import ReplayBuffer
from Swarm2d.policies.critics.updatecritic import update_critic_fn_sac
from Swarm2d.policies.actors.MAAC.updateactorMAAC import update_actor_maac
from Swarm2d.policies.actors.NCA.updateactorNCA import update_actor_nca
from Swarm2d.policies.actors.SHARED.updateactorShared import update_actor_shared

from batchhelpers import (
    batch_obs_dicts,
    batch_actor_hidden_states,
    unbatch_hidden_states_helper,
    prepare_batch_for_update,
    get_data_list_from_graph
)
from Swarm2d.training.checkpointing import find_latest_checkpoint
from Swarm2d.training.log_utils import Logger
from Swarm2d.training.PlateauManager import TeamPlateauManager, EpisodeEarlyStopper
from Swarm2d.training.curriculum_learning import CurriculumManager  # Import the new CurriculumManager
from policyinstatiation import instantiate_maac_policy, instantiate_shared_policy, instantiate_nca_policy, best_maac_params, best_nca_params, best_sharedactor_params
from config import critic_config, default_critic_config

from Swarm2d.training.utils import set_seeds, assign_conceptual_teams_for_episode, soft_update_targets
from Swarm2d.training.observation_debug import observation_debugger

from Swarm2d.constants import *

# ===================================================================
#               EVALUATION PHASE FUNCTION
# ===================================================================

def run_evaluation_phase(
    eval_episode_num_global: int,
    num_eval_episodes: int,
    env: 'Swarm2DEnv',
    team_policies: Dict[int, Union['MAACPolicy', 'NCA_PINSANPolicy', 'SharedActorPolicy']],
    logger: Optional['Logger'],
    device: torch.device,
    log_run_name_eval: str
):
    """
    Runs a full evaluation phase, mirroring the logic of the training loop.
    - Uses event-driven, cached memory generation for efficiency.
    - Implements action repeat for consistency with training.
    - Correctly handles all policy-specific logic (MAAC roles, SharedActor trails).
    """
    print(f"\n--- Starting Evaluation Phase (Corresponds to Training Ep {eval_episode_num_global}) ---")

    # Store original training modes and set all policies to evaluation mode
    original_train_modes = {tid: p.training for tid, p in team_policies.items()}
    for p_eval in team_policies.values():
        p_eval.eval()

    eval_phase_rewards = {tid: [] for tid in team_policies.keys()}
    ACTION_REPEAT = 4 # Use the same value as in training for consistency

    for eval_ep_idx in range(num_eval_episodes):
        print(f"  Running Evaluation Episode {eval_ep_idx + 1}/{num_eval_episodes}...")

        # ==================================
        # 1. EPISODE INITIALIZATION (Mirrors Training Loop)
        # ==================================
        obs_live_S_t, _ = env.reset()
        agent_id_to_team, team_indices = assign_conceptual_teams_for_episode(env.agents, NUM_TEAMS, AGENTS_PER_TEAM_ENV)

        trail_managers = {}
        agent_role_assignments = {}

        for team_id, policy in team_policies.items():
            if isinstance(policy, MAACPolicy):
                indices_for_this_team = team_indices.get(team_id, [])
                if indices_for_this_team:
                    assignments = policy.assign_roles_for_episode(indices_for_this_team)
                    agent_role_assignments.update(assignments)
            elif isinstance(policy, SharedActorPolicy):
                policy.reset_trails()
                manager = TrailManager(cell_size=env.metadata['obs_radius'] * 1.0, device=device)
                manager.set_memories(policy.agent_trail_memories)
                trail_managers[team_id] = manager
        
        env.current_maac_roles = agent_role_assignments.copy()

        for manager in trail_managers.values():
            manager.rebuild_grid()
        
        actor_hidden_states = {i: team_policies[agent_id_to_team[i]].init_hidden(1) for i in range(env.num_agents) if i in agent_id_to_team}
        
        episode_reward_agg = {team: 0.0 for team in team_policies.keys()}
        done, truncated, step = False, False, 0

        while not (done or truncated) and step < MAX_STEPS:
            alive_agent_indices_t = [i for i, ag in enumerate(env.agents) if ag and ag.get('alive')]
            
            # --- "THINK & ACT" PHASE (Mirrors Training Loop) ---
            actions_all_env = [None] * env.num_agents
            next_actor_hidden_states = {}
            
            grouped_policy_inputs = {id(p): {'policy': p, 'indices': []} for p in set(team_policies.values())}
            for i in alive_agent_indices_t:
                conceptual_team_id = agent_id_to_team.get(i)
                if conceptual_team_id is not None:
                    policy = team_policies[conceptual_team_id]
                    grouped_policy_inputs[id(policy)]['indices'].append(i)

            with torch.no_grad():
                for group_data in grouped_policy_inputs.values():
                    if not group_data['indices']: continue
                    policy, indices = group_data['policy'], group_data['indices']

                    obs_list_for_batch = [obs_live_S_t[i] for i in indices]
                    h_states_list = [actor_hidden_states.get(i) for i in indices]
                    
                    # Batch inputs and get actions
                    obs_batch = batch_obs_dicts(obs_list_for_batch, device)
                    hidden_batch = batch_actor_hidden_states(h_states_list, len(indices), policy, device)
                    
                    policy_kwargs = {'obs_batch': obs_batch, 'hidden_state': hidden_batch, 'noise_scale': 0.0} # No noise in eval
                    if isinstance(policy, MAACPolicy):
                        policy_kwargs['all_agent_types_in_team'] = [agent_role_assignments.get(i, policy.agent_types[0]) for i in indices]
                    elif isinstance(policy, (NCA_PINSANPolicy, SharedActorPolicy)):
                        policy_kwargs['agent_policy_indices'] = torch.tensor([
                            team_indices.get(agent_id_to_team.get(gid), []).index(gid)
                            for gid in indices if agent_id_to_team.get(gid) is not None and gid in team_indices.get(agent_id_to_team.get(gid), [])
                        ], dtype=torch.long, device=device)
                    
                    action_batch_dict, next_hidden_batch, _, aux_outputs_batch = policy.act_batch(**policy_kwargs)

                    unbatched_h = unbatch_hidden_states_helper(next_hidden_batch, list(range(len(indices))))
                    for i, global_idx in enumerate(indices):
                        action = action_batch_dict['joint_action'][i].cpu().numpy()
                        actions_all_env[global_idx] = {"movement": action[:MOVEMENT_DIM], "pickup": int(np.clip(round(action[MOVEMENT_DIM]), 0, 2))}
                        next_actor_hidden_states[global_idx] = unbatched_h[i]

                    # Trail Writing (for SharedActorPolicy, mirrors training)
                    if isinstance(policy, SharedActorPolicy) and aux_outputs_batch and aux_outputs_batch.get('semantic_vector') is not None:
                        messages_to_write = aux_outputs_batch['semantic_vector']
                        policy_indices = policy_kwargs['agent_policy_indices'].cpu().numpy()
                        for i, local_policy_idx in enumerate(policy_indices):
                            global_env_idx = indices[i]
                            agent_pos_tensor = torch.tensor(env.agents[global_env_idx]['pos'], device=device)
                            policy.agent_trail_memories[local_policy_idx].update(new_vector=messages_to_write[i], new_position=agent_pos_tensor)

            for i in range(env.num_agents):
                if actions_all_env[i] is None: actions_all_env[i] = {'movement': np.zeros(2), 'pickup': 0}

            # --- "STEP" & "WRITE/STORE" PHASE (Mirrors Training Loop) ---
            # Rebuild trail grids before stepping the environment
            for manager in trail_managers.values():
                manager.rebuild_grid()
                
            # Action Repeat Loop (mirrors training)
            accumulated_rewards_S_tp1 = [{key: 0.0 for key in REWARD_COMPONENT_KEYS} for _ in range(env.num_agents)]
            final_obs_live_S_tp1 = None
            for _ in range(ACTION_REPEAT):
                obs_live_S_tp1_intermediate, rewards_S_tp1_intermediate, done, truncated_env, _ = env.step(actions_all_env)
                for agent_idx in range(env.num_agents):
                    for key, value in rewards_S_tp1_intermediate[agent_idx].items():
                        accumulated_rewards_S_tp1[agent_idx][key] += value
                final_obs_live_S_tp1 = obs_live_S_tp1_intermediate
                if done or truncated_env: break
            
            rewards_S_tp1 = accumulated_rewards_S_tp1
            obs_live_S_tp1 = final_obs_live_S_tp1
            truncated = truncated_env
            
            # NOTE: Memory states (ActorMapState, PersistentGraphMemory) are NOT updated
            # during evaluation, as it's purely for assessing the current policy state.

            # Aggregate rewards
            for i, r_dict in enumerate(rewards_S_tp1):
                team_id = agent_id_to_team.get(i)
                if team_id is not None:
                    episode_reward_agg[team_id] += sum(v for v in r_dict.values() if isinstance(v, (int, float)))

            # Update loop variables
            obs_live_S_t, actor_hidden_states, step = obs_live_S_tp1, next_actor_hidden_states, step + 1
        
        # End of episode loop
        for team_id, total_reward in episode_reward_agg.items():
            eval_phase_rewards[team_id].append(total_reward)

    # --- LOGGING AND CLEANUP ---
    final_avg_reward = 0.0
    if logger:
        eval_summary = {}
        print("\n--- Evaluation Phase Results ---")
        for team_id, rewards_list in eval_phase_rewards.items():
            avg_reward = np.mean(rewards_list) if rewards_list else 0.0
            std_reward = np.std(rewards_list) if rewards_list else 0.0
            team_suffix = "_L" if team_id < 3 else "_G"
            policy_type_idx = team_id % 3
            policy_name = "MAAC" if policy_type_idx == 0 else "NCA" if policy_type_idx == 1 else "Shared"
            eval_summary[f'Evaluation/Team_{team_id}{team_suffix}_AvgReward'] = avg_reward
            eval_summary[f'Evaluation/Team_{team_id}{team_suffix}_StdReward'] = std_reward
            print(f"  Team {team_id} ({policy_name}): Avg Reward = {avg_reward:.2f} +/- {std_reward:.2f}")
        logger.log_metrics_dict(eval_summary, step=eval_episode_num_global)
        
        # For hyperparameter search, we need a single metric to optimize.
        # We will use the average reward across all teams.
        all_rewards = [r for rewards in eval_phase_rewards.values() for r in rewards]
        final_avg_reward = np.mean(all_rewards) if all_rewards else 0.0

    # Restore original training modes
    for tid_restore, p_restore in team_policies.items():
        if hasattr(p_restore, 'train'):
            p_restore.train(original_train_modes.get(tid_restore, True))
    print("--- Evaluation Phase Completed ---\n")
    return final_avg_reward



def run_training_trial(env: Swarm2DEnv, params: Dict, trial_hyperparams: Optional[Dict] = None, optuna_trial: Optional['optuna.Trial'] = None):
    """
    (V2 - Optimized) Encapsulates a full training run.
    - Groups agents by policy for batched observation generation.
    - Integrated with Optuna for hyperparameter search.
    """
    print("--- INSIDE run_training_trial ---")
    set_seeds(params['seed'])
    log_run_name = params['run_name']
    print(f"\n{'='*20} STARTING TRIAL: {log_run_name} (Seed: {params['seed']}) {'='*20}\n")
    logger = Logger(log_dir="logs", run_name=log_run_name, use_tensorboard=True)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define constants from environment
    AGENTS_PER_TEAM_ENV = getattr(env, 'num_agents_per_team', 20)
    NUM_TEAMS = getattr(env, 'num_teams', 6)
    CRITIC_SEQUENCE_LENGTH = 3  # How many consecutive steps the critic sees
    TOTAL_AGENTS_ENV = getattr(env, 'num_agents', NUM_TEAMS * AGENTS_PER_TEAM_ENV)

    log_freq_steps = 50 # Log step-level metrics every N steps

    # --- Initialize Policies, Critics, and Optimizers ---
    team_policies = {}
    team_optimizers = {}
    team_critics1, team_critics2 = {}, {}
    team_target_critics1, team_target_critics2 = {}, {}

    # Use trial hyperparameters if provided, otherwise fall back to defaults
    maac_params = trial_hyperparams.get("maac", best_maac_params)
    nca_params = trial_hyperparams.get("nca", best_nca_params)
    shared_params = trial_hyperparams.get("shared", best_sharedactor_params)

    policy_t0, optimizers_t0 = instantiate_maac_policy(maac_params, env, device, verbose=True)
    team_policies[0], team_optimizers[0] = policy_t0, optimizers_t0
    team_policies[3], team_optimizers[3] = copy.deepcopy(policy_t0), instantiate_maac_policy(maac_params, env, device, verbose=False)[1]
    
    policy_t2, optimizer_t2 = instantiate_shared_policy(shared_params, env, device, num_agents_on_team=AGENTS_PER_TEAM_ENV, verbose=True)
    team_policies[2], team_optimizers[2] = policy_t2, optimizer_t2
    team_policies[5], team_optimizers[5] = copy.deepcopy(policy_t2), instantiate_shared_policy(shared_params, env, device, num_agents_on_team=AGENTS_PER_TEAM_ENV, verbose=False)[1]

    policy_t1, optimizer_t1 = instantiate_nca_policy(nca_params, env, device, num_agents_on_team=AGENTS_PER_TEAM_ENV, verbose=True)
    team_policies[1], team_optimizers[1] = policy_t1, optimizer_t1
    team_policies[4] = copy.deepcopy(policy_t1)
    team_optimizers[4] = instantiate_nca_policy(nca_params, env, device, num_agents_on_team=AGENTS_PER_TEAM_ENV, verbose=False)[1]

    # --- Instantiate Critics (with verbosity control) ---
    critic_core_instance = UnifiedCriticCore(**critic_config, verbose=True).to(device)
    for team_id in range(NUM_TEAMS):
        critic_lr = 3e-4
        
        # Instantiate the first critic pair from the verbose core instance
        if team_id == 0:
            c1 = BaseCriticWrapper(critic_core_instance, critic_lr)
            c2 = BaseCriticWrapper(copy.deepcopy(critic_core_instance), critic_lr)
        else:
            # Subsequent critics are copies and should be silent
            c1 = BaseCriticWrapper(copy.deepcopy(critic_core_instance), critic_lr)
            c2 = BaseCriticWrapper(copy.deepcopy(critic_core_instance), critic_lr)

        team_critics1[team_id], team_critics2[team_id] = c1, c2
        team_target_critics1[team_id], team_target_critics2[team_id] = copy.deepcopy(c1), copy.deepcopy(c2)

    log_alphas = {tid: torch.zeros(1, requires_grad=True, device=device) for tid in range(NUM_TEAMS)}
    alpha_optimizers = {tid: torch.optim.AdamW([log_alphas[tid]], lr=3e-4) for tid in range(NUM_TEAMS)}
    target_entropy = -float(MOVEMENT_DIM)

    # --- Initialize Buffers and State Managers ---
    replay_buffers = {team: ReplayBuffer(params['buffer_capacity'], sequence_length=CRITIC_SEQUENCE_LENGTH + 1) for team in range(NUM_TEAMS)}
    
    # --- Plateau Manager Initialization ---
    plateau_manager = TeamPlateauManager(
        team_ids=list(range(NUM_TEAMS)),
        window_size=20,
        patience=10,
        min_episodes=100,
        min_improvement_threshold=0.02
    )
    
    start_episode, global_step_counter = 0, 0
    
    # --- Initialize Curriculum Manager ---
    curriculum_manager = CurriculumManager(num_teams=NUM_TEAMS)
    if logger:
        logger.log_hyperparams({'curriculum_stage': curriculum_manager.get_current_stage_name()})

    # Apply initial reward overrides from the curriculum
    initial_reward_overrides = curriculum_manager.get_current_reward_overrides()
    env.update_reward_overrides(initial_reward_overrides)

    # --- Checkpoint Loading Logic ---
    LOAD_CHECKPOINT = True # Set to False to force starting from scratch
    CHECKPOINT_DIR_BASE = "checkpoints_" # Matches saving pattern
    CHECKPOINT_LOAD_EPISODE = None # Or set to a specific episode number, e.g., 100

    # Training Params
    num_episodes = 200 if optuna_trial is None else 50 # Shorter runs for HPO
    gamma = 0.99
    save_every = 20
    render_every = float('inf') # Disable rendering during HPO
    
    # The number of transitions to sample from the buffer for each training update.
    # Must be large enough for stable gradients. 256 is a standard, effective value.
    batch_size = 32
    
    
    # The maximum number of transitions to store in the replay buffer.
    # Should be large to ensure diverse experiences are sampled. 1e6 is standard.
    buffer_capacity = int(1e4) 
    
    # The frequency of network updates relative to environment steps can be tuned.
    # These settings mean for every 1 env step, we do 1 critic update and 0.5 actor updates on average.
    critic_update_steps = 1
    actor_update_steps = 2
    
    # Rate for soft-updating the target networks.
    target_tau = 0.005
    start_episode = 0
    global_step_counter = 0 # Initialize global step counter

    if LOAD_CHECKPOINT:
        print("\n--- Attempting to Load Checkpoint ---")
        chkpt_dir_load = f"{CHECKPOINT_DIR_BASE}{log_run_name}" # Directory based on run name
        latest_episode_found = find_latest_checkpoint(chkpt_dir_load, CHECKPOINT_LOAD_EPISODE)

        if latest_episode_found != -1:
            print(f"  Found latest checkpoint at episode {latest_episode_found}. Loading...")
            start_episode = latest_episode_found + 1 # Resume from the NEXT episode
            try:
                # --- Load Training State ---
                training_state_path = os.path.join(chkpt_dir_load, f"training_state_ep{latest_episode_found}.pt")
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path, map_location=device)
                    start_episode = training_state.get('episode', latest_episode_found) + 1
                    global_step_counter = training_state.get('global_step', 0)
                    print(f"    Loaded training state: Resuming from Ep {start_episode}, Global Step {global_step_counter}")
                else:
                    print(f"    Warning: Training state file not found. Global step counter reset.")
                    global_step_counter = 0

                # --- Load Alpha ---
                alpha_path = os.path.join(chkpt_dir_load, f"alpha_state_ep{latest_episode_found}.pt")
                if os.path.exists(alpha_path):
                    alpha_state = torch.load(alpha_path, map_location=device)
                    loaded_log_alphas = alpha_state.get('log_alphas', {})
                    loaded_alpha_optimizers = alpha_state.get('alpha_optimizers', {})
                    for team_id_load, log_a_data in loaded_log_alphas.items():
                        if team_id_load in log_alphas:
                            with torch.no_grad():
                                if isinstance(log_a_data, torch.Tensor): log_alphas[team_id_load].copy_(log_a_data.to(log_alphas[team_id_load].device))
                                else: print(f"Warn: Invalid log_alpha data type T{team_id_load}")
                    for team_id_load, opt_state in loaded_alpha_optimizers.items():
                        if team_id_load in alpha_optimizers:
                            try: alpha_optimizers[team_id_load].load_state_dict(opt_state)
                            except Exception as e: print(f"Warn: Failed loading alpha optim T{team_id_load}: {e}")
                    print("    Alpha state loaded.")
                else: print(f"    Warning: Alpha state file not found: {alpha_path}")

                # --- Load Policies & Optimizers ---
                for team_id_load in range(NUM_TEAMS):
                    if team_id_load not in team_policies: continue
                    policy_load = team_policies[team_id_load]
                    optimizers_load_dict = team_optimizers.get(team_id_load)
                    team_suffix_load = "_L" if team_id_load < 3 else "_G"
                    policy_type_idx_load = team_id_load % 3
                    policy_suffix_load = "_MAAC" if policy_type_idx_load == 0 else "_NCA" if policy_type_idx_load == 1 else "_Shared"
                    policy_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_policy{policy_suffix_load}.pt")
                    optim_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_policy{policy_suffix_load}_optim.pt")

                    if os.path.exists(policy_path):
                        try:
                            # Added weights_only=False to handle UninitializedParameter from lazy layers, a common issue in newer PyTorch versions.
                            # This is safe as we trust the source of our own checkpoints.
                            state_dict = torch.load(policy_path, map_location=device, weights_only=False)
                            policy_load.load_state_dict(state_dict, strict=False)
                            print(f"    Loaded policy T{team_id_load}{team_suffix_load} (strict=False).")
                        except Exception as e:
                            print(f"    ERROR loading policy T{team_id_load}{team_suffix_load}: {e}")

                    if optimizers_load_dict and os.path.exists(optim_path):
                        try:
                            loaded_optim_state = torch.load(optim_path, map_location=device)
                            if policy_suffix_load == "_MAAC":
                                role_optims_saved = loaded_optim_state.get('role_optimizers', {}); shared_gnn_optim_state_saved = loaded_optim_state.get('shared_gnn_optimizer_state')
                                role_optims_current = optimizers_load_dict.get('role_optimizers'); shared_gnn_optim_current = optimizers_load_dict.get('shared_gnn_optimizer')
                                if role_optims_saved and role_optims_current:
                                    for role, optims_saved in role_optims_saved.items():
                                        if role in role_optims_current and optims_saved:
                                            actor_state = optims_saved.get('actor'); reward_state = optims_saved.get('reward')
                                            if actor_state and role_optims_current[role].get('actor'): role_optims_current[role]['actor'].load_state_dict(actor_state)
                                            if reward_state and role_optims_current[role].get('reward'): role_optims_current[role]['reward'].load_state_dict(reward_state)
                                if shared_gnn_optim_state_saved and shared_gnn_optim_current: shared_gnn_optim_current.load_state_dict(shared_gnn_optim_state_saved)
                            else:
                                if 'policy' in optimizers_load_dict: optimizers_load_dict['policy'].load_state_dict(loaded_optim_state)
                            print(f"    Loaded optimizers T{team_id_load}{team_suffix_load}.")
                        except Exception as e: print(f"    ERROR loading optimizer T{team_id_load}{team_suffix_load}: {e}")
                    elif os.path.exists(optim_path): print(f"    Warning: Optimizer file found but no optimizer object T{team_id_load}")

                # --- Load Critics and Targets ---
                for team_id_load in range(NUM_TEAMS):
                    team_suffix_load = "_L" if team_id_load < 3 else "_G"
                    c1_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_critic1.pt"); c2_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_critic2.pt")
                    tc1_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_target_critic1.pt"); tc2_path = os.path.join(chkpt_dir_load, f"T{team_id_load}{team_suffix_load}_ep{latest_episode_found}_target_critic2.pt")
                    if team_id_load in team_critics1 and os.path.exists(c1_path): team_critics1[team_id_load].load_state_dict(torch.load(c1_path, map_location=device))
                    if team_id_load in team_critics2 and os.path.exists(c2_path): team_critics2[team_id_load].load_state_dict(torch.load(c2_path, map_location=device))
                    if team_id_load in team_target_critics1 and os.path.exists(tc1_path): team_target_critics1[team_id_load].load_state_dict(torch.load(tc1_path, map_location=device))
                    if team_id_load in team_target_critics2 and os.path.exists(tc2_path): team_target_critics2[team_id_load].load_state_dict(torch.load(tc2_path, map_location=device))
                print("    Attempted loading critics and targets.")

                # --- Load Replay Buffers ---
                buffer_load_path = os.path.join(chkpt_dir_load, f"replay_buffers_ep{latest_episode_found}.pkl")
                if os.path.exists(buffer_load_path):
                    try:
                        with open(buffer_load_path, 'rb') as f_buf_load:
                            loaded_buffers = pickle.load(f_buf_load)
                            for team_id_load, buffer_deque in loaded_buffers.items():
                                if team_id_load in replay_buffers:
                                    replay_buffers[team_id_load].buffer = deque(buffer_deque, maxlen=replay_buffers[team_id_load].capacity)
                        print(f"    Replay buffers loaded.")
                    except Exception as e: print(f"    Error loading replay buffer: {e}. Starting empty.")
                else: print(f"    Warning: Replay buffer file not found. Starting empty.")

                print(f"--- Checkpoint Loading Completed. Resuming from Episode {start_episode} ---")
            except Exception as e_load_outer:
                print(f"!!!!!! ERROR during checkpoint loading process (Ep {latest_episode_found}): {e_load_outer} !!!!!!")
                print("  Starting training from scratch."); start_episode = 0; global_step_counter = 0
        else: print("  No valid checkpoint directory found or specified. Starting training from scratch.")
    else: print("Checkpoint loading disabled by flag.")

    
    # --- 3. Main Training Loop ---
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler(device.type)
        print(f"AMP GradScaler initialized for {device.type}.")
    elif device.type == 'cpu': # Explicitly handle CPU for clarity, though GradScaler might only be for CUDA
        scaler = None # Or torch.amp.GradScaler(device.type)
        print("AMP GradScaler not initialized (not using CUDA, or CPU AMP does not require GradScaler here).")
    else:
        scaler = None
        print(f"AMP GradScaler disabled (device type: {device.type}).")

    # --- Initialize Aggregate Reward Tracking ---
    episode_rewards_sum = {team: [] for team in range(NUM_TEAMS)}
    episode_losses_critic = {team: [] for team in range(NUM_TEAMS)} # Store avg loss per EPISODE
    episode_losses_actor = {team: [] for team in range(NUM_TEAMS)}  # Store avg loss per EPISODE
    episode_reward_components_sum = {team: {key: 0.0 for key in REWARD_COMPONENT_KEYS} for team in range(NUM_TEAMS)}

    print(f"Starting training for {num_episodes} episodes...")
    global_step_counter = 0

    # ===================================================================
    #                       PROFILING CONFIGURATION
    # ===================================================================
    ENABLE_PROFILING = True  # Set to True to enable profiling, False to disable
    PROFILE_TORCH_EPISODE = 0 # Which episode to run the detailed torch.profiler on
    PROFILE_CPROFILE = True   # Set to True to run cProfile over the whole loop (or selected range)
    # ===================================================================
    # --- cProfile Setup (if enabled) ---
    if ENABLE_PROFILING or PROFILE_CPROFILE:
        print("cProfile Enabled.")
        profiler_cprofile = cProfile.Profile()
        profiler_cprofile.enable()
        
        # Set up signal handler for graceful profiling output on interruption
        global profiler_cprofile_global
        profiler_cprofile_global = profiler_cprofile
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        print("Signal handlers set up for profiling output on interruption.")
    else:
        profiler_cprofile = None
    
    # --- Start Training Loop ---
    plateau_manager = TeamPlateauManager(
        team_ids=list(range(NUM_TEAMS)),
        window_size=20,         # Average reward over the last 20 episodes
        patience=10,            # Stop after 10 checks (e.g., 100 episodes) with no improvement
        min_episodes=100,       # Start checking for plateaus after episode 100
        min_improvement_threshold=0.02 # Require at least a 2% improvement to reset patience
    )
    
    early_stopper = EpisodeEarlyStopper(num_teams=NUM_TEAMS)

    # --- Flatten GRU parameters for performance ---
    print("--- Flattening GRU parameters for performance ---")
    for policy in team_policies.values():
        # Flatten GRU in SharedActorPolicy
        if hasattr(policy, 'network'):
            # This handles the GRU inside the HighLevelTemporalModule
            if hasattr(policy.network, 'temporal_module') and hasattr(policy.network.temporal_module, 'gru'):
                if policy.network.temporal_module.gru is not None:
                    print(f"  Flattening GRU for policy: {type(policy).__name__}")
                    policy.network.temporal_module.gru.flatten_parameters()

        # Flatten GRUs in MAACPolicy
        if isinstance(policy, MAACPolicy):
            print(f"  Flattening GRUs for MAACPolicy actors...")
            if hasattr(policy, 'shared_actor_gnn') and hasattr(policy.shared_actor_gnn, 'gru'):
                policy.shared_actor_gnn.gru.flatten_parameters()

    # --- Start of Episode Loop ---
    print(f"Starting training loop from episode {start_episode}...")
    for episode in range(start_episode, num_episodes):
        start_time_episode = time.time()
        # ==================================
        # 1. EPISODE INITIALIZATION
        # ==================================
        obs_live_S_t, _ = env.reset()
        early_stopper.reset()
        
        # 1. Assign agents to conceptual teams first
        agent_id_to_team, team_indices = assign_conceptual_teams_for_episode(env.agents, NUM_TEAMS, AGENTS_PER_TEAM_ENV)
        
        # 2. Initialize the managers that will be populated
        trail_managers = {}
        agent_role_assignments = {}

        # 3. Loop through policies to set up roles, reset trails, and create managers
        for team_id, policy in team_policies.items():
            if isinstance(policy, MAACPolicy):
                indices_for_this_team = team_indices.get(team_id, [])
                if indices_for_this_team:
                    assignments = policy.assign_roles_for_episode(indices_for_this_team)
                    agent_role_assignments.update(assignments)
            
            elif isinstance(policy, SharedActorPolicy):
                # Reset this policy's internal trail memories for the new episode
                policy.reset_trails() 
                
                # Create a manager for this team and link it to the newly reset memories
                manager = TrailManager(cell_size=env.metadata['obs_radius'] * 1.0, device=device)
                manager.set_memories(policy.agent_trail_memories)
                trail_managers[team_id] = manager

        # 4. Set roles for rendering in the environment
        env.current_maac_roles = agent_role_assignments.copy()

        # 5. Now that all managers are created and linked, build their initial grids
        for manager in trail_managers.values():
            manager.rebuild_grid() # Initial build on what will be empty trails

        # 6. Initialize other per-agent state objects for the new episode
        actor_hidden_states = {i: team_policies[agent_id_to_team[i]].init_hidden(1) for i in range(env.num_agents) if i in agent_id_to_team}
        
        # --- Initialize Episode Aggregators ---
        episode_reward_agg = {team: 0.0 for team in range(NUM_TEAMS)}
        interval_reward_agg = {team: 0.0 for team in range(NUM_TEAMS)}
        interval_reward_components_agg = {team: {key: 0.0 for key in REWARD_COMPONENT_KEYS} for team in range(NUM_TEAMS)}
        interval_critic_losses = {team: [] for team in range(NUM_TEAMS)}
        interval_actor_losses = {team: [] for team in range(NUM_TEAMS)}
        interval_detailed_actor_losses = {team: {} for team in range(NUM_TEAMS)}
        interval_critic_grad_norms = {team: [] for team in range(NUM_TEAMS)}
        interval_actor_grad_norms = {team: [] for team in range(NUM_TEAMS)}
        episode_resources_delivered = 0 # Reset episode counters
        episode_agent_kills = 0         # Reset episode counters
        
        done, truncated = False, False
        episode_steps_count = 0
        step = 0
        
        # Add tqdm progress bar for the episode
        with tqdm(total=MAX_STEPS, desc=f"Episode {episode}", unit="step") as pbar:
            while not (done or truncated):
                # Remove the 500 step limit for actual training
                # if global_step_counter >= 500:
                #     print(f"--- Profiling run complete after {global_step_counter} steps. ---")
                #     truncated = True
                #     continue

                alive_agent_indices_t = [i for i, ag in enumerate(env.agents) if ag and ag.get('alive')]
                
                # --- START: ENVIRONMENT STATE DEBUG ---
                if episode == 0 and step < 2:
                    print(f"\n--- Environment State Debug (Episode {episode}, Step {step}) ---")
                    print(f"  - Total agents: {env.num_agents}")
                    print(f"  - Alive agents: {len(alive_agent_indices_t)}")
                    print(f"  - Agent teams: {[env.agents[i].get('team', -1) for i in alive_agent_indices_t[:10]]}")  # First 10
                    print(f"  - Agent positions: {[env.agents[i].get('pos', [0,0]) for i in alive_agent_indices_t[:3]]}")  # First 3
                    print(f"  - Agent obs radii: {[env.agents[i].get('obs_radius', 50.0) for i in alive_agent_indices_t[:3]]}")  # First 3
                # --- END: ENVIRONMENT STATE DEBUG ---
                
                # --- START: COMPREHENSIVE OBSERVATION DEBUG ---
                if episode == 0 and step < 5 and alive_agent_indices_t:
                    debug_agent_idx = alive_agent_indices_t[0]
                    agent_obs = obs_live_S_t[debug_agent_idx]
                    
                    # Debug observation structure
                    obs_debug_info = observation_debugger.debug_observation_structure(agent_obs, debug_agent_idx, step)
                    
                    # Debug foveation/occlusion if available
                    if 'map' in agent_obs and hasattr(env, 'agents') and env.agents[debug_agent_idx]:
                        agent_pos = np.array(env.agents[debug_agent_idx]['pos'])
                        obs_radius = env.agents[debug_agent_idx].get('obs_radius', 50.0)
                        fovea_radius = obs_radius * 1.0  # Assuming fovea is same as obs radius
                        
                        foveation_debug = observation_debugger.debug_foveation_occlusion(
                            agent_pos, obs_radius, fovea_radius, 
                            occlusion_map=agent_obs.get('memory_map'),  # Use memory_map as occlusion proxy
                            agent_idx=debug_agent_idx, step=step
                        )
                        obs_debug_info.update(foveation_debug)
                    
                    # Debug memory map
                    if 'memory_map' in agent_obs:
                        memory_debug = observation_debugger.debug_memory_map(
                            agent_obs['memory_map'], debug_agent_idx, step
                        )
                        obs_debug_info.update(memory_debug)
                    
                    # Debug unified graph
                    if 'graph' in agent_obs:
                        graph_debug = observation_debugger.debug_unified_graph(
                            agent_obs['graph'], debug_agent_idx, step
                        )
                        obs_debug_info.update(graph_debug)
                    
                    # Debug observation radius and spatial relationships
                    if hasattr(env, 'agents') and env.agents[debug_agent_idx]:
                        agent_pos = np.array(env.agents[debug_agent_idx]['pos'])
                        obs_radius = env.agents[debug_agent_idx].get('obs_radius', 50.0)
                        radius_debug = observation_debugger.debug_observation_radius(
                            agent_pos, obs_radius, env.agents, debug_agent_idx, step
                        )
                        obs_debug_info.update(radius_debug)
                    
                    # Print comprehensive debug summary
                    observation_debugger.print_debug_summary(obs_debug_info, debug_agent_idx, step)
                    
                    # Save debug data (only if enabled)
                    if step == 0 and observation_debugger.save_debug_images:  # Save only first step to avoid too many files
                        try:
                            observation_debugger.save_debug_data(obs_debug_info, f"obs_debug_agent_{debug_agent_idx}_step_{step}.json")
                        except Exception as e:
                            print(f"Warning: Could not save debug data: {e}")
                # --- END: COMPREHENSIVE OBSERVATION DEBUG ---

                # --- "THINK & ACT" PHASE (OPTIMIZED) ---
                actions_all_env = [None] * env.num_agents
                roles_info_all_env = [None] * env.num_agents
                next_actor_hidden_states = {}
                all_aux_outputs = {}
                
                # Group agents by policy for efficient, batched processing
                grouped_policy_inputs = {id(p): {'policy': p, 'indices': []} for p in set(team_policies.values())}
                for i in alive_agent_indices_t:
                    conceptual_team_id = agent_id_to_team.get(i)
                    if conceptual_team_id is not None:
                        policy = team_policies[conceptual_team_id]
                        grouped_policy_inputs[id(policy)]['indices'].append(i)

                with torch.no_grad():
                    for group_data in grouped_policy_inputs.values():
                        if not group_data['indices']: continue
                        policy, indices = group_data['policy'], group_data['indices']

                        # The observation from the environment now contains the complete, unified graph.
                        # No redundant processing is needed here.
                        obs_list_for_batch = [obs_live_S_t[i] for i in indices]
                        h_states_list = [actor_hidden_states.get(i) for i in indices]
                        
                        # 4. Batch inputs and get actions
                        obs_batch = batch_obs_dicts(obs_list_for_batch, device)
                        hidden_batch = batch_actor_hidden_states(h_states_list, len(indices), policy, device)
                        
                        if step % 50 == 0 and step > 0:
                            policy_name = type(policy).__name__
                            print(f"\n--- DEBUG: Batched obs for {policy_name} (Ep {episode}, Step {step}) ---")
                            for key, value in obs_batch.items():
                                if isinstance(value, torch.Tensor):
                                    print(f"  - {key}: Tensor shape={value.shape}")
                                elif isinstance(value, Batch):
                                    print(f"  - {key}: Graph Batch, num_graphs={value.num_graphs}, total_nodes={value.num_nodes}")
                            print("-" * 20)

                        policy_kwargs = {'obs_batch': obs_batch, 'hidden_state': hidden_batch, 'noise_scale': 0.1}
                        if isinstance(policy, MAACPolicy):
                            policy_kwargs['all_agent_types_in_team'] = [agent_role_assignments.get(i, policy.agent_types[0]) for i in indices]
                        elif isinstance(policy, (NCA_PINSANPolicy, SharedActorPolicy)):
                            policy_kwargs['agent_policy_indices'] = torch.tensor([
                                team_indices.get(agent_id_to_team.get(gid), []).index(gid)
                                for gid in indices if agent_id_to_team.get(gid) is not None and gid in team_indices.get(agent_id_to_team.get(gid), [])
                            ], dtype=torch.long, device=device)
                        
                        action_batch_dict, next_hidden_batch, role_batch_dict, aux_outputs_batch = policy.act_batch(**policy_kwargs)

                        # 5. Distribute results back to the global lists
                        unbatched_h = unbatch_hidden_states_helper(next_hidden_batch, list(range(len(indices))))
                        for i, global_idx in enumerate(indices):
                            action = action_batch_dict['joint_action'][i].cpu().numpy()
                            actions_all_env[global_idx] = {"movement": action[:MOVEMENT_DIM], "pickup": int(np.clip(round(action[MOVEMENT_DIM]), 0, 2))}
                            roles_info_all_env[global_idx] = {k: (v[i].cpu().numpy() if hasattr(v, 'cpu') and v is not None and v.numel() > 0 else (v[i] if isinstance(v, list) else v)) for k, v in role_batch_dict.items()}
                            next_actor_hidden_states[global_idx] = unbatched_h[i]
                            if aux_outputs_batch and aux_outputs_batch.get('semantic_vector') is not None:
                                all_aux_outputs[global_idx] = {'semantic_vector': aux_outputs_batch['semantic_vector'][i]}

                        if isinstance(policy, SharedActorPolicy):
                            if aux_outputs_batch and aux_outputs_batch.get('semantic_vector') is not None:
                                messages_to_write = aux_outputs_batch['semantic_vector']
                                # Get the local policy indices for this batch
                                policy_indices = policy_kwargs['agent_policy_indices'].cpu().numpy()
                                
                                for i, local_policy_idx in enumerate(policy_indices):
                                    global_env_idx = indices[i]
                                    agent_pos_tensor = torch.tensor(env.agents[global_env_idx]['pos'], device=device)
                                    # Update the trail memory for this specific agent
                                    policy.agent_trail_memories[local_policy_idx].update(
                                        new_vector=messages_to_write[i],
                                        new_position=agent_pos_tensor
                                    )


                # Fill in actions for dead agents
                for i in range(env.num_agents):
                    if actions_all_env[i] is None: actions_all_env[i] = {'movement': np.zeros(2), 'pickup': 0}
                
                # --- START: ACTION DEBUG ---
                if episode == 0 and step < 2 and alive_agent_indices_t:
                    debug_agent_idx = alive_agent_indices_t[0]
                    print(f"\n--- Action Debug (Agent {debug_agent_idx}, Step {step}) ---")
                    print(f"  - Action: {actions_all_env[debug_agent_idx]}")
                    if roles_info_all_env[debug_agent_idx]:
                        print(f"  - Role info: {roles_info_all_env[debug_agent_idx]}")
                # --- END: ACTION DEBUG ---

                # --- Rendering 2d
                if episode % render_every == 0:
                    try:
                        env.render()
                    except Exception as e:
                        print(f"Warning: Rendering failed with error: {e}. Disabling for this run.")
                        render_every = float('inf') # Prevent future render attempts


                # --- "STEP" & "WRITE/STORE" PHASE ---
                packed_env_tensors_S_t = {
                    "all_pos": env.current_step_all_pos_t.cpu(),
                    "all_feat": env.current_step_all_feat_t.cpu(),
                    "all_types": env.current_step_all_types_t.cpu(),
                    "all_teams": env.current_step_all_teams_t.cpu(),
                    "all_radii": env.current_step_all_radii_t.cpu(),
                    "all_coop": env.current_step_all_coop_t.cpu(),
                    "step": env.step_counter,
                    "metadata": env.metadata # Keep metadata for normalization constants
                }

                # This needs to be done BEFORE env.step()
                for manager in trail_managers.values():
                    manager.rebuild_grid()

                # Now, query the grids efficiently for each agent
                nearby_trails_S_t = {}
                for agent_idx in alive_agent_indices_t:
                    conceptual_team_id = agent_id_to_team.get(agent_idx)
                    if conceptual_team_id in trail_managers:
                        agent_pos_np = env.agents[agent_idx]['pos']
                        query_radius = env.agents[agent_idx].get('obs_radius', OBS_RADIUS)
                        
                        # O(1) query instead of O(N) loop
                        nearby_vectors_tensor = trail_managers[conceptual_team_id].query_nearby_trails(agent_pos_np, query_radius)
                        
                        if nearby_vectors_tensor is not None and nearby_vectors_tensor.numel() > 0:
                            nearby_trails_S_t[agent_idx] = nearby_vectors_tensor
                        else:
                            # Store an empty tensor to maintain consistent data structure in the buffer
                            policy_ref = team_policies[conceptual_team_id]
                            nearby_trails_S_t[agent_idx] = torch.empty(0, policy_ref.semantic_dim, device=device)


                # Now, step the environment
                # --- ACTION REPEAT LOOP ---
                obs_live_S_tp1, rewards_S_tp1, done, truncated_env, _ = env.step(actions_all_env)

                # --- START: REWARD DEBUG ---
                if episode == 0 and step < 2 and alive_agent_indices_t:
                    debug_agent_idx = alive_agent_indices_t[0]
                    print(f"\n--- Reward Debug (Agent {debug_agent_idx}, Step {step}) ---")
                    reward_dict = rewards_S_tp1[debug_agent_idx]
                    total_reward = sum(v for v in reward_dict.values() if isinstance(v, (int, float)))
                    print(f"  - Total reward: {total_reward:.4f}")
                    print(f"  - Reward components:")
                    for key, value in reward_dict.items():
                        if isinstance(value, (int, float)) and value != 0:
                            print(f"    {key}: {value:.4f}")
                    print(f"{'='*50}\n")
                # --- END: REWARD DEBUG ---

                # --- Update stoppers and other logic using the final state of the action repeat sequence ---
                # early_stopper.update(rewards_S_tp1, step)          
                # if early_stopper.should_stop():
                #     truncated = True
                #     print(f"--- Episode {episode} stopped early at step {step} due to stagnation. ---")
                # else:
                #     truncated = truncated_env
                truncated = truncated_env
            
                # Note: The environment's `step` (and thus `obs_live_S_tp1`) now handles
                # all memory updates internally. No further memory updates are needed here.
             
                # Add transitions to the replay buffer
                for agent_idx in alive_agent_indices_t:
                    conceptual_team_id = agent_id_to_team.get(agent_idx)
                    if conceptual_team_id is None: continue

                    agent_policy_idx = -1
                    team_agent_list = team_indices.get(conceptual_team_id, [])
                    if agent_idx in team_agent_list:
                        agent_policy_idx = team_agent_list.index(agent_idx)
                    
                    # The observation dictionaries from the environment are now complete and self-contained.
                    # No need to manually reconstruct them from different memory views.
                    full_actor_obs_S_t = obs_live_S_t[agent_idx]
                    full_actor_obs_S_tp1 = obs_live_S_tp1[agent_idx]


                    replay_buffers[conceptual_team_id].add(
                        full_actor_obs_S_t={k: v.detach().cpu() if hasattr(v, 'detach') else copy.deepcopy(v) for k, v in full_actor_obs_S_t.items()},
                        full_actor_obs_S_tp1={k: v.detach().cpu() if hasattr(v, 'detach') else copy.deepcopy(v) for k, v in full_actor_obs_S_tp1.items()},
                        action_S_t=actions_all_env[agent_idx],
                        reward_S_tp1=rewards_S_tp1[agent_idx],
                        done_tp1=(done or truncated),
                        role_info_S_t=roles_info_all_env[agent_idx],
                        h_actor_t=actor_hidden_states.get(agent_idx),
                        h_actor_tp1=next_actor_hidden_states.get(agent_idx),
                        nearby_trails_S_t=nearby_trails_S_t.get(agent_idx),
                        packed_env_tensors_S_t=packed_env_tensors_S_t,
                        team_memory_map_S_t=None, # These will be aggregated later
                        team_raw_map_S_t=None,
                        team_persistent_graph_S_t=None,
                        episode=episode, step=step, global_agent_idx=agent_idx,
                        agent_policy_idx=agent_policy_idx
                    )

                # Update loop variables
                obs_live_S_t = obs_live_S_tp1
                actor_hidden_states = next_actor_hidden_states
                global_step_counter += 1
                step += 1
                episode_steps_count +=1
                pbar.update(1) # Increment the progress bar

                # --- Reward Aggregation ---
                current_delivered = getattr(env, 'resources_delivered_count', episode_resources_delivered)
                current_kills = getattr(env, 'agents_killed_count', episode_agent_kills)
                episode_resources_delivered = current_delivered
                episode_agent_kills = current_kills

                for agent_idx_reward in range(TOTAL_AGENTS_ENV):
                    conceptual_team_reward = agent_id_to_team.get(agent_idx_reward)
                    if conceptual_team_reward is None: continue
                    agent_reward_dict = rewards_S_tp1[agent_idx_reward] 
                    scalar_reward = sum(v for v in agent_reward_dict.values() if isinstance(v, (int, float)))
                    interval_reward_agg[conceptual_team_reward] += scalar_reward
                    episode_reward_agg[conceptual_team_reward] += scalar_reward
                    for comp_key in REWARD_COMPONENT_KEYS:
                        reward_val = agent_reward_dict.get(comp_key, 0.0)
                        interval_reward_components_agg[conceptual_team_reward][comp_key] += reward_val
                        episode_reward_components_sum[conceptual_team_reward][comp_key] += reward_val

                # --- Step-level Logging ---
                if global_step_counter % log_freq_steps == 0 and logger:
                    step_log_metrics = {}
                    for team_id_log in range(NUM_TEAMS):
                        team_suffix_log = "_L" if team_id_log < 3 else "_G"
                        step_log_metrics[f'Reward_Interval/Team_{team_id_log}{team_suffix_log}_Sum'] = interval_reward_agg[team_id_log]
                        for comp_key in REWARD_COMPONENT_KEYS: step_log_metrics[f'Reward_Components_Interval/Team_{team_id_log}{team_suffix_log}/{comp_key.replace("r_","")}'] = interval_reward_components_agg[team_id_log][comp_key]
                        if interval_critic_losses[team_id_log]: step_log_metrics[f'Loss_StepAvg/Team_{team_id_log}{team_suffix_log}_Critic'] = np.mean(interval_critic_losses[team_id_log])
                        if interval_actor_losses[team_id_log]: step_log_metrics[f'Loss_StepAvg/Team_{team_id_log}{team_suffix_log}_Actor_Total'] = np.mean(interval_actor_losses[team_id_log])
                        q_list = interval_detailed_actor_losses.get(team_id_log, {}).get('SAC_Avg_Q_Pred_Online', [])
                        if q_list: step_log_metrics[f'SAC_StepAvg/Team_{team_id_log}{team_suffix_log}_AvgQPred'] = np.mean(q_list)
                        if interval_critic_grad_norms[team_id_log]: step_log_metrics[f'Gradients_StepAvg/Team_{team_id_log}{team_suffix_log}_Critic_Norm_PreClip'] = np.mean(interval_critic_grad_norms[team_id_log])
                        crit_post_list = interval_detailed_actor_losses.get(team_id_log, {}).get('Critic_GradNorm_PostClip', [])
                        if crit_post_list: step_log_metrics[f'Gradients_StepAvg/Team_{team_id_log}{team_suffix_log}_Critic_Norm_PostClip'] = np.mean(crit_post_list)
                        if interval_actor_grad_norms[team_id_log]: step_log_metrics[f'Gradients_StepAvg/Team_{team_id_log}{team_suffix_log}_Actor_Norm_PreClip'] = np.mean(interval_actor_grad_norms[team_id_log])
                        act_post_list = interval_detailed_actor_losses.get(team_id_log, {}).get('Actor_GradNorm_PostClip', [])
                        if act_post_list: step_log_metrics[f'Gradients_StepAvg/Team_{team_id_log}{team_suffix_log}_Actor_Norm_PostClip'] = np.mean(act_post_list)
                        for detail_name, detail_list in interval_detailed_actor_losses.get(team_id_log, {}).items():
                            if detail_list and detail_name not in ['SAC_Avg_Q_Pred_Online', 'Critic_GradNorm_PostClip', 'Actor_GradNorm_PostClip']:
                                step_log_metrics[f'Loss_Detail_StepAvg/Team_{team_id_log}{team_suffix_log}/{detail_name.replace("[","_").replace("]","").replace(".","_")}'] = np.mean(detail_list)
                    logger.log_metrics_dict(step_log_metrics, step=global_step_counter)
                    
                    # --- Add Console Output ---
                    # Construct the log string
                    log_str = f"Ep {episode} | Step {step} (Global: {global_step_counter}) | "
                    team_logs = []
                    for team_id_log in range(NUM_TEAMS):
                        team_suffix_log = "_L" if team_id_log < 3 else "_G"
                        reward = step_log_metrics.get(f'Reward_Interval/Team_{team_id_log}{team_suffix_log}_Sum', 'N/A')
                        critic_loss = step_log_metrics.get(f'Loss_StepAvg/Team_{team_id_log}{team_suffix_log}_Critic', 'N/A')
                        
                        reward_str = f"{reward:.2f}" if isinstance(reward, float) else reward
                        critic_loss_str = f"{critic_loss:.4f}" if isinstance(critic_loss, float) else critic_loss
                        
                        team_logs.append(f"T{team_id_log}{team_suffix_log} R: {reward_str}, CL: {critic_loss_str}")
                    log_str += " | ".join(team_logs)
                    pbar.set_postfix_str(log_str)


                    # Reset Interval Accumulators
                    interval_reward_agg = {team: 0.0 for team in range(NUM_TEAMS)}; interval_reward_components_agg = {team: {key: 0.0 for key in REWARD_COMPONENT_KEYS} for team in range(NUM_TEAMS)}
                    interval_critic_losses = {team: [] for team in range(NUM_TEAMS)}; interval_actor_losses = {team: [] for team in range(NUM_TEAMS)}
                    for team_id_reset in range(NUM_TEAMS): interval_detailed_actor_losses[team_id_reset] = {}
                    interval_critic_grad_norms = {team: [] for team in range(NUM_TEAMS)}; interval_actor_grad_norms = {team: [] for team in range(NUM_TEAMS)}

                if step % 100 == 0 and step > 0:
                    # We can use pbar.set_description or just let the main bar update
                    pass

                # --- Training Updates ---
                buffer_ready = any(len(buf) >= batch_size for buf in replay_buffers.values())

                # This function encapsulates the entire update logic for one step.
                def training_update_step():
                    nonlocal did_perform_update_step # Allow modification of the outer scope variable

                    # --- Pre-computation for all teams (if any need an update) ---
                    # This block runs only once per `training_update_step` call.
                    prepped_batches_for_update = {}
                    
                    # Determine which teams actually need an update to avoid unnecessary work
                    teams_needing_update = []
                    if global_step_counter % critic_update_steps == 0:
                        teams_needing_update.extend(list(range(NUM_TEAMS)))
                    if global_step_counter % actor_update_steps == 0:
                        teams_needing_update.extend(list(range(NUM_TEAMS)))
                    teams_needing_update = sorted(list(set(teams_needing_update))) # Get unique teams

                    for team_id_prep in teams_needing_update:
                        if plateau_manager.is_team_frozen(team_id_prep):
                            continue
                        
                        # We only need to sample once per team for this entire update step
                        sample_result = replay_buffers[team_id_prep].sample(batch_size)
                        if sample_result:
                            raw_sequences, tree_indices, is_weights = sample_result
                            
                            # Prepare the comprehensive batch data. This is the expensive step.
                            processed_batch = prepare_batch_for_update(
                                raw_sequences, team_id_prep, (team_id_prep >= 3),
                                team_policies[team_id_prep], team_critics1[team_id_prep],
                                env.metadata, device
                            )
                            
                            if processed_batch:
                                # Store everything needed for both critic and actor updates
                                prepped_batches_for_update[team_id_prep] = {
                                    "processed_batch": processed_batch,
                                    "tree_indices": tree_indices,
                                    "is_weights": is_weights,
                                    "raw_transitions": raw_sequences # Store for actor update
                                }

                    # --- Critic Updates ---
                    if global_step_counter % critic_update_steps == 0:
                        update_order = list(prepped_batches_for_update.keys())
                        random.shuffle(update_order)
                        for team_id_update in update_order:
                            batch_data = prepped_batches_for_update[team_id_update]
                            
                            critic_results = update_critic_fn_sac(
                                team_critics1[team_id_update], team_critics2[team_id_update],
                                team_target_critics1[team_id_update], team_target_critics2[team_id_update],
                                team_policies[team_id_update], log_alphas[team_id_update],
                                batch_data["processed_batch"], gamma, 
                                is_weights=batch_data["is_weights"],
                                device=device, scaler=scaler
                            )
                            
                            if critic_results and critic_results[0] is not None:
                                did_perform_update_step = True
                                loss1_val, loss2_val, avg_q, pre_norm, post_norm, new_priorities = critic_results
                                
                                # --- START: DEBUG PRINTS ---
                                if global_step_counter % 1 == 0:  # Debug: Show every update
                                    team_suffix = "_L" if team_id_update < 3 else "_G"
                                    print(f"--- Critic Update (T{team_id_update}{team_suffix}, Ep {episode}, Step {global_step_counter}) ---")
                                    print(f"  - Losses (C1/C2): {loss1_val:.4f} / {loss2_val:.4f}")
                                    print(f"  - Grad Norms (Pre/Post): {pre_norm or 0.0:.4f} / {post_norm or 0.0:.4f}")
                                    print(f"  - Avg Q-Value: {avg_q:.4f}")
                                # --- END: DEBUG PRINTS ---

                                replay_buffers[team_id_update].update_priorities(batch_data["tree_indices"], new_priorities)

                                # Logging logic for critic
                                interval_critic_losses[team_id_update].append((loss1_val + loss2_val) / 2.0)
                                if avg_q is not None: interval_detailed_actor_losses.setdefault(team_id_update, {}).setdefault('SAC_Avg_Q_Pred_Online', []).append(avg_q)
                                if pre_norm is not None: interval_critic_grad_norms[team_id_update].append(pre_norm)
                                if post_norm is not None: interval_detailed_actor_losses.setdefault(team_id_update, {}).setdefault('Critic_GradNorm_PostClip', []).append(post_norm)

                    # --- Actor & Alpha Updates ---
                    if global_step_counter % actor_update_steps == 0:
                        update_order_actor = list(prepped_batches_for_update.keys())
                        random.shuffle(update_order_actor)
                        for team_id_update_actor in update_order_actor:
                            if team_id_update_actor not in prepped_batches_for_update:
                                continue

                            batch_data_actor = prepped_batches_for_update[team_id_update_actor]
                            policy_actor = team_policies.get(team_id_update_actor)
                            critic1_actor = team_critics1.get(team_id_update_actor)
                            log_alpha_actor = log_alphas.get(team_id_update_actor)
                            
                            actor_update_fn = None
                            if isinstance(policy_actor, MAACPolicy): actor_update_fn = update_actor_maac
                            elif isinstance(policy_actor, NCA_PINSANPolicy): actor_update_fn = update_actor_nca
                            elif isinstance(policy_actor, SharedActorPolicy): actor_update_fn = update_actor_shared

                            if actor_update_fn:
                                # Pass the pre-processed batch and the raw transitions to the actor update function
                                actor_results = actor_update_fn(
                                    policy_actor, critic1_actor, log_alpha_actor,
                                    batch_data_actor["processed_batch"], # Pass the already processed batch
                                    batch_data_actor, # Pass the full dictionary including raw transitions
                                    team_id_update_actor,
                                    team_optimizers, env.metadata, device, scaler
                                )
                                if actor_results and actor_results[0] is not None:
                                    did_perform_update_step = True
                                    actor_loss, detailed_losses, pre_norm, post_norm, log_pi = actor_results
                                    
                                    # --- START: DEBUG PRINTS ---
                                    if global_step_counter % 1 == 0:  # DEBUG: Show every update
                                        team_suffix = "_L" if team_id_update_actor < 3 else "_G"
                                        alpha_val = torch.exp(log_alpha_actor.detach()).item()
                                        print(f"--- Actor Update (T{team_id_update_actor}{team_suffix}, Ep {episode}, Step {global_step_counter}) ---")
                                        print(f"  - Total Loss: {actor_loss:.4f} | Alpha: {alpha_val:.5f}")
                                        print(f"  - Grad Norms (Pre/Post): {pre_norm or 0.0:.4f} / {post_norm or 0.0:.4f}")
                                        if detailed_losses:
                                            loss_str = " | ".join([f"{name}: {val:.3f}" for name, val in detailed_losses.items()])
                                            print(f"  - Detailed Losses: {loss_str}")
                                    # --- END: DEBUG PRINTS ---

                                    if actor_loss is not None: interval_actor_losses[team_id_update_actor].append(actor_loss)
                                    if pre_norm is not None: interval_actor_grad_norms[team_id_update_actor].append(pre_norm)
                                    if post_norm is not None: interval_detailed_actor_losses.setdefault(team_id_update_actor, {}).setdefault('Actor_GradNorm_PostClip', []).append(post_norm)
                                    if detailed_losses:
                                        for name, val in detailed_losses.items():
                                            interval_detailed_actor_losses.setdefault(team_id_update_actor, {}).setdefault(name, []).append(val)
                                    
                                    if log_pi is not None:
                                        alpha_optim_actor = alpha_optimizers[team_id_update_actor]
                                        alpha_loss = -(log_alpha_actor * (log_pi.sum(dim=1) + target_entropy).detach()).mean()
                                        alpha_optim_actor.zero_grad(set_to_none=True)
                                        if scaler: scaler.scale(alpha_loss).backward()
                                        else: alpha_loss.backward()
                                        if scaler: scaler.step(alpha_optim_actor)
                                        else: alpha_optim_actor.step()

                if buffer_ready and global_step_counter > batch_size:
                    did_perform_update_step = False
                    
                    # Debug: Show buffer status
                    if step % 50 == 0:
                        print(f"\n--- Training Update Debug (Step {step}) ---")
                        for tid, buf in replay_buffers.items():
                            print(f"  Team {tid} buffer: {len(buf)}/{buf.capacity} samples")
                        print(f"  Buffer ready: {buffer_ready}, Global step: {global_step_counter}")
                        print(f"  Critic update step: {global_step_counter % critic_update_steps == 0}")
                        print(f"  Actor update step: {global_step_counter % actor_update_steps == 0}")
                        print("-" * 40)

                    # *** Profiling and Execution Logic ***
                    run_torch_profiler_this_step = (ENABLE_PROFILING and episode == PROFILE_TORCH_EPISODE and (global_step_counter % critic_update_steps == 0 or global_step_counter % actor_update_steps == 0))
                    
                    if run_torch_profiler_this_step:
                        print(f"\n--- [Torch Profiler START - Ep {episode}, Step {global_step_counter}] ---", flush=True)
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True, profile_memory=True, with_stack=True
                        ) as prof:
                            training_update_step()
                        
                        if device.type == 'cuda': torch.cuda.synchronize()
                        
                        print("--- Torch Profiler Results ---", flush=True)
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
                        
                        # Profiler Saving Logic ###
                        try:
                            log_directory = logger.log_dir if logger and hasattr(logger, 'log_dir') else "."
                            os.makedirs(log_directory, exist_ok=True)
                            trace_path = os.path.join(log_directory, f"profile_ep{episode}_step{global_step_counter}.json")
                            prof.export_chrome_trace(trace_path)
                            print(f"  Profiler trace saved to {trace_path}", flush=True)
                        except Exception as e_prof_save:
                            print(f"  Warning: Could not save profiler trace: {e_prof_save}", flush=True)
                    else:
                        # Normal execution path
                        training_update_step()

                    # --- SOFT TARGET UPDATES & SCALER STEP (called once after all updates) ---
                    if did_perform_update_step:
                        for team_id_soft in range(NUM_TEAMS):
                            soft_update_targets(
                                team_critics1.get(team_id_soft), team_target_critics1.get(team_id_soft),
                                team_critics2.get(team_id_soft), team_target_critics2.get(team_id_soft),
                                target_tau
                            )
                        if scaler:
                            scaler.update()

        
        # ===================== END EPISODE STEP LOOP =====================
        # Remove the 500 step limit for actual training
        # if global_step_counter >= 500:
        #     break
        # --- Update the Plateau Manager (ONCE per episode) ---
        plateau_manager.update(episode, episode_reward_agg)

        end_time_episode = time.time()
        duration_episode = end_time_episode - start_time_episode
        print(f"--- Episode {episode} End ---")
        print(f"  Duration: {duration_episode:.2f}s, Steps: {episode_steps_count}")

        # --- Log Episode Summary Metrics ---
        if logger:
            # Get the final state of the environment for logging stats
            final_agents = env.agents
            final_hives = env.hives
            final_agent_counts = {t: 0 for t in range(NUM_TEAMS)}
            final_hive_health = {t: 0.0 for t in range(NUM_TEAMS)}
            final_hive_food = {t: 0.0 for t in range(NUM_TEAMS)}

            for agent_stat in final_agents:
                if agent_stat and agent_stat.get('alive', True):
                    env_team_id = agent_stat.get('team', -1)
                    if 0 <= env_team_id < NUM_TEAMS:
                        conceptual_team_id = agent_id_to_team.get(agent_stat.get('id',-1), -1)
                        if conceptual_team_id != -1:
                           final_agent_counts[conceptual_team_id] += 1
                           
            for team_id_stat, hive_data_stat in final_hives.items():
                if isinstance(team_id_stat, int) and 0 <= team_id_stat < NUM_TEAMS:
                    final_hive_health[team_id_stat] = hive_data_stat.get('health', 0.0)
                    final_hive_food[team_id_stat] = hive_data_stat.get('food_store', 0.0)
            
            summary_metrics = {}
            for team_id_log in range(NUM_TEAMS):
                team_suffix = "_L" if team_id_log < 3 else "_G"
                policy_type_idx = team_id_log % 3
                policy_name = "MAAC" if policy_type_idx == 0 else "NCA" if policy_type_idx == 1 else "Shared"
                
                # Log total reward for the episode
                reward_sum = episode_reward_agg.get(team_id_log, 0.0)
                episode_rewards_sum[team_id_log].append(reward_sum)
                summary_metrics[f'EpisodeSummary/Team_{team_id_log}{team_suffix}/TotalReward'] = reward_sum
                print(f"  Team {team_id_log}{team_suffix} ({policy_name}): Reward Sum = {reward_sum:.2f}")

                # Log average losses for the episode
                avg_critic_ep_loss = np.mean(interval_critic_losses[team_id_log]) if interval_critic_losses.get(team_id_log) else np.nan
                avg_actor_ep_loss = np.mean(interval_actor_losses[team_id_log]) if interval_actor_losses.get(team_id_log) else np.nan
                episode_losses_critic[team_id_log].append(avg_critic_ep_loss)
                episode_losses_actor[team_id_log].append(avg_actor_ep_loss)
                summary_metrics[f'EpisodeSummary/Team_{team_id_log}{team_suffix}/AvgCriticLoss_Ep'] = avg_critic_ep_loss
                summary_metrics[f'EpisodeSummary/Team_{team_id_log}{team_suffix}/AvgActorLoss_Total_Ep'] = avg_actor_ep_loss
                
                # Log reward components
                for comp_key in REWARD_COMPONENT_KEYS:
                    comp_name = comp_key.replace('r_', '')
                    comp_value_ep_sum = episode_reward_components_sum[team_id_log].get(comp_key, 0.0)
                    summary_metrics[f'Reward_Components_EpSum/Team_{team_id_log}{team_suffix}/{comp_name}'] = comp_value_ep_sum
                
                # Log final state stats
                summary_metrics[f'EpisodeSummary/Team_{team_id_log}{team_suffix}/FinalHiveHealth'] = final_hive_health.get(team_id_log, 0.0)
                summary_metrics[f'EpisodeSummary/Team_{team_id_log}{team_suffix}/FinalAgentCount'] = final_agent_counts.get(team_id_log, 0)

            # Log Overall Episode Stats
            summary_metrics['EpisodeSummary/Overall/Total_Resources_Delivered_Ep'] = env.resources_delivered_count
            summary_metrics['EpisodeSummary/Overall/Total_Agent_Kills_Ep'] = env.agents_killed_count
            summary_metrics['Meta/Steps_Ep'] = episode_steps_count
            summary_metrics['Meta/Duration_Ep_Sec'] = duration_episode
            if torch.cuda.is_available():
                summary_metrics['System/GPU_Memory_Allocated_MB_EpEnd'] = torch.cuda.memory_allocated() / (1024**2)
            
            logger.log_metrics_dict(summary_metrics, step=episode)
            print("-" * 40)
            
            # --- Reset EPISODE reward component sums for the next episode ---
            episode_reward_components_sum = {team: {key: 0.0 for key in REWARD_COMPONENT_KEYS} for team in range(NUM_TEAMS)}

        # --- Aggregate Metric Logging (Every 5 episodes) ---
        if episode % 5 == 0 and episode > 0 and logger:
            log_metrics_agg = {}
            team_avg_rewards_agg = {}
            for team_id_agg in range(NUM_TEAMS):
                team_suffix_agg = "_L" if team_id_agg < 3 else "_G"
                rewards_hist = episode_rewards_sum[team_id_agg]
                avg_r_hist = np.mean(rewards_hist[-5:]) if len(rewards_hist) >= 5 else (np.mean(rewards_hist) if rewards_hist else 0.0)
                log_metrics_agg[f'Reward_Avg5Ep/Team_{team_id_agg}{team_suffix_agg}_Sum'] = avg_r_hist
                team_avg_rewards_agg[team_id_agg] = avg_r_hist
                
                critic_losses_hist = [l for l in episode_losses_critic[team_id_agg][-5:] if l is not None and not np.isnan(l)]
                actor_losses_hist = [l for l in episode_losses_actor[team_id_agg][-5:] if l is not None and not np.isnan(l)]
                if critic_losses_hist: log_metrics_agg[f'Loss_Avg5Ep/Team_{team_id_agg}{team_suffix_agg}_Critic'] = np.mean(critic_losses_hist)
                if actor_losses_hist: log_metrics_agg[f'Loss_Avg5Ep/Team_{team_id_agg}{team_suffix_agg}_Actor_Total'] = np.mean(actor_losses_hist)
            
            # Comparison
            for i_comp in range(3):
                t_lim, t_glob = i_comp, i_comp + 3
                avg_r_lim, avg_r_glob = team_avg_rewards_agg.get(t_lim, 0.0), team_avg_rewards_agg.get(t_glob, 0.0)
                ratio = avg_r_lim / (avg_r_glob + 1e-6) if avg_r_glob != 0 or avg_r_lim != 0 else 1.0
                diff = avg_r_lim - avg_r_glob
                log_metrics_agg[f'Comparison_Avg5Ep/AvgReward_Ratio_T{t_lim}vT{t_glob}'] = ratio
                log_metrics_agg[f'Comparison_Avg5Ep/AvgReward_Diff_T{t_lim}vT{t_glob}'] = diff

            all_avg_rewards = list(team_avg_rewards_agg.values())
            log_metrics_agg['Reward_Avg5Ep/Overall_Sum'] = np.mean(all_avg_rewards) if all_avg_rewards else 0            
            log_metrics_agg['Meta/Steps_LastEp'] = episode_steps_count
           
            for tid_buf in range(NUM_TEAMS):
                log_metrics_agg[f'System/Buffer_T{tid_buf}'] = len(replay_buffers[tid_buf])
                
            logger.log_metrics_dict(log_metrics_agg, step=episode)

        # --- Checkpointing ---
        if episode > 0 and episode % save_every == 0 and optuna_trial is None: # Disable checkpointing during HPO
            print(f"--- Checkpointing Episode {episode} ---")
            chkpt_dir = f"checkpoints_{log_run_name}"
            os.makedirs(chkpt_dir, exist_ok=True)
            
            try:
                # --- Save Models and Optimizers ---
                for team_id_save in range(NUM_TEAMS):
                    if team_id_save not in team_policies: continue
                    
                    policy_save = team_policies[team_id_save]
                    optimizers_save = team_optimizers.get(team_id_save)
                    critic1_save = team_critics1.get(team_id_save)
                    critic2_save = team_critics2.get(team_id_save)
                    target_critic1_save = team_target_critics1.get(team_id_save)
                    target_critic2_save = team_target_critics2.get(team_id_save)

                    team_suffix_save = "_L" if team_id_save < 3 else "_G"
                    policy_type_idx_save = team_id_save % 3
                    policy_suffix_save = "_MAAC" if policy_type_idx_save == 0 else "_NCA" if policy_type_idx_save == 1 else "_Shared"
                    
                    save_path_base = os.path.join(chkpt_dir, f"T{team_id_save}{team_suffix_save}_ep{episode}")

                    torch.save(policy_save.state_dict(), f"{save_path_base}_policy{policy_suffix_save}.pt")
                    if critic1_save: torch.save(critic1_save.state_dict(), f"{save_path_base}_critic1.pt")
                    if critic2_save: torch.save(critic2_save.state_dict(), f"{save_path_base}_critic2.pt")
                    if target_critic1_save: torch.save(target_critic1_save.state_dict(), f"{save_path_base}_target_critic1.pt")
                    if target_critic2_save: torch.save(target_critic2_save.state_dict(), f"{save_path_base}_target_critic2.pt")

                    # --- Optimizer Saving ---
                    if optimizers_save:
                        optim_path = os.path.join(chkpt_dir, f"T{team_id_save}{team_suffix_save}_ep{episode}_policy{policy_suffix_save}_optim.pt")
                        if isinstance(policy_save, MAACPolicy):
                            # The optimizers are stored as {role_name: optimizer_object}
                            optim_state_to_save = {role: opt.state_dict() for role, opt in optimizers_save.items()}
                            torch.save(optim_state_to_save, optim_path)
                        elif isinstance(policy_save, (NCA_PINSANPolicy, SharedActorPolicy)):
                            # These policies have a single optimizer stored under the 'policy' key
                            if 'policy' in optimizers_save:
                                torch.save(optimizers_save['policy'].state_dict(), optim_path)

                # --- Save Replay Buffers ---
                # This can be slow and large. Consider if you really need to save/load it every time.
                # If disabled, training will always start with an empty buffer after loading a checkpoint.
                SAVE_BUFFERS = False 
                if SAVE_BUFFERS:
                    buffer_save_path = os.path.join(chkpt_dir, f"replay_buffers_ep{episode}.pkl")
                    buffers_to_save = {tid: buf.buffer for tid, buf in replay_buffers.items()}
                    with open(buffer_save_path, 'wb') as f_buf:
                        pickle.dump(buffers_to_save, f_buf)
                
                alpha_state = {'log_alphas': {cid: log_a.data.clone() for cid, log_a in log_alphas.items()}, 'alpha_optimizers': {cid: opt.state_dict() for cid, opt in alpha_optimizers.items()}}
                
                torch.save(alpha_state, os.path.join(chkpt_dir, f"alpha_state_ep{episode}.pt"))
                training_state = {'episode': episode, 'global_step': global_step_counter}
                torch.save(training_state, os.path.join(chkpt_dir, f"training_state_ep{episode}.pt"))

                print(f"  Checkpoint for episode {episode} saved successfully to {chkpt_dir}")
            
            except Exception as e:
                print(f"!!!!!! ERROR during checkpointing episode {episode}: {e} !!!!!!")
                traceback.print_exc()

        # --- Intermediate Profiling Output ---
        if profiler_cprofile and episode > 0 and episode % 2 == 0:  # Every 2 episodes
            print(f"\n--- Intermediate Profiling Results (Episode {episode}) ---")
            log_dir = logger.log_dir if logger and hasattr(logger, 'log_dir') else "."
            save_profiling_results(profiler_cprofile, log_dir, episode)

        # --- Garbage Collection ---
        if episode % 20 == 0 and episode > 0:
            # (Keep GC logic)
            print(f"--- Running Garbage Collection (Episode {episode}) ---"); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
       # --- Check if all teams have plateaued to end training early ---
        if plateau_manager.all_teams_frozen():
            print("\n" + "="*60)
            print(f"  ALL TEAMS HAVE PLATEAUED AT EPISODE {episode}. STOPPING TRAINING.")
            print("="*60 + "\n")
            break # Exit the main training loop

        # --- Evaluation Phase Trigger ---
        EVALUATION_INTERVAL = 10 if optuna_trial is None else 5 # Evaluate more often during HPO
        NUM_EVAL_EPISODES_PER_PHASE = 3 if optuna_trial is None else 1

        if episode > 0 and episode % EVALUATION_INTERVAL == 0:
            # --- After evaluation, check for plateau and advance curriculum ---
            # Note: `team_avg_rewards_agg` is calculated every X episodes, matching the EVAL_INTERVAL
            if plateau_manager.check_plateau(team_avg_rewards_agg):
                print("INFO: Plateau detected by PlateauManager.")
                if curriculum_manager.advance_stage():
                    new_reward_overrides = curriculum_manager.get_current_reward_overrides()
                    env.update_reward_overrides(new_reward_overrides)
                    if logger:
                        logger.log_hyperparams({'curriculum_stage': curriculum_manager.get_current_stage_name()})
                    # Optional: Reset plateau manager to give the new stage a fresh start
                    plateau_manager.reset()
            # --- End Plateau Check ---

            avg_eval_reward = run_evaluation_phase(
                eval_episode_num_global=episode,
                num_eval_episodes=NUM_EVAL_EPISODES_PER_PHASE,
                env=env,
                team_policies=team_policies,
                logger=logger,
                device=device,
                log_run_name_eval=log_run_name
            )

            # --- Optuna Pruning Integration ---
            if optuna_trial:
                optuna_trial.report(avg_eval_reward, episode)
                if optuna_trial.should_prune():
                    print(f"--- Trial pruned at episode {episode} with avg reward {avg_eval_reward:.3f} ---")
                    # Clean up GPU memory before exiting
                    del team_policies, team_critics1, team_critics2, team_target_critics1, team_target_critics2
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

    # --- End of Training ---
    if profiler_cprofile:
        profiler_cprofile.disable()

    if logger: logger.close(); print("Logger closed.")
    try: env.close(); print("Environment closed.")
    except Exception as e: print(f"Env close error: {e}")

    # Return the best average reward seen during the run
    if plateau_manager:
         best_perf = plateau_manager.get_best_average_performance()
         return np.mean(list(best_perf.values()))
    return 0.0