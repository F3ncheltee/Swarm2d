import torch
import numpy as np
import random
from typing import List, Dict, Tuple
import logging
import os
import sys
from tqdm import tqdm


from constants import (
    AGENT_BASE_STRENGTH,
    AGENT_BASE_SPEED,
    OBS_RADIUS,
    AGENT_RADIUS,
    AGENT_MAX_ENERGY,
    AGENT_MAX_HEALTH
)


def set_seeds(seed_value=42):
    """Sets all random seeds for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def setup_logging(save_dir: str):
    """Sets up logging to file and console."""
    log_file = os.path.join(save_dir, 'training.log')
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler(sys.stdout)      # Print to console
        ]
    )
    # Redirect tqdm's output to the logger to avoid clashing with console handler
    # Note: This might not be perfect for all setups, but is a common approach.
    tqdm.pandas(file=open(os.devnull, 'w'))

def soft_update_targets(online_c1, target_c1, online_c2, target_c2, tau):
    """ Helper function to soft-update a pair of target critics."""
    if online_c1 and target_c1:
        for target_param, online_param in zip(target_c1.parameters(), online_c1.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    if online_c2 and target_c2:
        for target_param, online_param in zip(target_c2.parameters(), online_c2.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def get_innate_properties_for_episode(env_agents_list: List[Dict]) -> Dict[int, Dict[str, float]]:
    """Extracts innate properties of agents after an environment reset."""
    props = {}
    for agent_data in env_agents_list:
        if agent_data: # Agent data might be None if not fully populated yet
            agent_id = agent_data.get('id')
            if agent_id is not None: # Ensure agent_id exists
                props[agent_id] = {
                    'strength': agent_data.get('strength', AGENT_BASE_STRENGTH),
                    'speed': agent_data.get('speed', AGENT_BASE_SPEED),
                    'obs_radius': agent_data.get('obs_radius', OBS_RADIUS),
                    'agent_radius': agent_data.get('agent_radius', AGENT_RADIUS),
                    'max_energy': agent_data.get('max_energy', AGENT_MAX_ENERGY),
                    'max_health': agent_data.get('max_health', AGENT_MAX_HEALTH)
                }
    return props

def assign_conceptual_teams_for_episode(
    env_agents_list: List[Dict],
    num_total_teams_conceptual: int, # Should be NUM_TEAMS
    agents_per_conceptual_team: int  # Should be NUM_CONCEPTUAL_AGENTS_PER_TEAM
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    agent_id_to_conceptual_team = {} # Maps global_env_idx to conceptual_team_id
    conceptual_team_indices = {team_idx: [] for team_idx in range(num_total_teams_conceptual)}
    conceptual_team_counts = {team_idx: 0 for team_idx in range(num_total_teams_conceptual)}

    alive_agents_info = []
    for global_idx, agent_data in enumerate(env_agents_list):
        if agent_data and agent_data.get('alive', True):
            alive_agents_info.append({
                'global_idx': global_idx,
                'env_team_id': agent_data.get('team', 0),
                'id': agent_data.get('id', -1)
            })
    alive_agents_info.sort(key=lambda x: x['env_team_id'])

    for agent_info in alive_agents_info:
        global_idx = agent_info['global_idx']
        env_team_id = agent_info['env_team_id']
        base_limited_conceptual_team = env_team_id % 3
        base_global_conceptual_team = base_limited_conceptual_team + 3
        
        assigned_conceptual_team = base_limited_conceptual_team
        # Simplified assignment: try to balance between the L/G pair
        if conceptual_team_counts.get(base_limited_conceptual_team, 0) > conceptual_team_counts.get(base_global_conceptual_team, 0) and \
           conceptual_team_counts.get(base_global_conceptual_team, 0) < agents_per_conceptual_team:
            assigned_conceptual_team = base_global_conceptual_team
        elif conceptual_team_counts.get(base_limited_conceptual_team, 0) == conceptual_team_counts.get(base_global_conceptual_team, 0):
            assigned_conceptual_team = random.choice([base_limited_conceptual_team, base_global_conceptual_team])
        
        # Fallback if preferred team is full
        if conceptual_team_counts.get(assigned_conceptual_team, 0) >= agents_per_conceptual_team:
            other_team = base_global_conceptual_team if assigned_conceptual_team == base_limited_conceptual_team else base_limited_conceptual_team
            if conceptual_team_counts.get(other_team, 0) < agents_per_conceptual_team:
                assigned_conceptual_team = other_team
            else: # Both full, assign to original preference (this might overfill)
                assigned_conceptual_team = base_limited_conceptual_team if random.random() < 0.5 else base_global_conceptual_team


        agent_id_to_conceptual_team[global_idx] = assigned_conceptual_team
        conceptual_team_indices.setdefault(assigned_conceptual_team, []).append(global_idx)
        conceptual_team_counts[assigned_conceptual_team] = conceptual_team_counts.get(assigned_conceptual_team, 0) + 1
            
    return agent_id_to_conceptual_team, conceptual_team_indices
