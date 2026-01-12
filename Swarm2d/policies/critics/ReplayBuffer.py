from typing import Optional, Tuple, List, Dict
import numpy as np
import random
from policies.critics.TITANhelpers import SumTree

class ReplayBuffer:
    """
    (V9.1 - Prioritized Sequence Sampling) Replay buffer compatible with sequence-based models like Titan.
    Tracks max_priority internally for cleaner `add` calls.
    """
    def __init__(self, capacity: int, sequence_length: int, per_alpha: float = 0.6, per_beta: float = 0.4, beta_increment: float = 0.0001):
        self.capacity = int(capacity)
        self.sequence_length = sequence_length
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.beta_increment = beta_increment
        self.e = 0.01
        self.max_priority = 1.0  # Initialize max priority

        self.tree = SumTree(self.capacity)
        self.buffer = [None] * self.capacity
        self.position = 0
        self.is_full = False
        self.continuity_lookup: Dict[Tuple[int, int, int], int] = {}

    def _get_priority(self, error):
        # The absolute error is clipped to 1 to bound priorities
        return (np.abs(error) + self.e) ** self.per_alpha

    def add(self, **kwargs): # REMOVED 'error' from signature
        """(MODIFIED) Adds a new transition with max priority, including pre-computed team views."""
        priority = self.max_priority
        
        transition_tuple = (
            kwargs['full_actor_obs_S_t'], kwargs['action_S_t'], kwargs['reward_S_tp1'],
            kwargs['done_tp1'], kwargs['role_info_S_t'], kwargs['h_actor_t'],
            kwargs['h_actor_tp1'], kwargs['full_actor_obs_S_tp1'], kwargs['episode'], kwargs['step'],
            kwargs['global_agent_idx'], kwargs['agent_policy_idx'],
            kwargs['packed_env_tensors_S_t'], # This is now the dictionary of tensors
            kwargs.get('nearby_trails_S_t'),
            kwargs.get('team_memory_map_S_t'),
            kwargs.get('team_raw_map_S_t'),
            kwargs.get('team_persistent_graph_S_t'),
            # Add the new team_live_graph for the limited critic's fovea
            kwargs.get('team_live_graph_S_t'), # <<< ADD THIS
        )
        
        # This lookup logic for managing sequences remains the same
        if self.buffer[self.position] is not None:
            old_trans = self.buffer[self.position]
            # Indices for lookup: episode (8), global_agent_idx (10), step (9)
            self.continuity_lookup.pop((old_trans[8], old_trans[10], old_trans[9]), None)
            # Remove the old sequence start from the priority tree
            self.tree.update(self.position + self.capacity - 1, 0)

        self.buffer[self.position] = transition_tuple
        self.continuity_lookup[(kwargs['episode'], kwargs['global_agent_idx'], kwargs['step'])] = self.position

        # Check if this new transition completes a full sequence
        required_start_step = kwargs['step'] - self.sequence_length + 1
        start_idx = self.continuity_lookup.get((kwargs['episode'], kwargs['global_agent_idx'], required_start_step))
        
        if start_idx is not None and self._is_sequence_still_valid(start_idx):
            # If a valid sequence can be formed starting at start_idx, add its priority
            self.tree.add(priority, start_idx)

        self.position = (self.position + 1) % self.capacity
        if self.position == 0 and not self.is_full:
            self.is_full = True

    def sample(self, batch_size: int) -> Optional[Tuple[List[List[Tuple]], np.ndarray, np.ndarray]]:
        # This method remains the same as before
        if self.tree.n_entries < batch_size:
            return None

        batch_sequences = []; tree_indices = np.empty((batch_size,), dtype=np.int32); is_weights = np.empty((batch_size, 1), dtype=np.float32)
        segment = self.tree.total() / batch_size
        self.per_beta = np.min([1., self.per_beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            (tree_idx, p, start_idx) = self.tree.get(s)
            
            if not self._is_sequence_still_valid(start_idx):
                 s_retry = random.uniform(0, self.tree.total())
                 (tree_idx, p, start_idx) = self.tree.get(s_retry)
                 if not self._is_sequence_still_valid(start_idx): continue

            sequence = [self.buffer[(start_idx + j) % self.capacity] for j in range(self.sequence_length)]
            batch_sequences.append(sequence)
            tree_indices[i] = tree_idx
            
            sampling_probabilities = p / self.tree.total()
            is_weights[i, 0] = np.power(self.tree.n_entries * sampling_probabilities, -self.per_beta)
        
        if not batch_sequences: return None
        is_weights /= is_weights.max()
        return batch_sequences, tree_indices, is_weights

    def update_priorities(self, tree_indices, errors):
        # This method remains the same as before
        for i, idx in enumerate(tree_indices):
            p = self._get_priority(errors[i])
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p) # Update max priority seen
            
    def _is_sequence_still_valid(self, start_idx: int) -> bool:
        # This method remains the same as before
        first_trans = self.buffer[start_idx]
        if first_trans is None: return False
        ep, agent_id, start_step = first_trans[8], first_trans[10], first_trans[9]
        for i in range(1, self.sequence_length):
            next_idx = (start_idx + i) % self.capacity
            next_trans = self.buffer[next_idx]
            if next_trans is None or next_trans[8] != ep or next_trans[10] != agent_id or next_trans[9] != start_step + i:
                return False
        return True

    def __len__(self):
        return self.tree.n_entries
