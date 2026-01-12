"""
Map-Based Heuristic Policy - Optimized for Smart Resource Gathering
"""
import numpy as np
import torch
from Swarm2d.constants import SELF_OBS_MAP

class MapHeuristic:
    def __init__(self, action_space):
        """
        Smart heuristic that uses map observations.
        - Moves towards resources if seen
        - Returns to hive if carrying
        - Maintains exploration momentum
        """
        self.action_space = action_space
        self.state = {} # Store state per agent (momentum)
        
    def act(self, obs: dict) -> dict:
        """
        Determines action based on self observation with momentum.
        """
        self_obs = obs['self']
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(self_obs):
            self_obs = self_obs.cpu().numpy()
            
        # Extract features using proper indices
        # 'rel_res_x_norm': 4, 'rel_res_y_norm': 5
        # 'rel_hive_x_norm': 6, 'rel_hive_y_norm': 7
        # 'is_carrying': 8
        # 'agent_id': 23
        
        agent_id = int(self_obs[SELF_OBS_MAP['agent_id']]) if len(self_obs) > 23 else 0
        is_carrying = self_obs[SELF_OBS_MAP['is_carrying']] > 0.5
        
        rel_res_x = self_obs[SELF_OBS_MAP['rel_res_x_norm']]
        rel_res_y = self_obs[SELF_OBS_MAP['rel_res_y_norm']]
        
        # Check for distance channel if it exists (Added in V10)
        res_dist = 1.0
        if 'res_dist_norm' in SELF_OBS_MAP:
             res_dist = self_obs[SELF_OBS_MAP['res_dist_norm']]
        
        rel_hive_x = self_obs[SELF_OBS_MAP['rel_hive_x_norm']]
        rel_hive_y = self_obs[SELF_OBS_MAP['rel_hive_y_norm']]
        
        # Initialize state for this agent if needed
        if agent_id not in self.state:
            self.state[agent_id] = {
                'momentum_dir': np.random.randn(2),
                'momentum_timer': 0
            }
            
        # 1. LOGIC: If Carrying -> Go Home
        if is_carrying:
            # Vector to hive
            vec_mag = np.sqrt(rel_hive_x**2 + rel_hive_y**2) + 1e-6
            direction = np.array([rel_hive_x / vec_mag, rel_hive_y / vec_mag])
            # Add slight noise to avoid getting stuck
            direction += np.random.normal(0, 0.1, 2)
            return {"movement": direction, "pickup": 0} # 0 = No pickup/drop action (just hold)
            
        # 2. LOGIC: If See Resource -> Go Get It
        # Check if resource vector is non-zero (meaning a resource is in range)
        if abs(rel_res_x) > 1e-3 or abs(rel_res_y) > 1e-3:
            vec_mag = np.sqrt(rel_res_x**2 + rel_res_y**2) + 1e-6
            direction = np.array([rel_res_x / vec_mag, rel_res_y / vec_mag])
            
            # If very close, try to pickup
            pickup = 0
            
            # Use explicit distance channel if available (more reliable)
            if 'res_dist_norm' in SELF_OBS_MAP:
                if res_dist < 0.02: 
                     pickup = 1
            else:
                # FIX: With Unit Vectors, vec_mag is always 1.0. 
                # We must use the MAP to check for close proximity.
                # Check 3x3 center of Resource Channel (Index 2)
                try:
                    map_obs = obs['map']
                    if torch.is_tensor(map_obs): map_obs = map_obs.cpu().numpy()
                    
                    # Assuming Channel 2 is Resource (Standard)
                    # Map is usually 32x32. Center is 16.
                    c = map_obs.shape[1] // 2
                    # Check 3x3 window around center
                    center_slice = map_obs[2, c-1:c+2, c-1:c+2] 
                    if np.max(center_slice) > 0.1:
                        pickup = 1
                except:
                    # Fallback if map fails: Random pickup if vector exists
                    # (Blind luck is better than broken logic)
                    if np.random.random() < 0.1:
                        pickup = 1
                
            return {"movement": direction, "pickup": pickup}
            
        # 3. LOGIC: Explore with Momentum (avoid jitter)
        state = self.state[agent_id]
        if state['momentum_timer'] <= 0:
            # Pick new random direction
            angle = np.random.uniform(0, 2 * np.pi)
            state['momentum_dir'] = np.array([np.cos(angle), np.sin(angle)])
            state['momentum_timer'] = np.random.randint(20, 50) # Stick to it for 20-50 steps
            
        state['momentum_timer'] -= 1
        return {"movement": state['momentum_dir'], "pickup": 0}

    def reset_state(self):
        self.state = {}
