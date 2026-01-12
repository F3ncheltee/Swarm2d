import torch
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from PyQt6.QtWidgets import QFileDialog

# --- SimpleCNNMapPolicy Definition (Copied for standalone usage) ---
class SimpleCNNMapPolicy(nn.Module):
    """
    CNN-based policy for MAP-based observations + Self Vector.
    This is the agent's 'Brain'.
    MODIFIED: IGNORES MEMORY MAP INPUTS FOR FAST TRAINING.
    """
    def __init__(self, map_channels, map_size, memory_channels, memory_size, self_dim, hidden_dim=256):
        super(SimpleCNNMapPolicy, self).__init__()
        self.map_size = map_size
        
        # Store input dim for robust loading/slicing
        self.self_input_dim = self_dim
        
        # WE IGNORE MEMORY CHANNELS NOW
        self.total_channels = map_channels
        
        # CNN for Map Processing (The "Eyes")
        # Input: (Batch, Map_Channels_Only, 32, 32)
        self.conv1 = nn.Conv2d(self.total_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after pooling
        # 32x32 -> (pool) -> 16x16 -> (pool) -> 8x8
        self.flat_map_size = 64 * 8 * 8
        
        # 1. Map Processing Stream (The "Eyes")
        self.map_fc = nn.Sequential(
            nn.Linear(self.flat_map_size, 256),
            nn.ReLU()
        )
        
        # 2. Self Vector Processing Stream (The "Compass")
        # We give this a dedicated layer so it doesn't get drowned out by the 4096 map features
        self.self_fc = nn.Sequential(
            nn.Linear(self_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Combined Decision Stream
        # Input: Map Features (256) + Self Features (64)
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Head for movement (x, y) - Continuous
        self.move_head = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        
        # Head for pickup (0=None, 1=Pickup/Grapple, 2=Drop/Break) - Discrete
        self.pickup_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, obs_map, self_vec):
        # Map processing (IGNORE MEMORY STACK)
        x = F.relu(self.conv1(obs_map))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        
        x = x.flatten(start_dim=1)
        
        # STREAM 1: Process Map
        map_features = self.map_fc(x)
        
        # STREAM 2: Process Self Vector
        # Ensure self_vec matches expected dim
        if self_vec.shape[1] > self.self_input_dim:
            self_vec = self_vec[:, :self.self_input_dim]
        elif self_vec.shape[1] < self.self_input_dim:
            padding = torch.zeros(self_vec.shape[0], self.self_input_dim - self_vec.shape[1], device=self_vec.device)
            self_vec = torch.cat([self_vec, padding], dim=1)
            
        self_features = self.self_fc(self_vec)
        
        # COMBINE
        combined = torch.cat([map_features, self_features], dim=1)
        
        # DECISION
        x = self.fc(combined)
        
        # Outputs
        move_mean = torch.tanh(self.move_head(x)) # Range [-1, 1]
        pickup_logits = self.pickup_head(x)
        
        return move_mean, pickup_logits

    def act(self, obs_map, memory_map, self_vec, deterministic=False):
        """
        Selects an action for the given observation.
        """
        # Ensure inputs are tensors and on correct device
        if not isinstance(obs_map, torch.Tensor):
            obs_map = torch.tensor(obs_map, dtype=torch.float32)
        if not isinstance(self_vec, torch.Tensor):
            self_vec = torch.tensor(self_vec, dtype=torch.float32)
            
        # Add batch dim if needed
        if obs_map.dim() == 3: obs_map = obs_map.unsqueeze(0)
        if self_vec.dim() == 1: self_vec = self_vec.unsqueeze(0)
        
        device = next(self.parameters()).device
        obs_map = obs_map.to(device)
        self_vec = self_vec.to(device)
        
        with torch.no_grad():
            move_mean, pickup_logits = self(obs_map, self_vec)
            
            if deterministic:
                move = move_mean
                pickup = torch.argmax(pickup_logits, dim=1)
            else:
                # Movement (Gaussian)
                std = torch.exp(self.log_std)
                dist_move = torch.distributions.Normal(move_mean, std)
                move = dist_move.sample()
                move = torch.clamp(move, -1.0, 1.0) # Clip
                
                # Pickup (Categorical)
                dist_pickup = torch.distributions.Categorical(logits=pickup_logits)
                pickup = dist_pickup.sample()
                
            # Log probs (optional, skipped for inference)
            
        return move.cpu().numpy()[0], pickup.cpu().item(), None, None

# --- Policy Wrapper for GUI ---
class TrainedCnnPolicy:
    def __init__(self, env, team_id):
        self.env = env
        self.team_id = team_id
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing TrainedCnnPolicy for Team {team_id}...")
        
        # Try to find path in environment (passed from GUI)
        file_path = None
        if hasattr(self.env, 'team_policy_paths'):
            file_path = self.env.team_policy_paths.get(self.team_id)
            if file_path:
                # Handle string keys if they were passed that way
                pass
            else:
                # Try string key
                file_path = self.env.team_policy_paths.get(str(self.team_id))

        if file_path:
            self.load_checkpoint(file_path)
        else:
            print(f"No checkpoint path found for Team {team_id}. Please use the 'Load Model' button in the GUI.")
            # Note: We CANNOT open a QFileDialog here because this runs in a background thread.

    def load_checkpoint(self, path):
        try:
            print(f"Loading checkpoint from: {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # --- Initialize Policy Architecture ---
            # We need to match the architecture used in simple_rl_training_MAP_ONLY.py
            # Parameters should ideally come from env metadata or config, but we'll use defaults 
            # consistent with the training script if not available.
            
            map_channels = self.env.raw_ch_count # Should be 8
            map_size = self.env.raw_map_grid_size # Should be 32
            self_dim = self.env.env_self_obs_dim # Should be 10 or similar
            
            self.policy = SimpleCNNMapPolicy(
                map_channels=map_channels,
                map_size=map_size,
                memory_channels=0, # Ignored by this policy class
                memory_size=0,
                self_dim=self_dim
            ).to(self.device)
            
            # Load weights
            if 'policy' in checkpoint:
                self.policy.load_state_dict(checkpoint['policy'])
            else:
                self.policy.load_state_dict(checkpoint)
                
            self.policy.eval()
            print(f"Successfully loaded policy for Team {self.team_id}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None

    def get_actions(self, obs, agent_ids):
        """
        Get actions for a list of agents.
        """
        if not self.policy:
            return [{'movement': np.zeros(2), 'pickup': 0} for _ in agent_ids]
            
        actions = []
        for i in agent_ids:
            # Prepare observations
            agent_obs = obs[i]
            
            # Extract components
            # Note: The training script passes (map, memory_map, self). 
            # Even though memory_map is ignored by the policy logic, we pass it to match the signature.
            obs_map = agent_obs['map']
            memory_map = agent_obs.get('memory_map', None) # Might be None if not generated
            self_vec = agent_obs['self']
            
            # Run inference
            move, pickup, _, _ = self.policy.act(obs_map, memory_map, self_vec, deterministic=True)
            
            actions.append({'movement': move, 'pickup': pickup})
            
        return actions

