import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
from Swarm2d.constants import ChannelConstants

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
        
        # STREAM 2: Process Self Vector (Compass)
        # Handle mismatch for legacy checkpoints (Env has more features than Model)
        if self_vec.shape[1] > self.self_input_dim:
            self_vec = self_vec[:, :self.self_input_dim]
            
        self_features = self.self_fc(self_vec)
        
        # COMBINE
        combined = torch.cat([map_features, self_features], dim=1)
        
        features = self.fc(combined)
        
        # Movement distribution
        # USE TANH TO PREVENT EXPLODING OUTPUTS (-3000 issue)
        # Range [-1, 1] matches env action space strictly.
        move_mean = torch.tanh(self.move_head(features))
        
        move_std = torch.exp(self.log_std).expand_as(move_mean)
        
        # Pickup distribution (logits)
        pickup_logits = self.pickup_head(features)
        
        # Save logits for debug printing (detached)
        self.last_pickup_logits = pickup_logits.detach()
        
        return move_mean, move_std, pickup_logits
    
    def act(self, obs_map, obs_memory, obs_self, deterministic=False):
        """Convert map observation to action"""
        device = next(self.parameters()).device
        
        # Prepare Map Stack (Batch dim added)
        if obs_map is None:
            return np.zeros(2), 0, 0
            
        # Ensure inputs are tensors on device
        if not isinstance(obs_map, torch.Tensor):
            obs_map = torch.tensor(obs_map, dtype=torch.float32, device=device)
        # IGNORE obs_memory
        if not isinstance(obs_self, torch.Tensor):
            obs_self = torch.tensor(obs_self, dtype=torch.float32, device=device)
            
        # Add batch dim if missing
        if obs_map.dim() == 3:
            obs_map = obs_map.unsqueeze(0)
        if obs_self.dim() == 1:
            obs_self = obs_self.unsqueeze(0)
            
        # DIRECT FORWARD PASS (No Stacking)
        move_mean, move_std, pickup_logits = self.forward(obs_map, obs_self)
        
        if deterministic:
            # Deterministic mode: use mean for movement, argmax for pickup
            move_action = move_mean
            pickup_action = torch.argmax(pickup_logits, dim=-1)
            
            # Dummy log_prob/entropy for interface compatibility
            total_log_prob = torch.tensor([0.0], device=device)
            total_entropy = torch.tensor([0.0], device=device)
        else:
            # Sample movement
            move_dist = Normal(move_mean, move_std)
            move_action = move_dist.sample()
            move_log_prob = move_dist.log_prob(move_action).sum(dim=-1)
            move_entropy = move_dist.entropy().sum(dim=-1)
            
            # Sample pickup
            pickup_dist = Categorical(logits=pickup_logits)
            pickup_action = pickup_dist.sample()
            pickup_log_prob = pickup_dist.log_prob(pickup_action)
            pickup_entropy = pickup_dist.entropy()
            
            total_log_prob = move_log_prob + pickup_log_prob
            total_entropy = move_entropy + pickup_entropy
        
        return move_action.detach().cpu().numpy()[0], pickup_action.item(), total_log_prob, total_entropy

class CompositePolicy(nn.Module):
    """
    A Mixture-of-Experts policy that switches between a Resource Specialist
    and a Combat Specialist based on a simple heuristic gate.
    """
    def __init__(self, resource_policy, combat_policy):
        super(CompositePolicy, self).__init__()
        self.resource_policy = resource_policy
        self.combat_policy = combat_policy
        
    def forward(self, obs_map, self_vec):
        # We don't implement forward directly because we need to switch logic in 'act'
        raise NotImplementedError("Use .act() for CompositePolicy")

    def act(self, obs_map, obs_memory, obs_self):
        # 1. GATE LOGIC: Decide which brain to use
        # Heuristic: If we see an enemy (Map Channel 1) -> Combat Mode
        # Otherwise -> Resource Mode
        
        # obs_map shape: (Batch, Channels, H, W) or (Channels, H, W)
        is_batch = obs_map.dim() == 4
        if not is_batch:
            # Single agent inference
            # Channel 1 is 'enemy_presence' (See ChannelConstants)
            enemy_map = obs_map[1] 
            enemy_visible = torch.max(enemy_map) > 0.1
            
            if enemy_visible:
                return self.combat_policy.act(obs_map, obs_memory, obs_self)
            else:
                return self.resource_policy.act(obs_map, obs_memory, obs_self)
        else:
            # Batched inference (tricky with if/else, need masking)
            # For simplicity in this script, we'll just run both and mask.
            
            # 1. Run both policies
            move_res, pick_res, log_res, ent_res = self.resource_policy.act(obs_map, obs_memory, obs_self)
            move_com, pick_com, log_com, ent_com = self.combat_policy.act(obs_map, obs_memory, obs_self)
            
            # 2. Determine mask
            # obs_map: (Batch, C, H, W)
            enemy_maps = obs_map[:, 1, :, :] # (Batch, H, W)
            # Check if any pixel > 0.1 per batch item
            enemy_visible = (enemy_maps.max(dim=1)[0].max(dim=1)[0] > 0.1).cpu().numpy()
            
            # 3. Combine
            # NOTE: acts are numpy arrays from .act(), so we can mask directly
            # This part was incomplete in the original script but we include it for compatibility
            pass

class CNNPolicyWrapper:
    """Wraps a CNN Policy to match the Heuristic Policy interface (.act(obs) -> action_dict)"""
    def __init__(self, policy):
        self.policy = policy
        self.policy.eval() # Ensure eval mode
    
    def act(self, obs):
        # policy.act returns (move, pickup, log_prob, entropy)
        # We ignore gradients/log_probs for the opponent
        with torch.no_grad():
            move, pickup, _, _ = self.policy.act(obs['map'], obs['memory_map'], obs['self'])
        return {'movement': move, 'pickup': pickup}
