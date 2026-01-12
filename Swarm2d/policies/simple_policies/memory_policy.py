import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class SimpleCNNMemoryPolicy(nn.Module):
    """
    CNN-based policy for MEMORY-based observations + Self Vector.
    This is the agent's 'Brain'.
    MODIFIED: IGNORES RAW MAP, USES ONLY MEMORY MAP.
    """
    def __init__(self, map_channels, map_size, memory_channels, memory_size, self_dim, hidden_dim=256):
        super(SimpleCNNMemoryPolicy, self).__init__()
        self.map_size = memory_size # Memory map size (usually same or larger, but defined by input)
        
        # WE USE MEMORY CHANNELS NOW
        self.total_channels = memory_channels
        
        # CNN for Memory Map Processing (The "Memory Reader")
        # Input: (Batch, Memory_Channels, 64, 64) -> Assuming memory map is larger or same
        self.conv1 = nn.Conv2d(self.total_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Extra pool for larger memory map (if 64x64)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_final = nn.MaxPool2d(2, 2)
        
        # Final size calculation:
        # 64 -> 32 -> 16 -> 8
        self.flat_map_size = 64 * 8 * 8
        
        # 1. Map Processing Stream
        self.map_fc = nn.Sequential(
            nn.Linear(self.flat_map_size, 256),
            nn.ReLU()
        )
        
        # 2. Self Vector Processing Stream
        self.self_fc = nn.Sequential(
            nn.Linear(self_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Combined Decision Stream
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
        
    def forward(self, obs_memory, self_vec):
        # Memory Map processing
        x = F.relu(self.conv1(obs_memory))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool_final(x) # Extra pool for larger memory map
        
        x = x.flatten(start_dim=1)
        
        # STREAM 1: Process Map
        map_features = self.map_fc(x)
        
        # STREAM 2: Process Self Vector (Compass)
        self_features = self.self_fc(self_vec)
        
        # COMBINE
        combined = torch.cat([map_features, self_features], dim=1)
        
        features = self.fc(combined)
        
        # Movement distribution
        move_mean = torch.tanh(self.move_head(features))
        move_std = torch.exp(self.log_std).expand_as(move_mean)
        
        # Pickup distribution (logits)
        pickup_logits = self.pickup_head(features)
        
        return move_mean, move_std, pickup_logits
    
    def act(self, obs_map, obs_memory, obs_self):
        """Convert MEMORY observation to action"""
        device = next(self.parameters()).device
        
        # Prepare Memory Stack
        if obs_memory is None:
             return np.zeros(2), 0, 0, 0
            
        # Ensure inputs are tensors on device
        if not isinstance(obs_memory, torch.Tensor):
            obs_memory = torch.tensor(obs_memory, dtype=torch.float32, device=device)
        if not isinstance(obs_self, torch.Tensor):
            obs_self = torch.tensor(obs_self, dtype=torch.float32, device=device)
            
        # Add batch dim if missing
        if obs_memory.dim() == 3:
            obs_memory = obs_memory.unsqueeze(0)
        if obs_self.dim() == 1:
            obs_self = obs_self.unsqueeze(0)
            
        # FORWARD PASS using MEMORY ONLY
        move_mean, move_std, pickup_logits = self.forward(obs_memory, obs_self)
        
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
