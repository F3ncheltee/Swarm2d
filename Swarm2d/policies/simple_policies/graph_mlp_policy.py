import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class SimpleMLP(nn.Module):
    """Simple MLP policy for graph-based observations"""
    def __init__(self, input_dim=64, hidden_dim=128, action_dim=2):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        mean = self.network(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def act(self, obs_graph):
        """Convert graph observation to action"""
        # Extract key features from graph
        if obs_graph.x.shape[0] == 0:
            return np.zeros(2), 0.0
        
        # Find ego node
        ego_mask = obs_graph.x[:, 24] > 0.5  # is_ego feature
        if not torch.any(ego_mask):
            return np.zeros(2), 0.0
        
        ego_idx = torch.where(ego_mask)[0][0]
        ego_features = obs_graph.x[ego_idx]
        
        # Simple feature aggregation: ego + mean of nearby nodes
        if obs_graph.x.shape[0] > 1:
            neighbor_features = obs_graph.x.mean(dim=0)
            features = torch.cat([ego_features[:32], neighbor_features[:32]])
        else:
            features = torch.cat([ego_features[:32], torch.zeros(32, device=ego_features.device)])
        
        mean, std = self.forward(features)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        return action.detach().cpu().numpy(), log_prob
