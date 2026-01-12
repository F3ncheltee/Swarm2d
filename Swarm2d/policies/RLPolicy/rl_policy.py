import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RLPolicy(nn.Module):
    def __init__(self, obs_dim, movement_dim, pickup_dim):
        super(RLPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.movement_head = nn.Linear(128, movement_dim)
        self.pickup_head = nn.Linear(128, pickup_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        features = self.net(obs)
        movement = torch.tanh(self.movement_head(features)) # Tanh for -1 to 1 range
        pickup_logits = self.pickup_head(features)
        return movement, pickup_logits

    def act(self, obs):
        movement, pickup_logits = self.forward(obs)
        pickup_action = torch.argmax(pickup_logits, dim=-1)
        
        return {
            "movement": movement.cpu().detach().numpy(),
            "pickup": pickup_action.cpu().detach().numpy()
        }

    def update(self, batch):
        pass
