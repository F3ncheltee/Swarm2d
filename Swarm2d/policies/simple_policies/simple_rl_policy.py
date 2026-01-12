import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class SimpleRLPolicy(nn.Module):
    def __init__(self, obs_dim, movement_dim, pickup_dim, lr=1e-4, gamma=0.99):
        super(SimpleRLPolicy, self).__init__()
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.movement_head = nn.Linear(128, movement_dim)
        self.pickup_head = nn.Linear(128, pickup_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        # If obs is a batch, add a batch dimension if it's not there
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self.net(obs)
        movement = torch.tanh(self.movement_head(features))
        pickup_logits = self.pickup_head(features)
        return movement, pickup_logits

    def act(self, obs):
        movement, pickup_logits = self.forward(obs)
        
        # For pickup, sample from the categorical distribution
        pickup_prob = torch.softmax(pickup_logits, dim=-1)
        pickup_dist = Categorical(pickup_prob)
        pickup_action = pickup_dist.sample()

        # Get log probability for the REINFORCE update
        log_prob = pickup_dist.log_prob(pickup_action)

        action_dict = {
            "movement": movement.cpu().detach().numpy().squeeze(),
            "pickup": pickup_action.cpu().item()
        }
        return action_dict, log_prob

    def update(self, trajectories):
        """
        Update the policy using the REINFORCE algorithm.
        Args:
            trajectories (list): A list of trajectories. Each trajectory is a list 
                                 of (log_prob, reward) tuples for one agent.
        """
        policy_loss = []
        all_returns = []

        for trajectory in trajectories:
            if not trajectory:
                continue

            log_probs = [lp for lp, r in trajectory]
            rewards = [r for lp, r in trajectory]

            # Calculate discounted returns
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            all_returns.extend(returns)
            returns = torch.tensor(returns, device=log_probs[0].device)
            
            # The REINFORCE loss for this trajectory
            for log_prob, R_t in zip(log_probs, returns):
                policy_loss.append(-log_prob * R_t)
        
        if not policy_loss:
            return 0.0

        # Normalize all returns across all agents for this episode
        all_returns_tensor = torch.tensor(all_returns)
        all_returns_tensor = (all_returns_tensor - all_returns_tensor.mean()) / (all_returns_tensor.std() + 1e-9)

        # Re-calculate loss with normalized returns
        policy_loss = []
        return_idx = 0
        for trajectory in trajectories:
            if not trajectory:
                continue
            
            log_probs = [lp for lp, r in trajectory]
            for log_prob in log_probs:
                policy_loss.append(-log_prob * all_returns_tensor[return_idx])
                return_idx += 1

        self.optimizer.zero_grad()
        policy_loss_tensor = torch.stack(policy_loss).sum()
        policy_loss_tensor.backward()
        self.optimizer.step()

        return policy_loss_tensor.item()
